import os, json, time, threading, datetime

from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string, g
from functools import wraps
import requests
import base64

app = Flask(__name__)

# ─── Config from Environment ──────────────────────────────────────────────────
RAW_KEYS   = os.environ.get("GEMINI_API_KEYS", "").split(",")
RAW_MODELS = os.environ.get("GEMINI_MODELS", "gemini-2.0-flash-lite:30:1500,gemini-2.0-flash:15:1500,gemini-2.5-flash-lite:15:1000,gemini-2.5-flash:10:250,gemini-1.5-flash-latest:15:1500,gemma-4-26b-a4b-it:15:50,gemma-4-31b-it:15:50,gemini-1.5-pro-latest:2:50").split(",")

API_KEYS = list(dict.fromkeys([k.strip() for k in RAW_KEYS if k.strip()]))

MODELS = []
for m in RAW_MODELS:
    if not m.strip(): continue
    parts = m.split(":")
    name = parts[0].strip()
    rpm  = int(parts[1].strip()) if len(parts) > 1 else 15
    rpd  = int(parts[2].strip()) if len(parts) > 2 else (1500 if "flash" in name else 50)
    MODELS.append({"name": name, "rpm": rpm, "rpd": rpd})

if not MODELS:
    MODELS = [{"name": "gemini-2.0-flash-lite", "rpm": 30, "rpd": 1500}]

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

# ─── State ────────────────────────────────────────────────────────────────────
state_lock = threading.RLock()

# RPM window: (key, model_name) -> deque of monotonic timestamps
rpm_window: dict = {}

# RPD counts: (key, model_name) -> int
rpd_count: dict = {}

# RPM cooldown (429 penalty): (key, model_name) -> monotonic timestamp until blocked
rpm_cooldown: dict = {}

# RPD daily penalty: key -> UTC timestamp when it unlocks (midnight IST)
key_daily_penalty: dict = {}

# Last RPD reset date (IST)
_last_rpd_reset = datetime.datetime.now(IST).strftime("%Y-%m-%d")

def _today_ist():
    return datetime.datetime.now(IST).strftime("%Y-%m-%d")

def _maybe_reset_rpd():
    """Reset RPD counts at midnight IST. Call inside state_lock."""
    global _last_rpd_reset
    today = _today_ist()
    if today != _last_rpd_reset:
        print(f"[RPD RESET] Midnight IST - clearing all RPD counts (was {_last_rpd_reset})")
        rpd_count.clear()
        # Also clear daily penalties since a new day started
        key_daily_penalty.clear()
        _last_rpd_reset = today

def _get_key(key, model_name, d, default_factory):
    k = (key, model_name)
    if k not in d:
        d[k] = default_factory()
    return d[k]

def _prune_rpm(key, model_name):
    window = _get_key(key, model_name, rpm_window, deque)
    cutoff = time.monotonic() - 60.0
    while window and window[0] < cutoff:
        window.popleft()
    return window

def _rpm_available(key, model_name, rpm_limit):
    window = _prune_rpm(key, model_name)
    return len(window) < rpm_limit

def _rpd_available(key, model_name, rpd_limit):
    used = _get_key(key, model_name, rpd_count, lambda: 0)
    if isinstance(used, deque):  # safety
        used = 0
    return used < rpd_limit

def _is_rpm_cooldown(key, model_name):
    until = rpm_cooldown.get((key, model_name), 0)
    return time.monotonic() < until

def _is_daily_penalized(key):
    until = key_daily_penalty.get(key, 0)
    return time.time() < until

def _record_request(key, model_name):
    """Call BEFORE sending API request."""
    _prune_rpm(key, model_name)
    rpm_window[(key, model_name)].append(time.monotonic())
    k = (key, model_name)
    rpd_count[k] = rpd_count.get(k, 0) + 1

def _apply_rpm_cooldown(key, model_name, seconds=62):
    """Apply 62s cooldown to this key+model pair after 429."""
    rpm_cooldown[(key, model_name)] = time.monotonic() + seconds
    print(f"[COOLDOWN] key=…{key[-6:]} model={model_name} blocked for {seconds}s")

def _apply_daily_penalty(key):
    """Block this key until midnight IST after daily RPD hit."""
    now_ist = datetime.datetime.now(IST)
    next_midnight = (now_ist + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=5, microsecond=0)
    unlock_ts = next_midnight.astimezone(datetime.timezone.utc).timestamp()
    key_daily_penalty[key] = unlock_ts
    mins = round((unlock_ts - time.time()) / 60)
    print(f"[DAILY LIMIT] key=…{key[-6:]} penalized until midnight IST (~{mins} mins)")

# ─── Global Metrics ───────────────────────────────────────────────────────────
metrics = {
    "total_incoming_requests": 0,
    "successful_api_calls": 0,
    "rate_limit_hits": 0,
    "rpm_cooldowns_applied": 0,
    "daily_limits_hit": 0,
    "failed_requests": 0,
    "usage_by_key": {}
}

# ─── Dynamic model list (refreshed daily from API) ────────────────────────────
dynamic_models_lock = threading.Lock()
DYNAMIC_MODELS = []
OPENAI_MODELS_LIST = []

def refresh_models_loop():
    global DYNAMIC_MODELS, OPENAI_MODELS_LIST
    while True:
        if API_KEYS:
            for key in API_KEYS:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json().get("models", [])
                        new_rr, new_openai = [], []
                        now = int(time.time())
                        for m in data:
                            name = m["name"].replace("models/", "")
                            new_openai.append({"id": name, "object": "model", "created": now, "owned_by": "google"})
                            methods = m.get("supportedGenerationMethods", [])
                            skip = any(x in name.lower() for x in ["embedding","tts","image","transcribe","robotics","aqa"])
                            if "generateContent" in methods and not skip:
                                rpm = 30 if "flash-lite" in name else (15 if "flash" in name else 2)
                                rpd = 1500 if "flash-lite" in name else (1500 if "flash" in name else 50)
                                new_rr.append({"name": name, "rpm": rpm, "rpd": rpd})
                        for extra in ["dall-e-3", "whisper-1", "tts-1"]:
                            if not any(x["id"] == extra for x in new_openai):
                                new_openai.append({"id": extra, "object": "model", "created": now, "owned_by": "google"})
                        with dynamic_models_lock:
                            if new_rr:
                                DYNAMIC_MODELS = new_rr
                                OPENAI_MODELS_LIST = new_openai
                        break
                except Exception as e:
                    print(f"[MODEL REFRESH ERROR] {e}")
        time.sleep(86400 if DYNAMIC_MODELS else 60)

threading.Thread(target=refresh_models_loop, daemon=True).start()

def get_active_models():
    with dynamic_models_lock:
        return DYNAMIC_MODELS if DYNAMIC_MODELS else MODELS

# ─── Core: Smart Combo Picker ─────────────────────────────────────────────────
current_key_idx   = 0
current_model_idx = 0

def get_best_combo(requested_model: str):
    """
    Two modes:

    1. POOL MODE (requested_model is empty / generic alias):
       - Use ENV models as pool — try all key × model combos
       - High-RPM models first, keys rotate per model
       - This is the "all text models" path

    2. SPECIFIC MODEL MODE (user sent a real model name):
       - ONLY that model is used — no fallback to other models
       - But ALL keys are tried (rotate keys on 429)
       - If model is not in ENV pool, still use it with safe default limits

    Both modes filter out: daily-penalized keys, RPM cooldowns,
    RPD exhausted pairs, and RPM-full pairs.
    """
    with state_lock:
        _maybe_reset_rpd()
        active_models = get_active_models()
        env_model_names = {m["name"] for m in active_models}

        if not API_KEYS:
            return []

        # Determine mode
        GENERIC_ALIASES = {"gemini-pro","auto","default","round-robin",
                           "gemini-working-model",""}
        is_pool_mode = requested_model.lower() in GENERIC_ALIASES

        combos = []

        if is_pool_mode:
            # ── POOL MODE: all key × model, sorted by highest RPM first ──
            sorted_models = sorted(active_models, key=lambda m: -m["rpm"])
            K, M = len(API_KEYS), len(sorted_models)
            if M == 0:
                return []
            start = (current_key_idx % K) * M + (current_model_idx % M)
            for i in range(K * M):
                idx   = (start + i) % (K * M)
                k_i   = idx // M
                m_i   = idx % M
                key   = API_KEYS[k_i]
                model = sorted_models[m_i]
                combos.append((k_i, m_i, key, model))

        else:
            # ── SPECIFIC MODEL MODE: only this model, rotate keys ──
            # Look up limits from ENV pool; fall back to safe defaults
            model_dict = next(
                (m for m in active_models if m["name"] == requested_model), None
            )
            if model_dict is None:
                # Model not in ENV — use it anyway with conservative defaults
                # (gemma / pro models tend to have low limits)
                is_flash = "flash" in requested_model.lower()
                model_dict = {
                    "name": requested_model,
                    "rpm": 15 if is_flash else 2,
                    "rpd": 1500 if is_flash else 50,
                }

            # Rotate keys starting from current position
            K = len(API_KEYS)
            for i in range(K):
                k_i = (current_key_idx + i) % K
                key = API_KEYS[k_i]
                combos.append((k_i, -1, key, model_dict))

        # ── Filter: only viable (non-penalized, non-full) combos ──
        viable = []
        for (k_i, m_i, key, model) in combos:
            if _is_daily_penalized(key):
                continue
            if _is_rpm_cooldown(key, model["name"]):
                continue
            if not _rpd_available(key, model["name"], model["rpd"]):
                continue
            if not _rpm_available(key, model["name"], model["rpm"]):
                continue
            viable.append((k_i, m_i, key, model))

        return viable

def set_sticky_success(k_idx, m_idx):
    global current_key_idx, current_model_idx
    with state_lock:
        if k_idx >= 0: current_key_idx   = k_idx
        if m_idx >= 0: current_model_idx = m_idx

# ─── Request logs ─────────────────────────────────────────────────────────────
request_logs = deque(maxlen=150)

# ─── Auth ─────────────────────────────────────────────────────────────────────
def check_browser_auth(username, password):
    return password == os.environ.get("PASSWORD", "")

def request_browser_login():
    return Response('Login Required', 401,
        {'WWW-Authenticate': 'Basic realm="Admin (password field = your API PASSWORD)"'})

def requires_browser_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_browser_auth(auth.username, auth.password):
            return request_browser_login()
        return f(*args, **kwargs)
    return decorated

@app.before_request
def strict_password_and_log():
    if request.method == 'OPTIONS':
        return
    if request.path in ['/ping', '/healthz', '/logs', '/dashboard_data']:
        return
    expected_pass = os.environ.get("PASSWORD", "")
    auth_header   = request.headers.get("Authorization", "")
    is_correct    = (auth_header == f"Bearer {expected_pass}")

    msg, params_str = "", ""
    if request.is_json:
        try:
            body = request.get_json(silent=True) or {}
            params_str = json.dumps(body, indent=2)
            msgs = body.get("messages", [])
            if msgs:
                msg = msgs[-1].get("content", "")
        except Exception as e:
            params_str = str(e)

    if isinstance(msg, str) and len(msg) > 1:
        formatted_msg = f"{msg[0]}...{msg[-1]}"
    else:
        formatted_msg = str(msg) if msg else ""

    log_entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": request.headers.get("Cf-Connecting-Ip",
              request.headers.get("X-Forwarded-For", request.remote_addr)),
        "path": request.path,
        "password_used": "*** HIDDEN (CORRECT) ***" if is_correct else auth_header,
        "is_correct": is_correct,
        "message": formatted_msg,
        "params": params_str,
        "status": "Pending..."
    }
    g.log_entry = log_entry
    request_logs.appendleft(log_entry)

    if not is_correct:
        log_entry["status"] = "401 Blocked"
        return jsonify({"error": "Unauthorized Access. Invalid Password."}), 401

@app.after_request
def update_log_status(response):
    if hasattr(g, 'log_entry') and g.log_entry["status"] == "Pending...":
        g.log_entry["status"] = f"{response.status_code} {'Success' if response.status_code == 200 else 'Failed'}"
    return response

# ─── Dashboard ────────────────────────────────────────────────────────────────
@app.route('/dashboard_data', methods=['GET'])
@requires_browser_auth
def api_dashboard_data():
    now = time.time()
    now_mono = time.monotonic()
    with state_lock:
        _maybe_reset_rpd()
        penalized = {
            k[:5]+"..."+k[-5:]: round((ts - now)/60, 1)
            for k, ts in key_daily_penalty.items() if ts > now
        }
        cooldowns = {
            f"{k[:5]}...{k[-5:]}|{m}": round(ts - now_mono, 1)
            for (k, m), ts in rpm_cooldown.items() if ts > now_mono
        }
    return jsonify({
        "metrics": metrics,
        "active_keys": len(API_KEYS),
        "penalized_keys": penalized,
        "rpm_cooldowns": cooldowns,
        "models": get_active_models(),
        "logs": list(request_logs)
    })

@app.route('/logs', methods=['GET'])
@requires_browser_auth
def view_logs():
    # Same dashboard HTML as original — kept intact
    html = """<!DOCTYPE html><html><head><title>WAPI Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body{font-family:'Segoe UI',sans-serif;margin:0;padding:20px;background:#121212;color:#e0e0e0}
.container{max-width:1400px;margin:0 auto}
h1{color:#fff;text-align:center;margin-bottom:20px;display:flex;justify-content:center;align-items:center;gap:15px}
.status-panel{display:flex;flex-wrap:wrap;gap:15px;margin-bottom:25px}
.card{background:#1e1e1e;border-radius:8px;padding:15px;flex:1;min-width:200px;border:1px solid #333;box-shadow:0 4px 10px rgba(0,0,0,.3)}
.card h3{margin:0 0 10px;font-size:.9em;color:#888;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #333;padding-bottom:5px}
.card .val{font-size:1.8em;font-weight:bold;color:#fff;margin-bottom:5px}
.card .sub{font-size:.8em;color:#a1a1aa}
.table-wrapper{background:#1e1e1e;border-radius:8px;box-shadow:0 4px 15px rgba(0,0,0,.5);overflow-x:auto;border:1px solid #333;margin-bottom:30px}
table{width:100%;border-collapse:collapse;min-width:900px}
th,td{padding:12px 15px;text-align:left;border-bottom:1px solid #333}
th{background:#2c2c2c;color:#fff;font-weight:600;font-size:.95em;text-transform:uppercase;letter-spacing:.5px}
tr:hover{background:#252525}
.msg{max-width:400px;white-space:pre-wrap;word-break:break-word;font-size:.9em;color:#ce9178;background:#18181b;padding:8px;border-radius:4px;font-family:monospace}
.pwd-wrong{font-family:monospace;color:#fca5a5;background:#451a1a;padding:3px 6px;border-radius:3px;font-size:.9em}
.pwd-correct{font-family:monospace;color:#4ade80;font-style:italic;font-size:.9em;font-weight:bold}
.ip{font-family:monospace;color:#93c5fd}
.badge{padding:4px 8px;border-radius:4px;font-size:.85em;font-weight:bold}
.bg-green{background:rgba(74,222,128,.2);color:#4ade80}
.bg-red{background:rgba(248,113,113,.2);color:#f87171}
.flex-col{display:flex;flex-direction:column;gap:5px}
.tag{background:#333;padding:2px 6px;border-radius:4px;font-size:.8em;color:#ccc;border:1px solid #444}
.btn{background:#3b82f6;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-weight:bold;font-size:.9em;transition:.2s}
.btn:hover{background:#2563eb}
.params-box{display:none}
</style></head><body><div class="container">
<h1>🚀 WAPI Live Dashboard
<button class="btn" onclick="fetchData()" id="refresh-btn">🔄 Refresh</button></h1>
<div id="error-msg" style="color:#fca5a5;background:#451a1a;padding:10px;border-radius:5px;text-align:center;display:none;margin-bottom:15px;"></div>
<div class="status-panel" id="status-panel"><div class="card" style="text-align:center;padding:30px;">Loading...</div></div>
<h2 style="display:flex;justify-content:space-between;align-items:center;font-size:1.2em;color:#ccc;border-bottom:1px solid #333;padding-bottom:10px;">
🔐 Secure Access Logs
<button class="btn" onclick="toggleAllParams()" id="toggle-btn">👁️ Show All Params</button></h2>
<div class="table-wrapper"><table><thead><tr>
<th>Time</th><th>IP Address</th><th>Attempted Password</th><th>Status</th><th>Message / Prompt</th><th>Parameters</th>
</tr></thead><tbody id="logs-body"><tr><td colspan="6" style="text-align:center;color:#666;padding:30px;">Loading logs...</td></tr></tbody></table></div>
</div>
<script>
let isSelecting=false,allParamsVisible=false;
function toggleParams(idx){const el=document.getElementById('params-'+idx);el.style.display=(el.style.display==='none'||el.style.display==='')?'block':'none';}
function toggleAllParams(){allParamsVisible=!allParamsVisible;document.getElementById('toggle-btn').innerText=allParamsVisible?'🙈 Hide All Params':'👁️ Show All Params';const boxes=document.getElementsByClassName('params-box');for(let box of boxes)box.style.display=allParamsVisible?'block':'none';}
document.addEventListener('selectionchange',()=>{const s=window.getSelection();isSelecting=s.toString().length>0;});
async function fetchData(){
  const btn=document.getElementById('refresh-btn');btn.innerText='⏳...';
  try{
    const r=await fetch('/router/dashboard_data',{credentials:'same-origin'});
    if(!r.ok)throw new Error('HTTP '+r.status);
    const data=await r.json();
    document.getElementById('error-msg').style.display='none';
    if(!isSelecting)updateUI(data);
  }catch(e){
    const el=document.getElementById('error-msg');el.style.display='block';el.innerText='⚠️ Could not load data. '+e.message;
  }
  btn.innerText='🔄 Refresh';
}
function updateUI(data){
  const m=data.metrics;
  let penHtml='';
  if(Object.keys(data.penalized_keys||{}).length>0){
    for(const[k,v]of Object.entries(data.penalized_keys))
      penHtml+=`<span class="tag" style="background:rgba(248,113,113,.2);color:#f87171;border-color:#f87171;">${k} (${v}m)</span>`;
  }else penHtml='<span style="color:#4ade80;">All Clear ✅</span>';
  let cdHtml='';
  if(Object.keys(data.rpm_cooldowns||{}).length>0){
    for(const[k,v]of Object.entries(data.rpm_cooldowns))
      cdHtml+=`<span class="tag" style="background:rgba(251,191,36,.2);color:#fbbf24;border-color:#fbbf24;">${k} (${v}s)</span>`;
  }else cdHtml='<span style="color:#4ade80;">None</span>';
  document.getElementById('status-panel').innerHTML=`
    <div class="card"><h3>Active Keys</h3><div class="val" style="color:#60a5fa;">${data.active_keys}</div>
      <div class="sub flex-col">Daily penalized:<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:5px;">${penHtml}</div></div>
      <div class="sub flex-col" style="margin-top:8px;">RPM cooldowns:<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:5px;">${cdHtml}</div></div>
    </div>
    <div class="card"><h3>API Traffic</h3><div class="val">${m.total_incoming_requests}</div><div class="sub">Total Requests</div></div>
    <div class="card"><h3>Google API</h3><div class="val" style="color:#4ade80;">${m.successful_api_calls}</div>
      <div class="sub">RPM Hits: <span style="color:#fbbf24">${m.rpm_cooldowns_applied||0}</span> | Daily Hits: <span style="color:#f87171">${m.daily_limits_hit||0}</span></div></div>
    <div class="card"><h3>Results</h3><div class="val" style="color:#fbbf24;">${m.rate_limit_hits}</div>
      <div class="sub">Failed: <span style="color:#f87171">${m.failed_requests}</span></div></div>`;
  const tbody=document.getElementById('logs-body');
  if(!data.logs||data.logs.length===0){tbody.innerHTML='<tr><td colspan="6" style="text-align:center;color:#666;padding:30px;">No logs yet.</td></tr>';return;}
  let html='';
  data.logs.forEach((log,idx)=>{
    const pwdClass=log.is_correct?'pwd-correct':'pwd-wrong';
    const pwdText=log.is_correct?'🛡️ '+log.password_used:(log.password_used||'NONE');
    const badgeClass=log.is_correct?'bg-green':'bg-red';
    html+=`<tr>
      <td style="white-space:nowrap;color:#888;font-size:.9em;">${log.time||''}</td>
      <td class="ip">${log.ip||''}</td>
      <td><span class="${pwdClass}">${esc(pwdText)}</span></td>
      <td><span class="badge ${badgeClass}">${esc(log.status||'')}</span></td>
      <td><div class="msg">${esc(log.message||'No message')}</div></td>
      <td><button class="btn" style="padding:2px 6px;font-size:.7em;margin-bottom:5px;" onclick="toggleParams(${idx})">Show Params</button>
      <div id="params-${idx}" class="msg params-box" style="display:none;max-height:150px;overflow-y:auto;max-width:300px;">${esc(log.params||'No parameters')}</div></td>
    </tr>`;
  });
  tbody.innerHTML=html;
}
function esc(s){return(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;');}
fetchData();setInterval(fetchData,5000);
</script></body></html>"""
    return render_template_string(html)

# ─── Main Proxy ───────────────────────────────────────────────────────────────
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy_chat():
    if request.method == 'OPTIONS':
        return Response(status=200)

    with state_lock:
        metrics["total_incoming_requests"] += 1

    data = request.json or {}
    data.pop('session_id', None)
    data.pop('user', None)
    requested_model = data.get("model", "")

    last_resp = None
    tried = set()

    # Keep trying until we exhaust all viable combos
    max_outer_loops = 5  # safety cap
    for outer in range(max_outer_loops):
        combos = get_best_combo(requested_model)
        # Remove already-tried ones
        combos = [(ki, mi, k, m) for (ki, mi, k, m) in combos if (k, m["name"]) not in tried]

        if not combos:
            # No viable slot right now — wait for cooldowns to expire
            # Check if any cooldown is expiring soon (within 65s)
            with state_lock:
                now_mono = time.monotonic()
                soonest = min(
                    (ts for ts in rpm_cooldown.values() if ts > now_mono),
                    default=None
                )
            if soonest and (soonest - now_mono) <= 65:
                wait = soonest - time.monotonic() + 1.0
                print(f"[WAIT] All slots busy, waiting {wait:.1f}s for cooldown...")
                time.sleep(max(0, wait))
                continue
            break  # truly nothing available

        for (k_idx, m_idx, key, model) in combos:
            actual_model = model["name"]
            tried.add((key, actual_model))
            data["model"] = actual_model

            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

            with state_lock:
                _record_request(key, actual_model)

            try:
                resp = requests.post(url, json=data, headers=headers, stream=True, timeout=120)
                last_resp = resp

                if resp.status_code == 200:
                    set_sticky_success(k_idx, m_idx)
                    if hasattr(g, "log_entry"):
                        g.log_entry["status"] = f"200 Success ({actual_model})"
                    with state_lock:
                        metrics["successful_api_calls"] += 1
                        safe_key = key[:5] + "..." + key[-5:]
                        metrics["usage_by_key"].setdefault(safe_key, {})
                        metrics["usage_by_key"][safe_key][actual_model] = \
                            metrics["usage_by_key"][safe_key].get(actual_model, 0) + 1

                    excluded = ['content-encoding','content-length','transfer-encoding','connection']
                    out_headers = [(n, v) for n, v in resp.raw.headers.items() if n.lower() not in excluded]
                    return Response(resp.content, resp.status_code, out_headers)

                elif resp.status_code in [429, 403]:
                    err_text = ""
                    try: err_text = resp.text.lower()
                    except: pass

                    with state_lock:
                        metrics["rate_limit_hits"] += 1
                        if "per day" in err_text or "quota" in err_text or "daily" in err_text:
                            # Daily RPD hit → penalize key until midnight IST
                            _apply_daily_penalty(key)
                            metrics["daily_limits_hit"] = metrics.get("daily_limits_hit", 0) + 1
                        else:
                            # RPM hit → 62s cooldown for this key+model only
                            _apply_rpm_cooldown(key, actual_model, seconds=62)
                            metrics["rpm_cooldowns_applied"] = metrics.get("rpm_cooldowns_applied", 0) + 1

                    print(f"[429] key=…{key[-6:]} model={actual_model} → trying next combo")
                    continue  # try next combo in this loop

                elif resp.status_code in [500, 503]:
                    print(f"[{resp.status_code}] model={actual_model} server error, trying next...")
                    continue

                elif resp.status_code in [400, 404]:
                    print(f"[{resp.status_code}] model={actual_model} invalid, removing from pool")
                    with dynamic_models_lock:
                        global DYNAMIC_MODELS, OPENAI_MODELS_LIST
                        DYNAMIC_MODELS = [m for m in DYNAMIC_MODELS if m['name'] != actual_model]
                        OPENAI_MODELS_LIST = [m for m in OPENAI_MODELS_LIST if m['id'] != actual_model]
                    continue

                else:
                    with state_lock:
                        metrics["failed_requests"] += 1
                    excluded = ['content-encoding','content-length','transfer-encoding','connection']
                    out_headers = [(n, v) for n, v in resp.raw.headers.items() if n.lower() not in excluded]
                    return Response(resp.content, resp.status_code, out_headers)

            except Exception as e:
                print(f"[ERROR] key=…{key[-6:]} model={actual_model}: {e}")
                continue

    # All combos exhausted
    with state_lock:
        metrics["failed_requests"] += 1

    if last_resp is not None:
        excluded = ['content-encoding','content-length','transfer-encoding','connection']
        out_headers = [(n, v) for n, v in last_resp.raw.headers.items() if n.lower() not in excluded]
        return Response(last_resp.content, last_resp.status_code, out_headers)

    return jsonify({
        "error": {
            "message": "All API keys and models are exhausted. Daily limits may have been reached. Resets at midnight IST.",
            "type": "rate_limit_error"
        }
    }), 429

# ─── Audio Transcription ──────────────────────────────────────────────────────
@app.route('/v1/audio/transcriptions', methods=['POST', 'OPTIONS'])
def proxy_transcriptions():
    if request.method == 'OPTIONS':
        return Response(status=200)
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    b64_audio = base64.b64encode(file.read()).decode('utf-8')
    mime_type = file.content_type or "audio/wav"

    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": mime_type, "data": b64_audio}},
            {"text": "Transcribe the following audio accurately."}
        ]}],
        "generationConfig": {"temperature": 0.0}
    }

    for key in API_KEYS:
        if _is_daily_penalized(key):
            continue
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                with state_lock:
                    metrics["successful_api_calls"] += 1
                result = resp.json()
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return jsonify({"text": text.strip()})
                except KeyError:
                    return jsonify({"error": "Failed to parse transcription"}), 500
            elif resp.status_code in [429, 403]:
                with state_lock:
                    metrics["rate_limit_hits"] += 1
                continue
        except Exception as e:
            print(f"[TRANSCRIPTION ERROR] {e}")
            continue

    return jsonify({"error": {"message": "All keys exhausted for transcription.", "type": "rate_limit"}}), 429

# ─── Models List ──────────────────────────────────────────────────────────────
@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def proxy_models():
    if request.method == 'OPTIONS':
        return Response(status=200)
    with dynamic_models_lock:
        if OPENAI_MODELS_LIST:
            return jsonify({"object": "list", "data": OPENAI_MODELS_LIST})
    now = int(time.time())
    data = [{"id": m["name"], "object": "model", "created": now, "owned_by": "google"}
            for m in get_active_models()]
    return jsonify({"object": "list", "data": data})

# ─── Add key/model dynamically ────────────────────────────────────────────────
@app.route('/add', methods=['POST'])
def add_key_model():
    if request.headers.get("Authorization") != f"Bearer {os.environ.get('PASSWORD','')}":
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    with state_lock:
        if "key" in data and data["key"] not in API_KEYS:
            API_KEYS.append(data["key"])
        if "model" in data and "rpm" in data:
            MODELS.append({"name": data["model"], "rpm": int(data["rpm"]),
                           "rpd": int(data.get("rpd", 500))})
    return jsonify({"status": "success", "keys_count": len(API_KEYS), "models": MODELS})

@app.route('/status', methods=['GET'])
def get_status():
    now = time.time()
    now_mono = time.monotonic()
    with state_lock:
        penalized = {k[:5]+"..."+k[-5:]: round((ts-now)/60,1)
                     for k,ts in key_daily_penalty.items() if ts > now}
        cooldowns = {f"{k[:5]}...{k[-5:]}|{m}": round(ts-now_mono,1)
                     for (k,m),ts in rpm_cooldown.items() if ts > now_mono}
    return jsonify({
        "metrics": metrics,
        "active_keys": len(API_KEYS),
        "daily_penalized_keys_minutes_left": penalized,
        "rpm_cooldowns_seconds_left": cooldowns,
        "models": get_active_models()
    })

@app.route('/ping')
@app.route('/healthz')
def ping():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8085)
