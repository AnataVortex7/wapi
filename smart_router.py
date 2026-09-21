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
    rpm  = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 15
    # RPD is optional (your format is just "name:rpm") -> safe default when absent
    rpd  = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else (1500 if "flash" in name else 50)
    MODELS.append({"name": name, "rpm": rpm, "rpd": rpd})

if not MODELS:
    MODELS = [{"name": "gemini-2.0-flash-lite", "rpm": 30, "rpd": 1500}]

# Concurrency safety margin: reserve this many RPM slots as a buffer so that
# several requests admitted in the same instant (before their timestamps are
# recorded) can never push the key+model pair over its real RPM limit.
RPM_SAFETY_MARGIN = int(os.environ.get("RPM_SAFETY_MARGIN", "0"))

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

# ─── State ────────────────────────────────────────────────────────────────────
state_lock = threading.RLock()

# RPM window: (key, model_name) -> deque of monotonic timestamps
rpm_window: dict = {}

# RPD counts: (key, model_name) -> int
rpd_count: dict = {}

# RPM cooldown (429 penalty): (key, model_name) -> monotonic timestamp until blocked
rpm_cooldown: dict = {}

# RPD daily penalty: (key, model_name) -> UTC timestamp when it unlocks
# (midnight IST). Scoped per model, NOT per key -- a key that hits its daily
# quota on ONE model still works fine for every other model on that key.
# Google's free-tier RPD quota is actually per (project/key, model) anyway,
# so penalizing the whole key was both wrong and wasteful.
key_daily_penalty: dict = {}

# (key, model_name) pairs that returned a hard compatibility error (e.g.
# "This model only supports Interactions API", missing thought_signature)
# rather than a rate-limit error. These are not transient -- retrying won't
# help until the proxy code itself is changed to speak that model's newer
# API shape -- so they're excluded from routing entirely instead of being
# retried forever. Cleared on process restart.
PERMANENTLY_BROKEN_MODELS: set = set()

# ─── Conversation stickiness (fixes Gemini thought_signature 400s) ───────────
# Gemini 2.5/3 "thinking" models attach an encrypted thought_signature to
# function-call parts. That signature is only valid when it is echoed back
# to the SAME model (and same underlying key/project) that produced it. Our
# round-robin picks a fresh (key, model) on every incoming HTTP request --
# which is correct for independent, one-shot chats, but breaks any request
# that is a *continuation* of a tool-calling turn (an agent like Hermes
# sending the function result back): if that continuation lands on a
# different (key, model) than the one that emitted the function call, Gemini
# rejects it with "signature missing/mismatch" (400) and the agent's task
# stalls mid-execution.
#
# Fix: give each conversation a stable id and pin it to the (key, model)
# that handled its most recent successful turn, as long as that slot is
# still viable (not rate-limited / not broken). Independent conversations
# still spread freely across every key+model, so your overall quota usage
# is unaffected -- only steps *within* one tool-calling task stay glued to
# the same model.
CONV_STICKY_TTL = 3 * 60 * 60   # forget a caller's pin after 3h of no traffic
STICKY_MAX_WAIT = 90.0          # max seconds to hold a request waiting on RPM
conversation_sticky: dict = {}   # client_id -> (key, model_name, last_used_monotonic)

def _client_id_from_request(data: dict, remote_addr: str) -> str:
    """Identity used to pin a caller to one (key, model). Prefers an
    explicit id the client sends (session_id / user); otherwise falls back
    to the caller's IP. For a single-bot setup like Hermes → wapi, every
    call comes from the same IP anyway, so this behaves as one global pin --
    exactly what you want: whichever model is doing the current
    thinking/tool-call work stays fixed until its daily quota runs out."""
    explicit = data.get("_conv_hint")
    if explicit:
        return f"id:{explicit}"
    return f"ip:{remote_addr or 'unknown'}"

def _is_tool_continuation(messages: list) -> bool:
    """True if this request is continuing a function-calling turn (an
    assistant tool_calls message or a tool-result message is present) --
    i.e. it MUST stay on the (key, model) that started this turn, or Gemini
    will reject the thought_signature."""
    for m in messages:
        role = m.get("role")
        if role == "tool" or (role == "assistant" and m.get("tool_calls")):
            return True
    return False

def _prune_conv_sticky():
    cutoff = time.monotonic() - CONV_STICKY_TTL
    dead = [cid for cid, (_, _, ts) in conversation_sticky.items() if ts < cutoff]
    for cid in dead:
        conversation_sticky.pop(cid, None)

def _pin_hard_dead(key, model_name, model_dict) -> bool:
    """True only for failures that CANNOT be waited out: this (key, model)
    is permanently incompatible or its daily (RPD) quota is gone for today.
    RPM being momentarily full is NOT included here -- that's what we wait
    on instead of switching (see acquire_sticky_slot_wait)."""
    if (key, model_name) in PERMANENTLY_BROKEN_MODELS: return True
    if _is_daily_penalized(key, model_name): return True
    if not _rpd_available(key, model_name, model_dict["rpd"]): return True
    return False

def acquire_sticky_slot_wait(client_id: str):
    """
    Resolve the pinned slot for this client, WAITING (not switching) while
    it's only RPM-limited, since RPM resets within ~60s and switching model
    mid-task would break the thought_signature.

    Returns one of:
      ("ok", key, model_dict)   -> reserved and ready to use
      ("exhausted", None, None) -> daily quota / permanent break: pin
                                    cleared, caller should surface an error
                                    to the user; their NEXT fresh message
                                    will get a brand-new pin
      ("busy", None, None)      -> still RPM-limited after STICKY_MAX_WAIT;
                                    pin is kept (not cleared) so the next
                                    retry can resume on it
      (None, None, None)        -> no pin exists yet for this client
    """
    deadline = time.monotonic() + STICKY_MAX_WAIT
    while True:
        with state_lock:
            _prune_conv_sticky()
            entry = conversation_sticky.get(client_id)
            if not entry:
                return (None, None, None)
            key, model_name, _ = entry
            model_dict = next((m for m in MODELS if m["name"] == model_name), None)

            if model_dict is None or _pin_hard_dead(key, model_name, model_dict):
                conversation_sticky.pop(client_id, None)
                return ("exhausted", None, None)

            if _is_rpm_cooldown(key, model_name) or not _rpm_available(key, model_name, model_dict["rpm"]):
                rpm_blocked = True
            else:
                rpm_blocked = False

            if not rpm_blocked:
                _record_request(key, model_name)
                conversation_sticky[client_id] = (key, model_name, time.monotonic())
                return ("ok", key, model_dict)

        # RPM-limited only -- wait it out instead of switching models.
        if time.monotonic() >= deadline:
            return ("busy", None, None)
        time.sleep(1.0)

def remember_sticky_slot(client_id: str, key: str, model_name: str):
    with state_lock:
        conversation_sticky[client_id] = (key, model_name, time.monotonic())

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
    # RPM_SAFETY_MARGIN reserves headroom so several requests admitted in the
    # same instant (before their own timestamp is recorded) can never push
    # the pair over the real limit when many users hit the router at once.
    effective_limit = max(1, rpm_limit - RPM_SAFETY_MARGIN)
    return len(window) < effective_limit

def _rpd_available(key, model_name, rpd_limit):
    used = _get_key(key, model_name, rpd_count, lambda: 0)
    if isinstance(used, deque):  # safety
        used = 0
    return used < rpd_limit

def _is_rpm_cooldown(key, model_name):
    until = rpm_cooldown.get((key, model_name), 0)
    return time.monotonic() < until

def _is_daily_penalized(key, model_name):
    until = key_daily_penalty.get((key, model_name), 0)
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

def _apply_daily_penalty(key, model_name):
    """Block this (key, model) pair until midnight IST after its RPD limit
    is hit. Only this model is blocked on this key -- every other model on
    the same key keeps working normally."""
    now_ist = datetime.datetime.now(IST)
    next_midnight = (now_ist + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=5, microsecond=0)
    unlock_ts = next_midnight.astimezone(datetime.timezone.utc).timestamp()
    key_daily_penalty[(key, model_name)] = unlock_ts
    mins = round((unlock_ts - time.time()) / 60)
    print(f"[DAILY LIMIT] key=…{key[-6:]} model={model_name} penalized until midnight IST (~{mins} mins) — other models on this key are unaffected")

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
    # ALWAYS use the models you configured in GEMINI_MODELS for routing.
    # The dynamically-fetched list (DYNAMIC_MODELS) is informational only —
    # it used to silently replace your ENV pool in "auto" mode, which meant
    # auto-mode requests could be routed to models you never approved/rate-
    # limited in GEMINI_MODELS. That's fixed: MODELS (from env) is now the
    # single source of truth for routing.
    return MODELS

# ─── Core: Smart Combo Picker ─────────────────────────────────────────────────
current_key_idx   = 0
current_model_idx = 0

def _list_combos_in_order(requested_model: str):
    """
    Builds the ordered candidate list (key, model) for a request.
    Must be called while already holding state_lock.

    Two modes:
    1. POOL MODE ("auto"/empty/generic alias):
       - Uses ONLY the models configured in GEMINI_MODELS (env) — never the
         dynamically-fetched list — so auto mode always respects your RPM
         settings for gemini-3.1-flash-lite, gemini-3.5-flash, etc.
       - Tries all key × model combos, highest-RPM model first.
    2. SPECIFIC MODEL MODE (user sent a real model name):
       - Only that model is used, all keys are tried in rotation.
    """
    active_models = get_active_models()

    GENERIC_ALIASES = {"gemini-pro", "auto", "default", "round-robin",
                        "gemini-working-model", ""}
    is_pool_mode = requested_model.lower() in GENERIC_ALIASES

    combos = []

    if is_pool_mode:
        sorted_models = sorted(active_models, key=lambda m: -m["rpm"])
        K, M = len(API_KEYS), len(sorted_models)
        if M == 0 or K == 0:
            return []
        start = (current_key_idx % K) * M + (current_model_idx % M)
        for i in range(K * M):
            idx = (start + i) % (K * M)
            k_i = idx // M
            m_i = idx % M
            combos.append((k_i, m_i, API_KEYS[k_i], sorted_models[m_i]))
    else:
        model_dict = next(
            (m for m in active_models if m["name"] == requested_model), None
        )
        if model_dict is None:
            is_flash = "flash" in requested_model.lower()
            model_dict = {
                "name": requested_model,
                "rpm": 15 if is_flash else 2,
                "rpd": 1500 if is_flash else 50,
            }
        K = len(API_KEYS)
        if K == 0:
            return []
        for i in range(K):
            k_i = (current_key_idx + i) % K
            combos.append((k_i, -1, API_KEYS[k_i], model_dict))

    return combos


def acquire_next_slot(requested_model: str, exclude: set):
    """
    Atomically picks the next viable (key, model) combo AND reserves it
    (records the request timestamp) in one locked step.

    This is the fix for the multi-user race condition: previously,
    "find a viable combo" and "record that a request is using it" were two
    separate lock acquisitions, so two requests arriving at nearly the same
    moment could both pass the RPM check for the same key+model before
    either one recorded its usage — letting concurrent users occasionally
    slip past the RPM limit or collide on the same slot.

    Now the check-then-reserve happens under a single lock hold, so at most
    one caller can ever claim a given (key, model) slot for a given instant.
    Returns (k_i, m_i, key, model) or None if nothing is viable right now.
    """
    with state_lock:
        _maybe_reset_rpd()

        if not API_KEYS:
            return None

        combos = _list_combos_in_order(requested_model)

        for (k_i, m_i, key, model) in combos:
            if (key, model["name"]) in exclude:
                continue
            if (key, model["name"]) in PERMANENTLY_BROKEN_MODELS:
                continue
            if _is_daily_penalized(key, model["name"]):
                continue
            if _is_rpm_cooldown(key, model["name"]):
                continue
            if not _rpd_available(key, model["name"], model["rpd"]):
                continue
            if not _rpm_available(key, model["name"], model["rpm"]):
                continue

            # Reserve immediately, still inside the lock, before returning —
            # this is what closes the race window.
            _record_request(key, model["name"])
            return (k_i, m_i, key, model)

        return None


def get_best_combo(requested_model: str):
    """
    Kept for compatibility with /status and any external callers — returns
    the full viable list WITHOUT reserving anything. The actual proxy path
    uses acquire_next_slot() instead, which is race-free.
    """
    with state_lock:
        _maybe_reset_rpd()
        if not API_KEYS:
            return []
        combos = _list_combos_in_order(requested_model)
        viable = []
        for (k_i, m_i, key, model) in combos:
            if (key, model["name"]) in PERMANENTLY_BROKEN_MODELS:
                continue
            if _is_daily_penalized(key, model["name"]):
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
        # key_daily_penalty is now keyed by (key, model) — display it the
        # same way rpm_cooldowns is displayed, so the dashboard shows which
        # specific model on which key is daily-exhausted, not the whole key.
        penalized = {
            f"{k[:5]}...{k[-5:]}|{m}": round((ts - now)/60, 1)
            for (k, m), ts in key_daily_penalty.items() if ts > now
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
    # Capture a conversation id BEFORE popping these -- session_id/user are
    # the clearest hints a client can give us; we fall back to the caller's
    # IP otherwise (see _client_id_from_request).
    conv_hint = data.get('session_id') or data.get('user')
    if conv_hint:
        data['_conv_hint'] = conv_hint
    data.pop('session_id', None)
    data.pop('user', None)
    requested_model = data.get("model", "")
    messages = data.get("messages", []) or []
    remote_addr = request.headers.get("X-Real-IP") or \
                  (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()) or \
                  request.remote_addr
    client_id = _client_id_from_request(data, remote_addr)
    data.pop('_conv_hint', None)
    is_continuation = _is_tool_continuation(messages)

    last_resp = None
    tried = set()  # (key, model_name) already attempted this request

    if is_continuation:
        # MUST stay on the (key, model) that started this tool-calling
        # turn, or Gemini's thought_signature check fails. Wait through
        # short RPM limits instead of switching; only give up (and clear
        # the pin) once the daily quota is genuinely gone.
        outcome, key, model = acquire_sticky_slot_wait(client_id)

        if outcome == "exhausted":
            return jsonify({
                "error": {
                    "message": "Daily quota exhausted for the model handling this "
                                "task. This step could not be completed on the same "
                                "model, so the thought/tool signature can't carry "
                                "over. Please send a new message to start fresh "
                                "(it will pick up a working model automatically).",
                    "type": "quota_exceeded"
                }
            }), 429

        if outcome == "busy":
            return jsonify({
                "error": {
                    "message": "The model handling this task is briefly rate-limited. "
                                "Please retry in a few seconds -- it will resume on "
                                "the same model.",
                    "type": "rate_limit_error"
                }
            }), 429

        if outcome == "ok":
            print(f"[STICKY] {client_id} → staying on key=…{key[-6:]} model={model['name']}")
            tried.add((key, model["name"]))
            actual_model = model["name"]
            data["model"] = actual_model
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            try:
                resp = requests.post(url, json=data, headers=headers, stream=True, timeout=120)
                if resp.status_code == 200:
                    remember_sticky_slot(client_id, key, actual_model)
                    with state_lock:
                        metrics["successful_api_calls"] += 1
                    excluded = ['content-encoding','content-length','transfer-encoding','connection']
                    out_headers = [(n, v) for n, v in resp.raw.headers.items() if n.lower() not in excluded]
                    return Response(resp.content, resp.status_code, out_headers)
                # A non-200 on the pinned slot despite our own RPM/RPD checks
                # passing (e.g. Google's own limit differs slightly from
                # ours, or a genuine signature/payload issue) -- surface it
                # directly rather than silently trying a different model,
                # since a different model can't validate this signature
                # anyway.
                excluded = ['content-encoding','content-length','transfer-encoding','connection']
                out_headers = [(n, v) for n, v in resp.raw.headers.items() if n.lower() not in excluded]
                return Response(resp.content, resp.status_code, out_headers)
            except Exception as e:
                return jsonify({"error": {"message": f"Upstream error on pinned model: {e}", "type": "upstream_error"}}), 502
        # outcome is None -> no pin exists yet for this client (e.g. process
        # just restarted mid-task). Nothing safe to resume -- fall through
        # to normal routing below; this one step may still 400, but the
        # NEXT message from the user will get a clean new pin.

    # How many total distinct (key, model) attempts to allow per incoming
    # request before giving up. Sized to the whole pool so that under load
    # we genuinely exhaust every key×model combo instead of stopping early.
    max_attempts = max(20, len(API_KEYS) * max(1, len(MODELS)) + 5)

    for attempt in range(max_attempts):
        # Atomic: pick a viable (key, model) AND reserve it in one lock
        # hold. This closes the race window multiple simultaneous users
        # could hit under the old "list combos, then record separately"
        # approach. Fresh (non-continuation) requests are free to spread
        # across every key/model -- this is what keeps your whole pool's
        # quota in use.
        slot = acquire_next_slot(requested_model, exclude=tried)

        if slot is None:
            # Nothing available right now for any untried combo.
            # If something is about to come off cooldown soon, wait for it
            # instead of failing the user's request.
            with state_lock:
                now_mono = time.monotonic()
                soonest = min(
                    (ts for ts in rpm_cooldown.values() if ts > now_mono),
                    default=None
                )
            if soonest and (soonest - now_mono) <= 65:
                wait = soonest - time.monotonic() + 1.0
                print(f"[WAIT] All slots busy, waiting {wait:.1f}s for a cooldown to clear...")
                time.sleep(max(0, wait))
                continue
            break  # truly nothing available (e.g. all keys hit daily limit)

        k_idx, m_idx, key, model = slot
        actual_model = model["name"]
        tried.add((key, actual_model))
        data["model"] = actual_model

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

        try:
            resp = requests.post(url, json=data, headers=headers, stream=True, timeout=120)
            last_resp = resp

            if resp.status_code == 200:
                if k_idx >= 0:
                    set_sticky_success(k_idx, m_idx)
                # Pin this conversation to this exact (key, model) so that
                # if the assistant's reply contains a tool call, the NEXT
                # request (the tool result) lands back on the same slot and
                # its thought_signature still validates.
                remember_sticky_slot(client_id, key, actual_model)
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
                        # Daily RPD hit → penalize ONLY this (key, model) pair
                        # until midnight IST. Every other model on this same
                        # key keeps working — the key itself is not blocked,
                        # only the specific model whose daily quota ran out.
                        _apply_daily_penalty(key, actual_model)
                        metrics["daily_limits_hit"] = metrics.get("daily_limits_hit", 0) + 1
                    else:
                        # RPM/rate hit → short cooldown for this key+model
                        # pair only. Every other key and every other model
                        # stays fully usable — this is what makes the
                        # "instantly switch to a working model/key" behavior
                        # work for other concurrent users too.
                        _apply_rpm_cooldown(key, actual_model, seconds=62)
                        metrics["rpm_cooldowns_applied"] = metrics.get("rpm_cooldowns_applied", 0) + 1

                print(f"[429] key=…{key[-6:]} model={actual_model} → instantly switching to next key/model")
                continue  # instantly retry with the next best slot

            elif resp.status_code in [500, 503]:
                print(f"[{resp.status_code}] model={actual_model} server error, trying next...")
                continue

            elif resp.status_code in [400, 404]:
                err_text = ""
                try: err_text = resp.text.lower()
                except: pass

                # Two different kinds of 400 need different handling:
                #
                # 1. Payload/compatibility errors ("Interactions API" only,
                #    missing thought_signature, etc.) mean THIS MODEL cannot
                #    be used through this chat/completions-style proxy at
                #    all -- retrying it will just fail again, forever, and
                #    without this fix it silently got re-picked every time
                #    because the old code only ever removed models from the
                #    unused DYNAMIC_MODELS list, never from MODELS (the env
                #    pool that auto mode actually uses).
                #
                # 2. Genuine bad-request errors caused by the request body
                #    itself (bad JSON, unsupported field, etc.) would repeat
                #    on every model/key too, but are not model-specific --
                #    those should not silently disable a model, so we only
                #    hard-disable the model on the known compatibility
                #    signatures below and otherwise just move on and let the
                #    request fail after trying a couple of other slots.
                model_is_incompatible = any(sig in err_text for sig in [
                    "interactions api",
                    "thought_signature",
                    "not supported",
                    "unsupported",
                ])

                if model_is_incompatible:
                    with state_lock:
                        # Disable this model for THIS key permanently for the
                        # rest of the process lifetime (until restart/redeploy)
                        # -- it will never succeed on this key, so don't waste
                        # future requests retrying it. midnight-IST-style
                        # cooldown doesn't apply here since the problem isn't
                        # rate limiting, it's a hard incompatibility.
                        PERMANENTLY_BROKEN_MODELS.add((key, actual_model))
                        metrics["failed_requests"] += 1
                    print(f"[400 INCOMPATIBLE] key=…{key[-6:]} model={actual_model} "
                          f"does not work via this proxy shape → disabled for this key, switching instantly")
                else:
                    print(f"[{resp.status_code}] model={actual_model} bad request, trying next combo...")
                continue  # instantly try the next key/model regardless
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

    # Every key×model combo was tried (or the pool is genuinely exhausted).
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

    TRANSCRIBE_MODEL = "gemini-1.5-flash"
    for key in API_KEYS:
        if _is_daily_penalized(key, TRANSCRIBE_MODEL):
            continue
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{TRANSCRIBE_MODEL}:generateContent?key={key}"
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
        penalized = {f"{k[:5]}...{k[-5:]}|{m}": round((ts-now)/60,1)
                     for (k,m),ts in key_daily_penalty.items() if ts > now}
        cooldowns = {f"{k[:5]}...{k[-5:]}|{m}": round(ts-now_mono,1)
                     for (k,m),ts in rpm_cooldown.items() if ts > now_mono}
        broken = [f"{k[:5]}...{k[-5:]}|{m}" for (k,m) in PERMANENTLY_BROKEN_MODELS]
    return jsonify({
        "metrics": metrics,
        "active_keys": len(API_KEYS),
        "daily_penalized_keys_minutes_left": penalized,
        "rpm_cooldowns_seconds_left": cooldowns,
        "permanently_broken_key_model_pairs": broken,
        "models": get_active_models()
    })

@app.route('/ping')
@app.route('/healthz')
def ping():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8085)
