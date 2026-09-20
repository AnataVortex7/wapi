import os, json, time, threading, datetime
from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string, g
from functools import wraps
from collections import deque
import requests
import base64
import urllib.parse

import hashlib




app = Flask(__name__)

# Config loaded from Environment
RAW_KEYS = os.environ.get("GEMINI_API_KEYS", "").split(",")
RAW_MODELS = os.environ.get("GEMINI_MODELS", "gemini-1.5-flash:15,gemini-1.5-pro:2").split(",")

API_KEYS = list(dict.fromkeys([k.strip() for k in RAW_KEYS if k.strip()]))
MODELS = []
for m in RAW_MODELS:
    if not m.strip(): continue
    parts = m.split(":")
    name = parts[0].strip()
    rpm = int(parts[1].strip()) if len(parts) > 1 else 15
    rpd = int(parts[2].strip()) if len(parts) > 2 else (1500 if "flash" in name else 50)
    MODELS.append({"name": name, "rpm": rpm, "rpd": rpd})

if not MODELS:
    MODELS = [{"name": "gemini-1.5-flash", "rpm": 15, "rpd": 1500}]

# State tracking
key_penalties = {}  # key -> timestamp when it was banned
request_history = {} # (key, model) -> list of timestamps

# Global Metrics
metrics = {
    "total_incoming_requests": 0,
    "successful_api_calls": 0,
    "rate_limit_hits": 0,
    "fallback_calls": 0,
    "failed_requests": 0,
    "usage_by_key": {}
}

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
                        new_rr = []
                        new_openai = []
                        now = int(time.time())
                        for m in data:
                            name = m["name"].replace("models/", "")
                            new_openai.append({"id": name, "object": "model", "created": now, "owned_by": "google"})
                            methods = m.get("supportedGenerationMethods", [])
                            if "generateContent" in methods and "vision" not in name.lower() and "embedding" not in name.lower() and "preview" not in name.lower() and not name.startswith("gemini-2.5"):
                                rpm = 1500 if "flash" in name.lower() else 15
                                rpd = 1500 if "flash" in name.lower() else 50
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
                    print(f"Model refresh error: {e}")
                    
        if not DYNAMIC_MODELS:
            time.sleep(60)
        else:
            time.sleep(86400)

threading.Thread(target=refresh_models_loop, daemon=True).start()

def get_active_models():
    with dynamic_models_lock:
        return DYNAMIC_MODELS if DYNAMIC_MODELS else MODELS

state_lock = threading.Lock()
current_key_idx = 0
current_model_idx = 0

def clean_history(history):
    now = time.time()
    return [ts for ts in history if now - ts < 60.0]

def get_next_available_combo():
    global current_key_idx, current_model_idx
    now = time.time()
    
    with state_lock:
        active_models = get_active_models()
        if not API_KEYS or not active_models:
            return None, None
        
        total_combos = len(API_KEYS) * len(active_models)
        for _ in range(total_combos):
            # Dynamic bounds safety check
            current_key_idx = current_key_idx % len(API_KEYS)
            current_model_idx = current_model_idx % len(active_models)
            
            key = API_KEYS[current_key_idx]
            model = active_models[current_model_idx]
            
            # Advance pointers
            current_model_idx += 1
            if current_model_idx >= len(active_models):
                current_model_idx = 0
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
            
            # Check variable penalty
            if key in key_penalties:
                if now < key_penalties[key]:
                    continue
                else:
                    del key_penalties[key]
                    
            # Check RPM limit
            combo_id = f"{key}_{model['name']}"
            history = request_history.get(combo_id, [])
            history = clean_history(history)
            request_history[combo_id] = history
            
            if len(history) < model['rpm']:
                history.append(now)
                return key, model['name']
                
        return None, None

request_logs = deque(maxlen=150)

def check_browser_auth(username, password):
    expected_pass = os.environ.get("PASSWORD", "")
    return password == expected_pass

def request_browser_login():
    return Response(
        'Login Required to view logs.', 401,
        {'WWW-Authenticate': 'Basic realm="Admin Login (Use any username, put your API PASSWORD in password field)"'})

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
        
    # Exclude system routes from API password check (Logs has its own browser auth)
    if request.path in ['/ping', '/healthz', '/logs', '/dashboard_data']:
        return
        
    expected_pass = os.environ.get("PASSWORD", "")
    auth_header = request.headers.get("Authorization", "")
    
    is_correct = (auth_header == f"Bearer {expected_pass}")
    
    # Extract message and parameters
    msg = ""
    params_str = ""
    if request.is_json:
        try:
            body = request.get_json(silent=True) or {}
            params_str = json.dumps(body, indent=2)
            if "messages" in body and isinstance(body["messages"], list) and len(body["messages"]) > 0:
                msg = body["messages"][-1].get("content", "")
        except Exception as e:
            params_str = str(e)
            
    if not params_str and request.args:
        params_str = json.dumps(dict(request.args), indent=2)

    if isinstance(msg, str) and len(msg) > 1:
        formatted_msg = f"{msg[0]}...{msg[-1]}"
    elif isinstance(msg, str) and len(msg) == 1:
        formatted_msg = msg
    else:
        formatted_msg = str(msg) if msg else ""

    log_entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": request.headers.get("Cf-Connecting-Ip", request.headers.get("X-Forwarded-For", request.remote_addr)),
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
        log_entry["status"] = "401 Blocked (Hacker/Wrong Pass)"
        return jsonify({"error": "Unauthorized Access. Invalid Password."}), 401

@app.after_request
def update_log_status(response):
    if hasattr(g, 'log_entry'):
        if g.log_entry["status"] == "Pending...":
            if response.status_code == 200:
                g.log_entry["status"] = "200 Success"
            else:
                g.log_entry["status"] = f"{response.status_code} Failed"
    return response

@app.route('/dashboard_data', methods=['GET'])
@requires_browser_auth
def api_dashboard_data():
    now = time.time()
    penalized = {k[:5]+"..."+k[-5:]: round((ts - now)/60, 1) for k, ts in key_penalties.items()}
    
    return jsonify({
        "metrics": metrics,
        "active_keys": len(API_KEYS),
        "penalized_keys": penalized,
        "models": MODELS,
        "logs": list(request_logs)
    })

@app.route('/logs', methods=['GET'])
@requires_browser_auth
def view_logs():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WAPI Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #121212; color: #e0e0e0; }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: #ffffff; text-align: center; margin-bottom: 20px; text-shadow: 0 0 10px rgba(255,255,255,0.2); display: flex; justify-content: center; align-items: center; gap: 15px;}
            
            /* Status Panel */
            .status-panel { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 25px; }
            .card { background: #1e1e1e; border-radius: 8px; padding: 15px; flex: 1; min-width: 200px; border: 1px solid #333; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
            .card h3 { margin: 0 0 10px 0; font-size: 0.9em; color: #888; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 5px; }
            .card .val { font-size: 1.8em; font-weight: bold; color: #fff; margin-bottom: 5px; }
            .card .sub { font-size: 0.8em; color: #a1a1aa; }
            
            .table-wrapper { background: #1e1e1e; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); overflow-x: auto; border: 1px solid #333; margin-bottom: 30px;}
            table { width: 100%; border-collapse: collapse; min-width: 900px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #333; }
            th { background-color: #2c2c2c; color: #ffffff; font-weight: 600; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.5px; }
            tr:hover { background-color: #252525; }
            
            .msg { max-width: 400px; white-space: pre-wrap; word-break: break-word; font-size: 0.9em; color: #ce9178; background: #18181b; padding: 8px; border-radius: 4px; font-family: monospace; }
            .pwd-wrong { font-family: monospace; color: #fca5a5; background: #451a1a; padding: 3px 6px; border-radius: 3px; font-size: 0.9em; }
            .pwd-correct { font-family: monospace; color: #4ade80; font-style: italic; font-size: 0.9em; font-weight: bold; }
            .ip { font-family: monospace; color: #93c5fd; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
            .bg-green { background: rgba(74, 222, 128, 0.2); color: #4ade80; }
            .bg-red { background: rgba(248, 113, 113, 0.2); color: #f87171; }
            
            .flex-col { display: flex; flex-direction: column; gap: 5px; }
            .tag { background: #333; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; color: #ccc; border: 1px solid #444;}
            
            .btn { background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 0.9em; transition: 0.2s; }
            .btn:hover { background: #2563eb; }
            
            .usage-table th { background-color: #1f2937; }
            .params-box { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>
                🚀 WAPI Live Dashboard
                <button class="btn" onclick="fetchData()" id="refresh-btn">🔄 Refresh</button>
            </h1>
            
            <!-- Error message container -->
            <div id="error-msg" style="color: #fca5a5; background: #451a1a; padding: 10px; border-radius: 5px; text-align: center; display: none; margin-bottom: 15px;"></div>

            <div class="status-panel" id="status-panel">
                <div class="card" style="text-align:center; padding: 30px;">Loading dashboard metrics...</div>
            </div>
            
            <div class="status-panel" id="usage-panel">
                <!-- Usage by key will go here -->
            </div>
            
            <h2 style="display:flex; justify-content:space-between; align-items:center; font-size: 1.2em; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 10px;">
                🔐 Secure Access Logs
                <button class="btn" onclick="toggleAllParams()" id="toggle-btn">👁️ Show All Params</button>
            </h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>IP Address</th>
                            <th>Attempted Password</th>
                            <th>Status</th>
                            <th>Message / Prompt</th>
                            <th>Parameters</th>
                        </tr>
                    </thead>
                    <tbody id="logs-body">
                        <tr><td colspan="6" style="text-align: center; color: #666; padding: 30px;">Loading logs...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            let isSelecting = false;
            let allParamsVisible = false;
            
            function toggleParams(idx) {
                const el = document.getElementById('params-' + idx);
                el.style.display = (el.style.display === 'none' || el.style.display === '') ? 'block' : 'none';
            }
            
            function toggleAllParams() {
                allParamsVisible = !allParamsVisible;
                document.getElementById('toggle-btn').innerText = allParamsVisible ? '🙈 Hide All Params' : '👁️ Show All Params';
                const boxes = document.getElementsByClassName('params-box');
                for (let box of boxes) {
                    box.style.display = allParamsVisible ? 'block' : 'none';
                }
            }
            
            document.addEventListener('selectionchange', () => {
                const selection = window.getSelection();
                isSelecting = selection.toString().length > 0;
            });

            async function fetchData() {
                const refreshBtn = document.getElementById('refresh-btn');
                refreshBtn.innerText = "⏳...";
                
                try {
                    const response = await fetch('/router/dashboard_data', { credentials: 'same-origin' });
                    
                    if (!response.ok) {
                        throw new Error("HTTP " + response.status);
                    }
                    
                    const data = await response.json();
                    
                    document.getElementById('error-msg').style.display = 'none';
                    
                    if (!isSelecting) {
                        updateUI(data);
                    }
                } catch (error) {
                    console.error("Error fetching data:", error);
                    const errMsg = document.getElementById('error-msg');
                    errMsg.style.display = 'block';
                    errMsg.innerText = "⚠️ Could not load data. " + error.message;
                }
                
                refreshBtn.innerText = "🔄 Refresh";
            }

            function updateUI(data) {
                const metrics = data.metrics;
                
                // --- 1. Top Cards ---
                let penalizedHtml = '';
                if (Object.keys(data.penalized_keys).length > 0) {
                    for (const [k, v] of Object.entries(data.penalized_keys)) {
                        penalizedHtml += `<span class="tag" style="background: rgba(248,113,113,0.2); color: #f87171; border-color: #f87171;">${k} (${v}m)</span>`;
                    }
                } else {
                    penalizedHtml = `<span style="color: #4ade80;">All Clear</span>`;
                }

                let modelsHtml = '';
                data.models.forEach(m => {
                    let totalUsed = 0;
                    let keysHtml = '';
                    let exhaustedKeys = 0;
                    let penalizedForThisModel = 0;
                    
                    const usageKeys = Object.keys(metrics.usage_by_key || {});
                    usageKeys.forEach(keyPrefix => {
                        const count = metrics.usage_by_key[keyPrefix][m.name] || 0;
                        if(count > 0) {
                            totalUsed += count;
                            let remaining = m.rpd - count;
                            if(remaining < 0) remaining = 0;
                            
                            if(remaining === 0) exhaustedKeys++;
                            if(data.penalized_keys[keyPrefix]) penalizedForThisModel++;
                            
                            let color = remaining > 100 ? '#4ade80' : (remaining > 10 ? '#fbbf24' : '#f87171');
                            let penStatus = data.penalized_keys[keyPrefix] ? ` <span style="color:#f87171;font-weight:bold;">(Blocked)</span>` : '';
                            
                            keysHtml += `
                                <div style="display:flex; justify-content:space-between; font-size:0.85em; padding:3px 0; border-bottom:1px solid #333;">
                                    <span style="font-family:monospace; color:#93c5fd;">${keyPrefix}${penStatus}</span>
                                    <span>Used: <strong style="color:#fff;">${count}</strong> | Rem: <strong style="color:${color};">${remaining}</strong></span>
                                </div>
                            `;
                        }
                    });
                    
                    if(keysHtml === '') {
                        keysHtml = '<div style="font-size:0.85em; color:#666; padding:5px 0;">No usage yet.</div>';
                    }
                    
                    let badgeHtml = '';
                    if (exhaustedKeys > 0) {
                        badgeHtml += `<span style="background:rgba(248,113,113,0.2); color:#f87171; padding:2px 6px; border-radius:4px; font-size:0.8em; margin-left:5px; font-weight:bold;">⚠️ Limit Reached</span>`;
                    } else if (penalizedForThisModel > 0) {
                        badgeHtml += `<span style="background:rgba(251,191,36,0.2); color:#fbbf24; padding:2px 6px; border-radius:4px; font-size:0.8em; margin-left:5px; font-weight:bold;">⚠️ Key Blocked</span>`;
                    }

                    modelsHtml += `
                        <div class="model-box" style="background:#252526; border:1px solid #444; border-radius:6px; margin-bottom:10px; overflow:hidden;">
                            <div class="model-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'" style="cursor:pointer; padding:8px 12px; display:flex; justify-content:space-between; align-items:center; background:#2d2d30;">
                                <div>
                                    <strong style="color:#e0e0e0; font-size:0.95em;">${m.name}</strong>
                                    ${badgeHtml}
                                </div>
                                <div style="display:flex; gap:10px; align-items:center;">
                                    <span style="color:#4ade80; font-size:0.85em; font-weight:bold;">Used: ${totalUsed}</span>
                                    <span style="font-size:0.8em; color:#aaa; background:#1e1e1e; padding:2px 6px; border-radius:4px; border:1px solid #444;">${m.rpm} RPM | ${m.rpd} RPD</span>
                                </div>
                            </div>
                            <div class="model-details" style="display:none; padding:10px; background:#1e1e1e;">
                                <div style="font-size:0.85em; color:#888; margin-bottom:5px;">Session Usage by Key:</div>
                                ${keysHtml}
                            </div>
                        </div>
                    `;
                });

                document.getElementById('status-panel').innerHTML = `
                    <div class="card">
                        <h3>Active Keys</h3>
                        <div class="val" style="color: #60a5fa;">${data.active_keys}</div>
                        <div class="sub flex-col">Penalized: <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:5px;">${penalizedHtml}</div></div>
                    </div>
                    <div class="card">
                        <h3>API Traffic</h3>
                        <div class="val">${metrics.total_incoming_requests}</div>
                        <div class="sub">Total Requests</div>
                    </div>
                    <div class="card">
                        <h3>Google API</h3>
                        <div class="val" style="color: #4ade80;">${metrics.successful_api_calls}</div>
                        <div class="sub">Limits Hit: <span style="color:#f87171">${metrics.rate_limit_hits}</span></div>
                    </div>
                    <div class="card">
                        <h3>Fallback</h3>
                        <div class="val" style="color: #fbbf24;">${metrics.fallback_calls}</div>
                        <div class="sub">Failed Requests: <span style="color:#f87171">${metrics.failed_requests}</span></div>
                    </div>
                    <div class="card" style="flex: 2;">
                        <h3>🤖 Active Models (Click to view Key usage & RPD)</h3>
                        <div class="flex-col">
                            ${modelsHtml}
                        </div>
                    </div>
                `;

                const tbody = document.getElementById('logs-body');
                if (data.logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666; padding: 30px;">No requests logged yet.</td></tr>';
                    return;
                }

                let html = '';
                data.logs.forEach((log, idx) => {
                    const pwdClass = log.is_correct ? 'pwd-correct' : 'pwd-wrong';
                    const pwdText = log.is_correct ? '🛡️ ' + log.password_used : (log.password_used || 'NONE');
                    const badgeClass = log.is_correct ? 'bg-green' : 'bg-red';
                    
                    html += `
                        <tr>
                            <td style="white-space: nowrap; color: #888; font-size: 0.9em;">${log.time || ''}</td>
                            <td class="ip">${log.ip || ''}</td>
                            <td><span class="${pwdClass}">${escapeHtml(pwdText)}</span></td>
                            <td><span class="badge ${badgeClass}">${escapeHtml(log.status || '')}</span></td>
                            <td><div class="msg">${escapeHtml(log.message || 'No message')}</div></td>
                            <td>
                                <button class="btn" style="padding: 2px 6px; font-size: 0.7em; margin-bottom: 5px;" onclick="toggleParams(${idx})">Show Params</button>
                                <div id="params-${idx}" class="msg params-box" style="display: none; max-height: 150px; overflow-y: auto; max-width: 300px;">${escapeHtml(log.params || 'No parameters')}</div>
                            </td>
                        </tr>
                    `;
                });
                tbody.innerHTML = html;
            }

            function toggleParams(idx) {
                const el = document.getElementById('params-' + idx);
                if (el.style.display === 'none') {
                    el.style.display = 'block';
                } else {
                    el.style.display = 'none';
                }
            }

            function escapeHtml(unsafe) {
                return (unsafe || '').toString()
                     .replace(/&/g, "&amp;")
                     .replace(/</g, "&lt;")
                     .replace(/>/g, "&gt;")
                     .replace(/"/g, "&quot;")
                     .replace(/'/g, "&#039;");
            }

            fetchData();
            setInterval(fetchData, 5000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy_chat():
    if request.method == 'OPTIONS':
        return Response(status=200)
        
    with state_lock:
        metrics["total_incoming_requests"] += 1
        
    data = request.json or {}
    requested_model = data.get("model", "")
    
    # Try Real APIs across all keys and models
    attempts = 0
    last_resp = None
    
    # We will track which keys have been tried for specific models to avoid spamming
    tried_keys = set()
    is_round_robin = requested_model.lower() in ["gemini-pro", "auto", "default", "round-robin", "gemini-working-model", ""]
    
    max_retries = len(API_KEYS) * len(get_active_models()) if (API_KEYS and get_active_models() and is_round_robin) else (len(API_KEYS) if API_KEYS else 0)
    
    while attempts < max_retries:
        key, rr_model_name = get_next_available_combo()
        if not key:
            break
            
        attempts += 1
        
        # If user asked for gemini-pro/auto, do full round-robin. Else use their specific model.
        if is_round_robin:
            actual_model = rr_model_name
        else:
            actual_model = requested_model
            # Avoid trying the same key multiple times for the exact same specific model
            if key in tried_keys:
                continue
            tried_keys.add(key)
            
        data["model"] = actual_model
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        
        try:
            resp = requests.post(url, json=data, headers=headers, stream=True)
            last_resp = resp
            
            if resp.status_code == 200:
                with state_lock:
                    metrics["successful_api_calls"] += 1
                    safe_key = key[:5] + "..." + key[-5:]
                    if safe_key not in metrics["usage_by_key"]:
                        metrics["usage_by_key"][safe_key] = {}
                    if actual_model not in metrics["usage_by_key"][safe_key]:
                        metrics["usage_by_key"][safe_key][actual_model] = 0
                    metrics["usage_by_key"][safe_key][actual_model] += 1
                    
                excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                out_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                               if name.lower() not in excluded_headers]
                return Response(resp.content, resp.status_code, out_headers)
                
            elif resp.status_code in [429, 403]:
                error_text = resp.text.lower()
                
                # १. जर 403 (Safety/Permission Issue) असेल, तर इतर keys block करू नका. 
                if resp.status_code == 403:
                    print(f"Key {key[:5]}...{key[-5:]} hit 403 Forbidden. Stopping retries.")
                    with state_lock:
                        metrics["failed_requests"] += 1
                    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                    out_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                                   if name.lower() not in excluded_headers]
                    return Response(resp.content, resp.status_code, out_headers)

                if "per day" in error_text:
                    # Only block until midnight IST if strictly daily quota
                    import datetime
                    now_utc = datetime.datetime.utcnow()
                    ist_offset = datetime.timedelta(hours=5, minutes=30)
                    now_ist = now_utc + ist_offset
                    next_midnight_ist = datetime.datetime(now_ist.year, now_ist.month, now_ist.day) + datetime.timedelta(days=1)
                    next_midnight_utc = next_midnight_ist - ist_offset
                    unlock_time = next_midnight_utc.timestamp()
                    print(f"Key {key[:5]}... hit DAILY LIMIT. Penalized until Midnight IST.")
                else:
                    # Regular RPM limit or other 429/403 -> block for 60 seconds
                    unlock_time = time.time() + 60
                    print(f"Key {key[:5]}... hit RPM/403. Penalized for 60s.")
                    
                with state_lock:
                    key_penalties[key] = unlock_time
                    metrics["rate_limit_hits"] += 1
                    
                continue
                
            elif resp.status_code in [500, 503]:
                # Overload/500/503: Do NOT penalize or block keys. Try next combo.
                print(f"Model {actual_model} error ({resp.status_code}). Trying next combo without blocking key...")
                continue
            elif resp.status_code in [404, 400]:
                print(f"Model {actual_model} error ({resp.status_code}). Removing from active lists permanently.")
                with dynamic_models_lock:
                    global DYNAMIC_MODELS, OPENAI_MODELS_LIST
                    DYNAMIC_MODELS = [m for m in DYNAMIC_MODELS if m['name'] != actual_model]
                    OPENAI_MODELS_LIST = [m for m in OPENAI_MODELS_LIST if m['id'] != actual_model]
                max_retries += 1  # Give a free retry
                continue
            else:
                # Other non-200 responses
                with state_lock:
                    metrics["failed_requests"] += 1
                excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                out_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                               if name.lower() not in excluded_headers]
                return Response(resp.content, resp.status_code, out_headers)
                
        except Exception as e:
            print(f"API call error: {e}")
            continue
            
    # If all API key and model combos failed (e.g. all returned 500 or overload), return last response instead of blocking/failing wrongly
    if last_resp is not None:
        with state_lock:
            metrics["failed_requests"] += 1
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        out_headers = [(name, value) for (name, value) in last_resp.raw.headers.items()
                       if name.lower() not in excluded_headers]
        return Response(last_resp.content, last_resp.status_code, out_headers)
        
    return jsonify({"error": {"message": "All API keys are currently rate-limited (429) or exhausted. Please wait 1 minute.", "type": "rate_limit_error"}}), 429

@app.route('/v1/audio/transcriptions', methods=['POST', 'OPTIONS'])
def proxy_transcriptions():
    if request.method == 'OPTIONS':
        return Response(status=200)

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    audio_file = request.files['file']
    audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
    mime_type = audio_file.content_type or "audio/ogg"

    max_retries = len(API_KEYS) * len(get_active_models()) if API_KEYS and get_active_models() else 0
    attempts = 0

    while attempts < max_retries:
        key, model_name = get_next_available_combo()
        if not key:
            break
        attempts += 1
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Transcribe this audio. Output ONLY the exact text spoken, nothing else."},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_data
                        }
                    }
                ]
            }]
        }
        
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            if resp.status_code == 200:
                result = resp.json()
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return jsonify({"text": text.strip()})
                except KeyError:
                    return jsonify({"error": "Failed to parse transcription"}), 500
            elif resp.status_code in [429, 403]:
                with state_lock:
                    key_penalties[key] = time.time() + 120
                    metrics["rate_limit_hits"] += 1
                continue
            else:
                return Response(resp.content, resp.status_code)
                
        except Exception as e:
            print(f"Transcription error: {e}")
            continue

    return jsonify({"error": {"message": "All API keys are currently rate-limited (429) or exhausted.", "type": "rate_limit"}}), 429

@app.route('/v1/embeddings', methods=['POST', 'OPTIONS'])
def proxy_embeddings():
    if request.method == 'OPTIONS':
        return Response(status=200)
    
    data = request.json or {}
    data['model'] = 'gemini-embedding-2' # Force standard embedding model
    
    max_retries = len(API_KEYS) if API_KEYS else 0
    attempts = 0

    while attempts < max_retries:
        key, _ = get_next_available_combo()
        if not key:
            break
        attempts += 1
        
        url = "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        try:
            resp = requests.post(url, headers=headers, json=data)
            if resp.status_code == 200:
                with state_lock:
                    metrics["successful_api_calls"] += 1
                return jsonify(resp.json())
            elif resp.status_code in [429, 403]:
                with state_lock:
                    key_penalties[key] = time.time() + 120
                    metrics["rate_limit_hits"] += 1
                continue
            else:
                return Response(resp.content, resp.status_code)
        except Exception as e:
            print(f"Embedding error: {e}")
            continue
            
    return jsonify({"error": {"message": "All API keys are currently rate-limited (429) or exhausted.", "type": "rate_limit"}}), 429

@app.route('/v1/completions', methods=['POST', 'OPTIONS'])
def proxy_completions():
    if request.method == 'OPTIONS':
        return Response(status=200)
    
    data = request.json or {}
    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
        
    requested_model = data.get("model", "")
    chat_data = {
        "model": requested_model or "gemini-1.5-flash",
        "messages": [{"role": "user", "content": str(prompt)}]
    }
    
    is_round_robin = requested_model.lower() in ["gemini-pro", "auto", "default", "round-robin", "gemini-working-model", ""]
    max_retries = len(API_KEYS) * len(get_active_models()) if (API_KEYS and get_active_models() and is_round_robin) else (len(API_KEYS) if API_KEYS else 0)
    attempts = 0
    tried_keys = set()

    while attempts < max_retries:
        key, rr_model_name = get_next_available_combo()
        if not key:
            break
        attempts += 1
        
        if is_round_robin:
            actual_model = rr_model_name
        else:
            actual_model = requested_model
            if key in tried_keys:
                continue
            tried_keys.add(key)
            
        chat_data["model"] = actual_model
        
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        try:
            resp = requests.post(url, headers=headers, json=chat_data)
            if resp.status_code == 200:
                with state_lock:
                    metrics["successful_api_calls"] += 1
                
                chat_resp = resp.json()
                try:
                    text = chat_resp["choices"][0]["message"]["content"]
                    return jsonify({
                        "id": chat_resp.get("id", "cmpl-dummy"),
                        "object": "text_completion",
                        "created": chat_resp.get("created", int(time.time())),
                        "model": actual_model,
                        "choices": [
                            {"text": text, "index": 0, "finish_reason": "stop"}
                        ],
                        "usage": chat_resp.get("usage", {})
                    })
                except KeyError:
                    return jsonify(chat_resp)
            elif resp.status_code in [429, 403]:
                with state_lock:
                    key_penalties[key] = time.time() + 120
                    metrics["rate_limit_hits"] += 1
                continue
            else:
                return Response(resp.content, resp.status_code)
        except Exception as e:
            print(f"Completions error: {e}")
            continue
            
    return jsonify({"error": {"message": "All API keys are currently rate-limited (429) or exhausted.", "type": "rate_limit"}}), 429

@app.route('/v1/images/generations', methods=['POST', 'OPTIONS'])
def proxy_images():
    if request.method == 'OPTIONS':
        return Response(status=200)
    
    data = request.json or {}
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    
    return jsonify({
        "created": int(time.time()),
        "data": [{"url": image_url}]
    })

@app.route('/v1/audio/speech', methods=['POST', 'OPTIONS'])
def proxy_speech():
    if request.method == 'OPTIONS':
        return Response(status=200)
        
    data = request.json or {}
    text = data.get("input", "")
    
    if not text:
        return jsonify({"error": "Input text is required"}), 400
        
    encoded_text = urllib.parse.quote(text[:200])
    speech_url = f"http://translate.google.com/translate_tts?ie=UTF-8&total=1&idx=0&client=tw-ob&tl=en&q={encoded_text}"
    
    try:
        r = requests.get(speech_url)
        if r.status_code == 200:
            return Response(r.content, mimetype="audio/mpeg")
        else:
            return jsonify({"error": "TTS synthesis failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def proxy_models():
    if request.method == 'OPTIONS':
        return Response(status=200)
        
    with dynamic_models_lock:
        if OPENAI_MODELS_LIST:
            return jsonify({"object": "list", "data": OPENAI_MODELS_LIST})
            
    models_list = [
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-3.5-flash", "gemini-3.5-pro", "gemini-pro",
        "gemini-working-model", "gemini-1.5-pro-exp-0801", "gemini-1.5-pro-exp-0827",
        "gemini-1.5-flash-exp-0827", "gemini-1.5-flash-8b-exp-0827", "gemini-1.5-flash-8b-exp-0924",
        "text-embedding-004", "gemini-embedding-2", "dall-e-3", "whisper-1", "tts-1"
    ]
    
    data = []
    now = int(time.time())
    for m in models_list:
        data.append({
            "id": m,
            "object": "model",
            "created": now,
            "owned_by": "google"
        })
        
    return jsonify({
        "object": "list",
        "data": data
    })

@app.route('/add', methods=['POST'])
def add_key_model():
    expected_pass = os.environ.get("PASSWORD", "")
    if request.headers.get("Authorization") != f"Bearer {expected_pass}":
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.json
    with state_lock:
        if "key" in data and data["key"] not in API_KEYS:
            API_KEYS.append(data["key"])
        if "model" in data and "rpm" in data:
            MODELS.append({"name": data["model"], "rpm": int(data["rpm"])})
    return jsonify({"status": "success", "keys_count": len(API_KEYS), "models": MODELS})
    
@app.route('/status', methods=['GET'])
def get_status():
    now = time.time()
    penalized = {k[:5]+"..."+k[-5:]: round((ts - now)/60, 1) for k, ts in key_penalties.items()} # minutes left
    return jsonify({
        "metrics": metrics,
        "active_keys": len(API_KEYS),
        "penalized_keys_minutes_left": penalized,
        "models": MODELS
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8085)
