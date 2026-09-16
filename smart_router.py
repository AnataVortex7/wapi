"""
################################################################################
# WAPI - ADVANCED INTELLIGENT ROUTER (Future-Proof)
################################################################################
# HOW TO USE:
#
# 1. SETTING THE MASTER PASSWORD
#    Env Var: PASSWORD=YourSecretPass
#    (All API requests and /router/logs require this password)
#
# 2. ADDING KEYS FOR ANY PROVIDER (Future-Proof)
#    Just add comma-separated keys to the respective environment variables:
#    - GEMINI_API_KEYS=key1,key2,key3
#    - OPENAI_API_KEYS=sk-proj-...,sk-proj-...
#    - CLAUDE_API_KEYS=sk-ant-...,sk-ant-...
#    - HF_API_KEYS=hf_...,hf_...
#
# 3. SETTING THE "DEFAULT" ROUND-ROBIN POOL
#    When the user requests model "auto" (or no model), the proxy will 
#    round-robin across this exact pool.
#    Env Var: MODELS_POOL=gemini-1.5-flash:15,gpt-4o-mini:50,claude-3-haiku-20240307:20
#    (Format is model_name:requests_per_minute)
#
# 4. HOW IT WORKS:
#    - If you request /model custom/gpt-4o -> It automatically uses OPENAI_API_KEYS and OpenAI's API.
#    - If you request /model custom/claude-3-5-sonnet-20240620 -> It automatically uses CLAUDE_API_KEYS and translates to Anthropic's API.
#    - If a key fails or rate-limits, it instantly tries the next key.
#    - If all keys for a specific model fail, it falls back to your MODELS_POOL.
################################################################################
"""
import os, json, time, threading, datetime
from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string, g
from functools import wraps
from collections import deque
import requests


import os, json, time, threading, datetime
from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string, g
from functools import wraps
import requests

app = Flask(__name__)

# --- CONFIG & STATE ---
# Environment variables for keys
KEYS = {
    "gemini": [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()],
    "openai": [k.strip() for k in os.environ.get("OPENAI_API_KEYS", "").split(",") if k.strip()],
    "claude": [k.strip() for k in os.environ.get("CLAUDE_API_KEYS", "").split(",") if k.strip()],
    "huggingface": [k.strip() for k in os.environ.get("HF_API_KEYS", "").split(",") if k.strip()]
}

# The Round-Robin Model Pool (from environment)
RAW_MODELS = os.environ.get("MODELS_POOL", os.environ.get("GEMINI_MODELS", "gemini-1.5-flash:15")).split(",")
RR_MODELS = []
for m in RAW_MODELS:
    if not m.strip(): continue
    parts = m.split(":")
    name = parts[0].strip()
    rpm = int(parts[1].strip()) if len(parts) > 1 else 15
    RR_MODELS.append({"name": name, "rpm": rpm})

if not RR_MODELS:
    RR_MODELS = [{"name": "gemini-1.5-flash", "rpm": 15}]

key_penalties = {}  # key -> timestamp
request_history = {} # key -> list of timestamps
request_logs = deque(maxlen=150)

metrics = {
    "total_incoming_requests": 0,
    "successful_api_calls": 0,
    "rate_limit_hits": 0,
    "fallback_calls": 0,
    "failed_requests": 0,
    "usage_by_key": {}
}

state_lock = threading.Lock()
rr_key_idx = 0
rr_model_idx = 0

def clean_history(history):
    now = time.time()
    return [ts for ts in history if now - ts < 60.0]

def determine_provider(model_name):
    model_name = model_name.lower()
    if "gpt" in model_name or "o1" in model_name:
        return "openai", "https://api.openai.com/v1/chat/completions"
    elif "claude" in model_name or "anthropic" in model_name:
        # We will use Anthropic native API URL by default
        return "claude", os.environ.get("CLAUDE_BASE_URL", "https://api.anthropic.com/v1/messages")
    elif "meta" in model_name or "mistral" in model_name or "llama" in model_name:
        return "huggingface", os.environ.get("HF_BASE_URL", "https://api-inference.huggingface.co/models/" + model_name + "/v1/chat/completions")
    else:
        return "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" 


# Track key index per provider for round-robin
provider_key_idx = {"gemini": 0, "openai": 0, "claude": 0, "huggingface": 0}

def get_next_rr_combo():
    global rr_model_idx
    now = time.time()
    
    with state_lock:
        if not RR_MODELS: return None, None, None, None
        
        # Loop through models in RR_MODELS
        for _ in range(len(RR_MODELS)):
            model = RR_MODELS[rr_model_idx]
            rr_model_idx = (rr_model_idx + 1) % len(RR_MODELS)
            
            provider, url = determine_provider(model['name'])
            provider_keys = KEYS.get(provider, [])
            
            if not provider_keys:
                continue # We don't have keys for this provider, try the next model!
                
            # Now round-robin the keys for THIS provider
            p_idx = provider_key_idx.get(provider, 0)
            
            found_key = None
            for _ in range(len(provider_keys)):
                key = provider_keys[p_idx]
                p_idx = (p_idx + 1) % len(provider_keys)
                
                if key in key_penalties and now < key_penalties[key]:
                    continue
                    
                history = clean_history(request_history.get(key, []))
                request_history[key] = history
                if len(history) < model['rpm']:
                    history.append(now)
                    found_key = key
                    break
                    
            provider_key_idx[provider] = p_idx
            
            if found_key:
                return found_key, model['name'], provider, url
                
        return None, None, None, None

def get_key_for_provider(provider, rpm_limit=15):
    now = time.time()
    with state_lock:
        provider_keys = KEYS.get(provider, [])
        if not provider_keys: return None
        
        p_idx = provider_key_idx.get(provider, 0)
        
        for _ in range(len(provider_keys)):
            key = provider_keys[p_idx]
            p_idx = (p_idx + 1) % len(provider_keys)
            
            if key in key_penalties and now < key_penalties[key]:
                continue
            history = clean_history(request_history.get(key, []))
            request_history[key] = history
            if len(history) < rpm_limit:
                history.append(now)
                provider_key_idx[provider] = p_idx
                return key
                
        return None

# --- AUTH & LOGGING ---
def check_browser_auth(username, password):
    return password == os.environ.get("PASSWORD", "")

def requires_browser_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_browser_auth(auth.username, auth.password):
            return Response('Login Required', 401, {'WWW-Authenticate': 'Basic realm="Admin Login"'})
        return f(*args, **kwargs)
    return decorated

@app.before_request
def strict_password_and_log():
    if request.method == 'OPTIONS': return
    if request.path in ['/ping', '/healthz', '/logs']: return
        
    expected_pass = os.environ.get("PASSWORD", "")
    auth_header = request.headers.get("Authorization", "")
    is_correct = (auth_header == f"Bearer {expected_pass}")
    
    msg = ""
    if request.is_json:
        try:
            body = request.get_json(silent=True) or {}
            if "messages" in body and len(body.get("messages", [])) > 0:
                msg = body["messages"][-1].get("content", "")
        except: pass

    log_entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": request.headers.get("Cf-Connecting-Ip", request.remote_addr),
        "path": request.path,
        "password_used": "*** HIDDEN ***" if is_correct else auth_header,
        "is_correct": is_correct,
        "message": msg[:300] + "..." if len(msg)>300 else msg,
        "status": "Pending..."
    }
    g.log_entry = log_entry
    request_logs.appendleft(log_entry)
    
    if not is_correct:
        log_entry["status"] = "401 Blocked"
        return jsonify({"error": "Unauthorized Access."}), 401

@app.after_request
def update_log_status(response):
    if hasattr(g, 'log_entry') and g.log_entry["status"] == "Pending...":
        g.log_entry["status"] = f"{response.status_code} Success" if response.status_code == 200 else f"{response.status_code} Failed"
    return response

# --- ROUTER LOGIC ---
def make_api_call(data, key, model_name, url):
    data["model"] = model_name
    
    # Provider specific header and payload translation
    if "anthropic.com" in url:
        headers = {
            "x-api-key": key, 
            "anthropic-version": "2023-06-01", 
            "Content-Type": "application/json"
        }
        # Claude requires max_tokens natively
        if "max_tokens" not in data:
            data["max_tokens"] = 4096
    else:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, json=data, headers=headers, stream=True, timeout=60)
        if resp.status_code == 200:
            with state_lock:
                metrics["successful_api_calls"] += 1
            if hasattr(g, 'log_entry'):
                g.log_entry["backend_key"] = key[:5] + "..." + key[-3:] if len(key) > 8 else "***"
                g.log_entry["backend_model"] = model_name
            excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
            out_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
            return Response(resp.content, resp.status_code, out_headers)
        elif resp.status_code == 429:
            # Rate Limit: Block for 24 hours
            with state_lock:
                key_penalties[key] = time.time() + 86400
                metrics["rate_limit_hits"] += 1
            return None # Trigger fallback
        elif resp.status_code in [500, 503]:
            # High Demand / Server Error: Block for 5 minutes
            with state_lock:
                key_penalties[key] = time.time() + 300
            return None
        elif resp.status_code in [403, 400, 404]:
            # Wrong Model / Bad Request: DO NOT BLOCK at all!
            return None
        else:
            return None # Trigger fallback
    except Exception:
        return None # Trigger fallback

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy_chat():
    if request.method == 'OPTIONS': return Response(status=200)
    with state_lock: metrics["total_incoming_requests"] += 1
        
    data = request.json
    requested_model = data.get("model", "auto")
    
    # 1. SPECIFIC MODEL LOGIC
    if requested_model != "auto":
        provider, url = determine_provider(requested_model)
        
        # Try up to 3 keys for this specific provider
        for _ in range(3):
            key = get_key_for_provider(provider)
            if not key: break # No keys available for this provider
            
            response = make_api_call(data, key, requested_model, url)
            if response: return response
            
        print(f"Specific model {requested_model} failed. Falling back to ALL ROUND ROBIN.")
    
    # 2. ALL ROUND-ROBIN FALLBACK (or if 'auto' was requested)
    while True:
        key, rr_model_name, provider, url = get_next_rr_combo()
        if not key: break
        
        response = make_api_call(data, key, rr_model_name, url)
        if response: return response

    # 3. ABSOLUTE FALLBACK (Web2API)
    print("Falling back to Web2API...")
    with state_lock: metrics["fallback_calls"] += 1
    fallback_url = "http://127.0.0.1:8081/v1/chat/completions"
    fallback_headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']}
    
    try:
        resp = requests.post(fallback_url, json=data, headers=fallback_headers, timeout=60)
        if hasattr(g, 'log_entry'):
            g.log_entry["backend_key"] = "Web2API Fallback"
            g.log_entry["backend_model"] = "Fallback Model"
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        out_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
        return Response(resp.content, resp.status_code, out_headers)
    except:
        with state_lock: metrics["failed_requests"] += 1
        return jsonify({"error": {"message": "All APIs failed."}}), 500

@app.route('/logs', methods=['GET'])
@requires_browser_auth
def view_secure_logs():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Secure API Logs</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="5">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #121212; color: #e0e0e0; }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: #ffffff; text-align: center; margin-bottom: 30px; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
            .table-wrapper { background: #1e1e1e; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); overflow-x: auto; border: 1px solid #333; }
            table { width: 100%; border-collapse: collapse; min-width: 900px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #333; }
            th { background-color: #2c2c2c; color: #ffffff; font-weight: 600; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.5px; }
            tr:hover { background-color: #252525; }
            .status-correct { color: #4ade80; font-weight: bold; }
            .status-wrong { color: #f87171; font-weight: bold; }
            .msg { max-width: 400px; white-space: pre-wrap; word-break: break-word; font-size: 0.9em; color: #a1a1aa; background: #18181b; padding: 8px; border-radius: 4px; }
            .pwd-wrong { font-family: monospace; color: #fca5a5; background: #451a1a; padding: 3px 6px; border-radius: 3px; font-size: 0.9em; }
            .pwd-correct { font-family: monospace; color: #4ade80; font-style: italic; font-size: 0.9em; }
            .ip { font-family: monospace; color: #93c5fd; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
            .bg-green { background: rgba(74, 222, 128, 0.2); color: #4ade80; }
            .bg-red { background: rgba(248, 113, 113, 0.2); color: #f87171; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔐 Secure API Access Logs</h1>
            <p style="text-align: center; color: #888;">Showing up to last 150 requests.</p>
            <div class="table-wrapper">
                {% if logs %}
                <table>
                    <tr>
                        <th>Time</th>
                        <th>IP Address</th>
                        <th>Attempted Password</th>
                        <th>Backend Model & Key</th>
                        <th>Status</th>
                        <th>Message / Prompt</th>
                    </tr>
                    {% for log in logs %}
                    <tr>
                        <td style="white-space: nowrap; color: #888; font-size: 0.9em;">{{ log.time }}</td>
                        <td class="ip">{{ log.ip }}</td>
                        <td>
                            <div style='font-size: 0.85em; color: #a1a1aa; background: #252525; padding: 4px; border-radius: 4px; text-align: center; margin-bottom: 5px;'>
                                <strong>{{ log.backend_model or 'N/A' }}</strong><br>
                                <span style='font-family: monospace; color: #facc15;'>{{ log.backend_key or 'N/A' }}</span>
                            </div>
                        </td>
                        <td>
                            {% if log.is_correct %}
                                <span class="pwd-correct">🛡️ {{ log.password_used }}</span>
                            {% else %}
                                <span class="pwd-wrong">{{ log.password_used or "NONE" }}</span>
                            {% endif %}
                        </td>
                        <td>
                            <div style='font-size: 0.85em; color: #a1a1aa; background: #252525; padding: 4px; border-radius: 4px; text-align: center; margin-bottom: 5px;'>
                                <strong>{{ log.backend_model or 'N/A' }}</strong><br>
                                <span style='font-family: monospace; color: #facc15;'>{{ log.backend_key or 'N/A' }}</span>
                            </div>
                        </td>
                        <td>
                            {% if log.is_correct %}
                                <span class="badge bg-green">{{ log.status }}</span>
                            {% else %}
                                <span class="badge bg-red">{{ log.status }}</span>
                            {% endif %}
                        </td>
                        <td><div class="msg">{{ log.message or "No message" }}</div></td>
                    </tr>
                    {% endfor %}
                </table>
                {% else %}
                <div style="text-align: center; padding: 50px; color: #666;">No requests logged yet.</div>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, logs=list(request_logs))

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def proxy_models():
    if request.method == 'OPTIONS': return Response(status=200)
    
    # Return the actual models configured in the environment
    models_data = []
    for m in RR_MODELS:
        models_data.append({
            "id": m["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google" if "gemini" in m["name"] else "openai"
        })
        
    return jsonify({
        "object": "list",
        "data": models_data
    })

@app.route('/add', methods=['POST'])
def add_key_model():
    expected_pass = os.environ.get("PASSWORD", "")
    if request.headers.get("Authorization") != f"Bearer {expected_pass}":
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.json
    with state_lock:
        # Defaults to adding gemini keys if provider not specified
        provider = data.get("provider", "gemini")
        if provider not in KEYS: KEYS[provider] = []
        if "key" in data and data["key"] not in KEYS[provider]:
            KEYS[provider].append(data["key"])
        if "model" in data and "rpm" in data:
            RR_MODELS.append({"name": data["model"], "rpm": int(data["rpm"])})
    return jsonify({"status": "success", "keys_count": {p: len(k) for p,k in KEYS.items()}, "models": RR_MODELS})
    
@app.route('/clear', methods=['GET'])
def clear_penalties():
    with state_lock:
        key_penalties.clear()
    return jsonify({"status": "cleared"})
    
@app.route('/status', methods=['GET'])
def get_status():
    now = time.time()
    penalized = {k[:5]+"...": round((300 - (now - ts))/3600, 1) for k, ts in key_penalties.items()}
    return jsonify({
        "metrics": metrics,
        "active_keys": {p: len(k) for p,k in KEYS.items()},
        "penalized_keys_hours_left": penalized,
        "models": RR_MODELS
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8085)
