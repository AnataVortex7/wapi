import os, json, time, threading, datetime
from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string, g
from functools import wraps
from collections import deque
import requests

app = Flask(__name__)

# Config loaded from Environment
RAW_KEYS = os.environ.get("GEMINI_API_KEYS", "").split(",")
RAW_MODELS = os.environ.get("GEMINI_MODELS", "gemini-1.5-flash:15,gemini-1.5-pro:2").split(",")

API_KEYS = [k.strip() for k in RAW_KEYS if k.strip()]
MODELS = []
for m in RAW_MODELS:
    if not m.strip(): continue
    parts = m.split(":")
    name = parts[0].strip()
    rpm = int(parts[1].strip()) if len(parts) > 1 else 5
    MODELS.append({"name": name, "rpm": rpm})

if not MODELS:
    MODELS = [{"name": "gemini-1.5-flash", "rpm": 15}]

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
        if not API_KEYS or not MODELS:
            return None, None
        
        total_combos = len(API_KEYS) * len(MODELS)
        for _ in range(total_combos):
            key = API_KEYS[current_key_idx]
            model = MODELS[current_model_idx]
            
            # Advance pointers
            current_model_idx += 1
            if current_model_idx >= len(MODELS):
                current_model_idx = 0
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
            
            # Check 24-hour penalty (86400 seconds)
            if key in key_penalties:
                if now - key_penalties[key] < 86400:
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
    if request.path in ['/ping', '/healthz', '/logs']:
        return
        
    expected_pass = os.environ.get("PASSWORD", "")
    auth_header = request.headers.get("Authorization", "")
    
    is_correct = (auth_header == f"Bearer {expected_pass}")
    
    # Extract message
    msg = ""
    if request.is_json:
        try:
            body = request.get_json(silent=True) or {}
            if "messages" in body and isinstance(body["messages"], list) and len(body["messages"]) > 0:
                msg = body["messages"][-1].get("content", "")
        except:
            pass

    log_entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": request.headers.get("Cf-Connecting-Ip", request.headers.get("X-Forwarded-For", request.remote_addr)),
        "path": request.path,
        "password_used": "*** HIDDEN (CORRECT) ***" if is_correct else auth_header,
        "is_correct": is_correct,
        "message": msg[:300] + "..." if len(msg) > 300 else msg,
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

@app.route('/logs', methods=['GET'])
@requires_browser_auth
def view_secure_logs():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Secure API Logs</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
                        <th>Status</th>
                        <th>Message / Prompt</th>
                    </tr>
                    {% for log in logs %}
                    <tr>
                        <td style="white-space: nowrap; color: #888; font-size: 0.9em;">{{ log.time }}</td>
                        <td class="ip">{{ log.ip }}</td>
                        <td>
                            {% if log.is_correct %}
                                <span class="pwd-correct">🛡️ {{ log.password_used }}</span>
                            {% else %}
                                <span class="pwd-wrong">{{ log.password_used or "NONE" }}</span>
                            {% endif %}
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

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy_chat():
    if request.method == 'OPTIONS':
        return Response(status=200)
        
    with state_lock:
        metrics["total_incoming_requests"] += 1
        
    data = request.json
    original_auth = request.headers.get("Authorization", "")
    
    # Try Real APIs
    while True:
        key, model_name = get_next_available_combo()
        if not key:
            break
            
        data["model"] = model_name
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        
        try:
            resp = requests.post(url, json=data, headers=headers, stream=True)
            if resp.status_code == 200:
                with state_lock:
                    metrics["successful_api_calls"] += 1
                    safe_key = key[:5] + "..."
                    if safe_key not in metrics["usage_by_key"]:
                        metrics["usage_by_key"][safe_key] = {}
                    if model_name not in metrics["usage_by_key"][safe_key]:
                        metrics["usage_by_key"][safe_key][model_name] = 0
                    metrics["usage_by_key"][safe_key][model_name] += 1
                    
                excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                out_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                               if name.lower() not in excluded_headers]
                return Response(resp.content, resp.status_code, out_headers)
            elif resp.status_code in [429, 403]:
                with state_lock:
                    key_penalties[key] = time.time()
                    metrics["rate_limit_hits"] += 1
                print(f"Key {key[:5]}... hit 429/403. Penalized for 24h.")
                continue
            elif resp.status_code in [500, 503]:
                print(f"Model {model_name} overloaded (500/503). Trying next combo...")
                continue
            else:
                with state_lock:
                    metrics["failed_requests"] += 1
                return Response(resp.content, resp.status_code)
        except Exception as e:
            print(f"API call error: {e}")
            continue
            
    # FALLBACK to Web2API
    print("Falling back to Web2API on port 8081...")
    with state_lock:
        metrics["fallback_calls"] += 1
        
    fallback_url = "http://127.0.0.1:8081/v1/chat/completions"
    
    # Forward all original headers transparently (except host)
    fallback_headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']}
    
        
    try:
        resp = requests.post(fallback_url, json=request.json, headers=fallback_headers)
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        out_headers = [(name, value) for (name, value) in resp.raw.headers.items()
                       if name.lower() not in excluded_headers]
        return Response(resp.content, resp.status_code, out_headers)
    except Exception as e:
        with state_lock:
            metrics["failed_requests"] += 1
        return jsonify({"error": {"message": "All APIs failed and Fallback is down.", "type": "server_error"}}), 500

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def proxy_models():
    if request.method == 'OPTIONS':
        return Response(status=200)
    # Dummy models response to keep Hermes happy if it probes the endpoint
    return jsonify({
        "object": "list",
        "data": [{"id": "gemini-1.5-flash", "object": "model", "created": int(time.time()), "owned_by": "google"}]
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
    penalized = {k[:5]+"...": round((86400 - (now - ts))/3600, 1) for k, ts in key_penalties.items()}
    return jsonify({
        "metrics": metrics,
        "active_keys": len(API_KEYS),
        "penalized_keys_hours_left": penalized,
        "models": MODELS
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8085)
