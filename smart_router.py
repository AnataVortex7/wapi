import os, json, time, threading, datetime
from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string
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



@app.before_request
def strict_password_check():
    if request.method == 'OPTIONS':
        return
        
    # Allowed routes without password (if any)
    if request.path in ['/ping', '/healthz']:
        return
        
    expected_pass = os.environ.get("PASSWORD", "")
    # If the user also wants to support API_PASSWORD for backward compatibility, uncomment:
    # if not expected_pass: expected_pass = os.environ.get("API_PASSWORD", "Swapnpurti@1181")
    
    auth_header = request.headers.get("Authorization", "")
    
    if auth_header != f"Bearer {expected_pass}":
        return jsonify({"error": "Unauthorized Access. Invalid Password."}), 401




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
