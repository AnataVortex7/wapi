import time
import threading
import datetime
import requests

class SmartKeyManager:
    def __init__(self, keys, get_models_fn):
        self.keys = keys
        self.get_models_fn = get_models_fn  
        self.key_penalties = {}  # (key, model) -> unlock timestamp (daily)
        self.short_cooldowns = {} # (key, model) -> unlock timestamp (minute)
        self.request_history = {} # (key, model) -> list of timestamps
        self.lock = threading.Lock()
        
        self.current_idx = 0
        
    def _prune_history(self, pen_key, now):
        hist = self.request_history.get(pen_key, [])
        fresh = [ts for ts in hist if now - ts < 60.0]
        self.request_history[pen_key] = fresh
        return len(fresh)

    def get_best_combo(self, preferred_model):
        now = time.time()
        with self.lock:
            models = self.get_models_fn()
            if not self.keys or not models:
                return None, None
                
            M = len(models)
            K = len(self.keys)
            
            is_round_robin = preferred_model.lower() in ["gemini-pro", "auto", "default", "round-robin", "gemini-working-model", ""]
            
            valid_combos = []
            cooling_combos = []
            
            for i in range(K * M):
                idx = (self.current_idx + i) % (K * M)
                k_idx = idx // M
                m_idx = idx % M
                
                key = self.keys[k_idx]
                model = models[m_idx]
                actual_model = model['name'] if is_round_robin else preferred_model
                
                pen_key = (key, actual_model)
                
                # Check Daily Penalty
                if pen_key in self.key_penalties and now < self.key_penalties[pen_key]:
                    continue
                    
                # Check RPM Headroom
                used_rpm = self._prune_history(pen_key, now)
                rpm_limit = model.get('rpm', 15)
                
                if used_rpm >= rpm_limit:
                    continue # SKIP maxed out keys to prevent 429
                    
                # Check Short Cooldown
                if pen_key in self.short_cooldowns and now < self.short_cooldowns[pen_key]:
                    cooling_combos.append((key, actual_model, idx))
                else:
                    valid_combos.append((key, actual_model, idx))
            
            best = None
            if valid_combos:
                best = valid_combos[0]
            elif cooling_combos:
                # If everything is cooling down, try the one that will cool down first
                cooling_combos.sort(key=lambda x: self.short_cooldowns.get((x[0], x[1]), 0))
                best = cooling_combos[0]
                
            if best:
                key, actual_model, idx = best
                self.current_idx = (idx + 1) % (K * M) 
                self.request_history.setdefault((key, actual_model), []).append(now)
                return key, actual_model
                
            return None, None
            
    def mark_success(self, key, model):
        with self.lock:
            self.short_cooldowns.pop((key, model), None)

    def mark_failure(self, key, model, status_code, response_text):
        now = time.time()
        pen_key = (key, model)
        
        with self.lock:
            if status_code in [429, 403]:
                error_text = response_text.lower()
                if "per day" in error_text:
                    now_utc = datetime.datetime.utcnow()
                    ist_offset = datetime.timedelta(hours=5, minutes=30)
                    now_ist = now_utc + ist_offset
                    next_midnight_ist = datetime.datetime(now_ist.year, now_ist.month, now_ist.day) + datetime.timedelta(days=1)
                    next_midnight_utc = next_midnight_ist - ist_offset
                    self.key_penalties[pen_key] = next_midnight_utc.timestamp()
                else:
                    self.short_cooldowns[pen_key] = now + 90 


def make_request_with_smart_retry(manager, call_fn, preferred_model, max_retries=5):
    last_resp = None
    
    for attempt in range(max_retries):
        key, actual_model = manager.get_best_combo(preferred_model)
        
        if not key or not actual_model:
            break 
            
        try:
            resp = call_fn(key, actual_model)
            last_resp = resp
            
            if resp.status_code == 200:
                manager.mark_success(key, actual_model)
                return resp, key, actual_model
                
            elif resp.status_code in [429, 403]:
                manager.mark_failure(key, actual_model, resp.status_code, resp.text if hasattr(resp, 'text') else "")
                continue 
                
            elif resp.status_code in [500, 503]:
                continue 
                
            else:
                return resp, key, actual_model
                
        except Exception as e:
            print(f"Call failed: {e}")
            continue
            
    return last_resp, None, None
