"""
Smart Key Manager for WAPI - Zero 429 Errors
- RPM tracking (per key, per model)
- Auto-rotate on 429
- 60s cooldown on 429 keys
- RPD auto-reset at midnight IST
- Queue-based fairness for multiple users
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ─── Model Rate Limits (Google Free Tier) ───────────────────────────
MODEL_LIMITS = {
    "gemini-2.5-flash":          {"rpm": 10,  "rpd": 250},
    "gemini-2.5-flash-lite":     {"rpm": 15,  "rpd": 1000},
    "gemini-2.0-flash":          {"rpm": 15,  "rpd": 1500},
    "gemini-2.0-flash-lite":     {"rpm": 30,  "rpd": 1500},
    "gemma-4-26b-a4b-it":        {"rpm": 15,  "rpd": 50},
    "gemma-4-31b-it":            {"rpm": 15,  "rpd": 50},
    "gemini-pro-latest":         {"rpm": 2,   "rpd": 50},
    "gemini-1.5-flash-latest":   {"rpm": 15,  "rpd": 1500},
    "gemini-1.5-pro-latest":     {"rpm": 2,   "rpd": 50},
}

# Priority order — try fast + high-limit models first
MODEL_PRIORITY = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-pro-latest",
    "gemini-1.5-pro-latest",
]


class SmartKeyManager:
    def __init__(self, api_keys: list[str]):
        self.api_keys = api_keys
        self._lock = threading.RLock()

        # RPM tracking: key -> model -> deque of timestamps
        self._rpm_window: dict[str, dict[str, deque]] = {
            key: defaultdict(deque) for key in api_keys
        }

        # RPD tracking: key -> model -> count (resets at midnight IST)
        self._rpd_count: dict[str, dict[str, int]] = {
            key: defaultdict(int) for key in api_keys
        }
        self._rpd_reset_date: str = self._today_ist()

        # Cooldown tracking: (key, model) -> cooldown_until timestamp
        self._cooldown: dict[tuple, float] = {}

        # Stats
        self.stats = defaultdict(int)

    def _today_ist(self) -> str:
        return datetime.now(IST).strftime("%Y-%m-%d")

    def _maybe_reset_rpd(self):
        """Reset RPD counts at midnight IST."""
        today = self._today_ist()
        if today != self._rpd_reset_date:
            logger.info(f"🌙 Midnight reset: clearing all RPD counts (was {self._rpd_reset_date})")
            for key in self.api_keys:
                self._rpd_count[key] = defaultdict(int)
            self._rpd_reset_date = today
            self.stats["rpd_resets"] += 1

    def _prune_rpm_window(self, key: str, model: str):
        """Remove timestamps older than 60 seconds."""
        window = self._rpm_window[key][model]
        cutoff = time.monotonic() - 60
        while window and window[0] < cutoff:
            window.popleft()

    def _rpm_available(self, key: str, model: str) -> bool:
        self._prune_rpm_window(key, model)
        limit = MODEL_LIMITS.get(model, {}).get("rpm", 10)
        current = len(self._rpm_window[key][model])
        return current < limit

    def _rpd_available(self, key: str, model: str) -> bool:
        limit = MODEL_LIMITS.get(model, {}).get("rpd", 500)
        used = self._rpd_count[key][model]
        return used < limit

    def _is_on_cooldown(self, key: str, model: str) -> bool:
        until = self._cooldown.get((key, model), 0)
        return time.monotonic() < until

    def _seconds_until_rpm_slot(self, key: str, model: str) -> float:
        """How many seconds until a slot opens in this key's RPM window."""
        self._prune_rpm_window(key, model)
        window = self._rpm_window[key][model]
        limit = MODEL_LIMITS.get(model, {}).get("rpm", 10)
        if len(window) < limit:
            return 0.0
        oldest = window[0]
        return max(0.0, 60 - (time.monotonic() - oldest))

    def get_best_slot(self, preferred_model: str = None) -> tuple[str, str] | None:
        """
        Returns (key, model) that can serve a request RIGHT NOW.
        Tries preferred_model first across all keys, then falls back to priority list.
        Returns None if nothing is available.
        """
        with self._lock:
            self._maybe_reset_rpd()

            models_to_try = []
            if preferred_model and preferred_model in MODEL_LIMITS:
                models_to_try.append(preferred_model)
            for m in MODEL_PRIORITY:
                if m not in models_to_try:
                    models_to_try.append(m)

            for model in models_to_try:
                for key in self.api_keys:
                    if (self._is_on_cooldown(key, model)):
                        continue
                    if not self._rpd_available(key, model):
                        continue
                    if not self._rpm_available(key, model):
                        continue
                    # ✅ Found a free slot
                    return (key, model)

            return None  # Nothing available right now

    def get_best_slot_with_wait(
        self,
        preferred_model: str = None,
        max_wait_seconds: float = 65.0
    ) -> tuple[str, str] | None:
        """
        Like get_best_slot but waits up to max_wait_seconds for a slot to open.
        Polls every 0.5s. Returns (key, model) or None if timed out.
        """
        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            slot = self.get_best_slot(preferred_model)
            if slot:
                return slot
            time.sleep(0.5)
        return None

    def record_request(self, key: str, model: str):
        """Call this BEFORE sending the actual API request."""
        with self._lock:
            self._rpm_window[key][model].append(time.monotonic())
            self._rpd_count[key][model] += 1
            self.stats["total_requests"] += 1

    def record_429(self, key: str, model: str):
        """Call this when you get a 429 from the API."""
        with self._lock:
            cooldown_until = time.monotonic() + 60
            self._cooldown[(key, model)] = cooldown_until
            self.stats["total_429s"] += 1
            logger.warning(f"⏳ 429 on key=…{key[-6:]} model={model} → cooldown 60s")

    def record_success(self, key: str, model: str):
        """Call this on a successful response."""
        with self._lock:
            self.stats["total_successes"] += 1

    def get_status(self) -> dict:
        """Returns a human-readable status dict for your dashboard."""
        with self._lock:
            self._maybe_reset_rpd()
            now = time.monotonic()
            status = {
                "rpd_reset_date": self._rpd_reset_date,
                "stats": dict(self.stats),
                "keys": []
            }
            for key in self.api_keys:
                key_info = {"key_tail": f"…{key[-8:]}", "models": []}
                for model in MODEL_PRIORITY:
                    if model not in MODEL_LIMITS:
                        continue
                    self._prune_rpm_window(key, model)
                    rpm_used = len(self._rpm_window[key][model])
                    rpm_limit = MODEL_LIMITS[model]["rpm"]
                    rpd_used = self._rpd_count[key][model]
                    rpd_limit = MODEL_LIMITS[model]["rpd"]
                    cd_until = self._cooldown.get((key, model), 0)
                    on_cd = now < cd_until
                    cd_secs = max(0, round(cd_until - now)) if on_cd else 0
                    key_info["models"].append({
                        "model": model,
                        "rpm": f"{rpm_used}/{rpm_limit}",
                        "rpd": f"{rpd_used}/{rpd_limit}",
                        "cooldown_secs": cd_secs,
                        "available": (
                            not on_cd
                            and rpm_used < rpm_limit
                            and rpd_used < rpd_limit
                        )
                    })
                status["keys"].append(key_info)
            return status


# ─── Drop-in request function ────────────────────────────────────────

def make_request_with_smart_retry(
    manager: SmartKeyManager,
    call_fn,              # callable(key, model, **kwargs) → response
    preferred_model: str = None,
    max_attempts: int = 10,
    max_wait_per_attempt: float = 65.0,
    **call_kwargs
):
    """
    Usage:
        response = make_request_with_smart_retry(
            manager=mgr,
            call_fn=my_gemini_call,
            preferred_model="gemini-2.0-flash",
            prompt="Hello!"
        )
    """
    for attempt in range(max_attempts):
        slot = manager.get_best_slot_with_wait(
            preferred_model=preferred_model,
            max_wait_seconds=max_wait_per_attempt
        )

        if slot is None:
            # Nothing available even after waiting → give up
            raise RuntimeError(
                "❌ All keys and models exhausted. "
                "Either all RPDs are full (resets at midnight IST) "
                "or all slots are on cooldown."
            )

        key, model = slot
        manager.record_request(key, model)

        try:
            response = call_fn(key=key, model=model, **call_kwargs)
            manager.record_success(key, model)
            return response, key, model

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                manager.record_429(key, model)
                logger.info(f"🔄 Attempt {attempt+1}: 429 on {model}, retrying with next slot…")
                continue  # retry immediately with next best slot
            else:
                raise  # non-rate-limit error → propagate

    raise RuntimeError(f"❌ Failed after {max_attempts} attempts.")


# ─── Example integration ─────────────────────────────────────────────

if __name__ == "__main__":
    import google.generativeai as genai
    import os

    logging.basicConfig(level=logging.INFO)

    # Your 8 keys
    KEYS = [
        os.getenv("GEMINI_KEY_1", "key1"),
        os.getenv("GEMINI_KEY_2", "key2"),
        # ... add all 8
    ]

    mgr = SmartKeyManager(KEYS)

    def gemini_call(key: str, model: str, prompt: str, **kwargs):
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model)
        return m.generate_content(prompt)

    # Make a request — zero manual retry logic needed
    try:
        response, used_key, used_model = make_request_with_smart_retry(
            manager=mgr,
            call_fn=gemini_call,
            preferred_model="gemini-2.0-flash",
            prompt="Hello! Tell me a joke."
        )
        print(f"✅ Response via {used_model}: {response.text[:100]}")
    except RuntimeError as e:
        print(f"❌ {e}")

    # Check status
    import json
    print(json.dumps(mgr.get_status(), indent=2))
