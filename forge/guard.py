"""Abuse and spend protection for the public demo.

Two layers, both in memory per server instance (the Vercel firewall rule is the shared outer layer):
- request limit per IP for /api/* (429 when exceeded);
- a Claude budget per IP and per instance. Over budget, the app silently uses the rule parser alone,
  which is the safe baseline anyway: Claude can only add restrictions.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar

client_ip: ContextVar[str] = ContextVar("client_ip", default="local")

REQ_PER_MIN = int(os.getenv("RATE_REQ_PER_MIN", "40"))
CLAUDE_PER_IP_HOUR = int(os.getenv("CLAUDE_PER_IP_HOUR", "120"))
CLAUDE_PER_HOUR = int(os.getenv("CLAUDE_PER_HOUR", "600"))
CLAUDE_PER_DAY = int(os.getenv("CLAUDE_PER_DAY", "4000"))


class Window:
    """Sliding-window counters keyed by client, with bounded memory."""

    def __init__(self, limit: int, seconds: float, max_keys: int = 10_000):
        self.limit, self.seconds, self.max_keys = limit, seconds, max_keys
        self.hits: OrderedDict[str, deque] = OrderedDict()
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits.get(key)
            if q is None:
                q = self.hits[key] = deque()
                if len(self.hits) > self.max_keys:
                    self.hits.popitem(last=False)
            self.hits.move_to_end(key)
            while q and now - q[0] > self.seconds:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


requests = Window(REQ_PER_MIN, 60)
_claude_ip = Window(CLAUDE_PER_IP_HOUR, 3600)
_claude_hour = Window(CLAUDE_PER_HOUR, 3600)
_claude_day = Window(CLAUDE_PER_DAY, 86400)
stats = {"claude_calls": 0, "claude_skipped_budget": 0, "claude_cache_hits": 0, "rate_limited": 0}


def allow_request(ip: str) -> bool:
    ok = requests.allow(ip)
    if not ok:
        stats["rate_limited"] += 1
    return ok


def allow_claude() -> bool:
    ip = client_ip.get()
    # Check the per-IP budget first so one client can't drain the shared ones.
    ok = _claude_ip.allow(ip) and _claude_hour.allow("*") and _claude_day.allow("*")
    stats["claude_calls" if ok else "claude_skipped_budget"] += 1
    return ok


def ip_from(headers, fallback: str = "local") -> str:
    fwd = headers.get("x-forwarded-for", "")
    return (headers.get("x-real-ip") or fwd.split(",")[0]).strip() or fallback
