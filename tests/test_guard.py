from fastapi.testclient import TestClient

from forge import guard, llm_parser


def test_window_limits_and_recovers():
    w = guard.Window(3, 60)
    assert [w.allow("a", now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert w.allow("b", now=3)            # other clients unaffected
    assert w.allow("a", now=61.5)         # old hits expire


def test_claude_budget_per_ip_then_rules_only(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_parser, "_call_claude", lambda m: calls.append(m) or {"diets": []})
    monkeypatch.setattr(guard, "_claude_ip", guard.Window(2, 3600))
    llm_parser._CACHE.clear()
    guard.client_ip.set("1.2.3.4")
    assert llm_parser.llm_parse("no onion please") is not None
    assert llm_parser.llm_parse("no garlic please") is not None
    assert llm_parser.llm_parse("no ginger please") is None   # over budget -> rules alone
    assert len(calls) == 2


def test_exact_cache_only(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_parser, "_call_claude", lambda m: calls.append(m) or {"diets": []})
    llm_parser._CACHE.clear()
    guard.client_ip.set("5.6.7.8")
    llm_parser.llm_parse("paneer without onion")
    llm_parser.llm_parse("  Paneer   WITHOUT onion ")
    llm_parser.llm_parse("paneer with onion")  # looks similar, means the opposite: must not hit the cache
    assert calls == ["paneer without onion", "paneer with onion"]


def test_failed_call_is_not_cached(monkeypatch):
    results = iter([None, {"diets": []}])
    monkeypatch.setattr(llm_parser, "_call_claude", lambda m: next(results))
    llm_parser._CACHE.clear()
    guard.client_ip.set("9.9.9.9")
    assert llm_parser.llm_parse("hold the onions") is None
    assert llm_parser.llm_parse("hold the onions") is not None


def test_api_returns_429_over_request_limit(monkeypatch):
    from forge.api import app
    monkeypatch.setattr(guard, "requests", guard.Window(2, 60))
    client = TestClient(app)
    h = {"x-forwarded-for": "10.0.0.1"}
    codes = [client.post("/api/constraints", json={}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert client.get("/api/health", headers=h).status_code == 200  # health is never limited
    assert client.post("/api/constraints", json={}, headers={"x-forwarded-for": "10.0.0.2"}).status_code == 200
