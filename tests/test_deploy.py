"""Public-deployment guards: upload limits, rate limit, busy limit, query length, demo cache, headers."""
import dataclasses
import threading

import pytest
from fastapi.testclient import TestClient

import satquery.api.app as appmod

client = TestClient(appmod.app)


@pytest.fixture()
def cfg(monkeypatch):
    def set_(**kw):
        monkeypatch.setattr(appmod, "settings", dataclasses.replace(appmod.settings, **kw))
    return set_


def _tif(n_bytes: int):
    return {"image1": ("big.tif", b"\0" * n_bytes, "image/tiff")}


def test_upload_over_limit_is_rejected_while_streaming(cfg):
    cfg(max_upload_mb=1)
    r = client.post("/api/analyze", data={"query": "Is there water?"}, files=_tif(1_600_000))
    assert r.status_code == 413 and "exceeds 1 MB" in r.json()["detail"]


def test_request_body_over_limit_is_rejected_before_reading(cfg):
    cfg(max_upload_mb=0)  # body limit becomes 1 MB
    r = client.post("/api/analyze", data={"query": "Is there water?"}, files=_tif(1_200_000))
    assert r.status_code == 413 and "Upload too large" in r.json()["detail"]


def test_rate_limit_per_client(monkeypatch):
    monkeypatch.setattr(appmod, "limiter", appmod.RateLimiter(2))
    codes = [client.post("/api/inspect", data={"demo": "single_s2"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_busy_server_answers_503_instead_of_queueing_forever(cfg, monkeypatch):
    cfg(queue_timeout_s=0.1)
    sem = threading.BoundedSemaphore(1)
    sem.acquire()  # the only slot is taken by "another analysis"
    monkeypatch.setattr(appmod, "slots", sem)
    r = client.post("/api/inspect", data={"demo": "single_s2"})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"


def test_query_length_is_bounded(cfg):
    cfg(max_query_chars=20)
    r = client.post("/api/analyze", data={"query": "x" * 21, "demo": "single_s2"})
    assert r.status_code == 400


def test_demo_results_are_shared_and_not_rate_limited(monkeypatch):
    q = {"query": "How much of this scene is vegetation?", "demo": "single_s2"}
    first = client.post("/api/analyze", data=q).json()
    monkeypatch.setattr(appmod, "limiter", appmod.RateLimiter(1))
    appmod.limiter.allow("testclient")  # this client has now used its whole quota
    again = client.post("/api/analyze", data=q)
    assert again.status_code == 200 and again.json()["request_id"] == first["request_id"]
    assert client.get(f"/api/report/{first['request_id']}.json").status_code == 200


def test_security_headers():
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "referrer-policy" in r.headers
