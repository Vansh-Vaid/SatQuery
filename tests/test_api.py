from fastapi.testclient import TestClient

from satquery.api.app import app
from satquery.imagery.io import save_geotiff

client = TestClient(app)


def test_health_and_tools():
    assert client.get("/api/health").json()["status"] == "ok"
    names = [t["name"] for t in client.get("/api/tools").json()["tools"]]
    assert {"rs_vlm", "change_detector", "optical_sar_fusion", "region_grounder"} <= set(names)


def test_analyze_upload_and_reports(tmp_path, s2, s1):
    p2, p1 = save_geotiff(s2, tmp_path / "opt.tif"), save_geotiff(s1, tmp_path / "sar.tif")
    with p2.open("rb") as a, p1.open("rb") as b:
        r = client.post("/api/analyze", data={"query": "How much of the scene is water?"},
                        files={"image1": ("opt.tif", a), "image2": ("sar.tif", b)})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["task"] == "cross_modal_analysis" and d["viewer"]["images"] and d["viewer"]["overlays"]
    html = client.get(f"/api/report/{d['request_id']}.html")
    assert html.status_code == 200 and "Execution summary" in html.text
    assert client.get(f"/api/report/{d['request_id']}.json").json()["request_id"] == d["request_id"]


def test_analyze_demo():
    r = client.post("/api/analyze", data={"query": "What changed?", "demo": "bitemporal_s2"})
    assert r.status_code == 200 and r.json()["task"].startswith("change")


def test_rejects_bad_format(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with f.open("rb") as fh:
        r = client.post("/api/analyze", data={"query": "q"}, files={"image1": ("notes.txt", fh)})
    assert r.status_code == 415


def test_inspect_gate_returns_layers_and_footprint():
    r = client.post("/api/inspect", data={"demo": "cross_modal", "pair_type": "cross_modal"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["config"] == "cross_modal" and d["ok"]
    assert [im["modality"] for im in d["images"]] == ["multispectral", "sar"]
    assert d["images"][0]["bbox_wgs84"] and d["images"][0]["layers"][0]["png"].startswith("data:image/png")


def test_viewer_payload_and_report_bundle():
    import io
    import json
    import zipfile
    d = client.post("/api/analyze", data={"query": "Highlight the water body.", "demo": "single_s2"}).json()
    v = d["viewer"]
    assert v["width"] == 256 and v["boxes"] and v["overlays"] and v["geo"]["georeferenced"]
    rid = d["request_id"]
    g = json.loads(client.get(f"/api/report/{rid}.geojson").text)
    box = next(f for f in g["features"] if f["properties"]["kind"] == "evidence_box")
    lon, lat = box["geometry"]["coordinates"][0][0]
    assert 8.9 < lon < 9.1 and 47.8 < lat < 47.9          # synthetic scene sits in UTM 32N near 9E 47.85N
    assert client.get(f"/api/report/{rid}.pdf").content[:4] == b"%PDF"
    assert "## Execution trace" in client.get(f"/api/report/{rid}.md").text
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/report/{rid}.zip").content))
    assert set(z.namelist()) == {"report.pdf", "report.md", "report.html", "evidence.geojson", "trace.json"}


def test_unknown_report_404():
    assert client.get("/api/report/doesnotexist.html").status_code == 404
