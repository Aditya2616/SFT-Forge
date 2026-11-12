from __future__ import annotations

import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from conftest import HashEmbedder, StubToxicity
from sftforge import cli
from sftforge.models import Models
from sftforge.server import app as server


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "_models", Models(server._models_cfg.models, embedder=HashEmbedder(), toxicity=StubToxicity()))
    return TestClient(server.app)


def jsonl(n: int = 6) -> str:
    topics = ["rivers", "mountains", "cooking", "music", "football", "chess", "gardening", "planets"]
    rows = [{"instruction": f"Tell me about {topics[i % 8]} in detail {i}", "output": f"{topics[i % 8].title()} are a wide subject with many facts, number {i}.",
             "category": "open_qa" if i % 2 else "general_qa"} for i in range(n)]
    rows.append({"instruction": "Say something about colours", "output": "As an AI language model I like blue."})
    return "\n".join(json.dumps(r) for r in rows)


def wait(client, job_id: str) -> dict:
    for _ in range(100):
        d = client.get(f"/api/run/{job_id}").json()
        if d["status"] in ("done", "error"):
            return d
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_health(client):
    d = client.get("/health").json()
    assert d["status"] == "ok" and d["max_rows"] > 0


def test_live_run_returns_records_with_reasons(client):
    job = client.post("/api/run", json={"text": jsonl()}).json()
    d = wait(client, job["id"])
    assert d["status"] == "done", d
    recs = d["result"]["records"]
    assert len(recs) == 7 and d["result"]["meta"]["label"] == "Live run"
    bad = [r for r in recs if r["s"] == "x"]
    assert len(bad) == 1 and bad[0]["g"] == "quality" and bad[0]["rs"][0]["c"] == "refusal_boilerplate"
    assert sum(g["rejected"] for g in d["result"]["audit"]["gates"]) == 1


def test_export_zip(client):
    d = wait(client, client.post("/api/run", json={"text": jsonl()}).json()["id"])
    r = client.get(f"/api/run/{d['id']}/export.zip")
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert {"train.jsonl", "rejects.jsonl", "audit.json", "DATASET_CARD.md"} <= set(names)


def test_threshold_override_changes_outcome_and_models_are_locked(client):
    over = {"quality": {"refusal_boilerplate": False}, "models": {"embedder": "evil/model"}, "export": {"chat_template": "x"}}
    d = wait(client, client.post("/api/run", json={"text": jsonl(), "config": over}).json()["id"])
    assert d["status"] == "done"
    assert d["result"]["audit"]["config"]["models"]["embedder"] != "evil/model"
    assert d["result"]["audit"]["config"]["export"]["chat_template"] == "chatml"
    assert not [r for r in d["result"]["records"] if r["s"] == "x" and r["rs"][0]["c"] == "refusal_boilerplate"]


def test_invalid_config_is_422(client):
    assert client.post("/api/run", json={"text": jsonl(), "config": {"quality": {"nope": 1}}}).status_code == 422


def test_limits(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 500)
    assert client.post("/api/run", json={"text": "x" * 600}).status_code == 413
    assert client.post("/api/run/upload", files={"file": ("a.jsonl", b"y" * 600)}).status_code == 413
    monkeypatch.setattr(server, "MAX_BYTES", 2_000_000)
    monkeypatch.setattr(server, "MAX_ROWS", 3)
    d = wait(client, client.post("/api/run", json={"text": jsonl()}).json()["id"])
    assert d["status"] == "error" and "limit" in d["error"]


def test_bad_input_reports_error_not_crash(client):
    d = wait(client, client.post("/api/run", json={"text": "this is not json\n{broken"}).json()["id"])
    assert d["status"] == "error"
    assert client.get("/api/run/nope").status_code == 404


def test_upload_and_non_utf8(client):
    job = client.post("/api/run/upload", files={"file": ("d.jsonl", jsonl().encode())}).json()
    assert wait(client, job["id"])["status"] == "done"
    assert client.post("/api/run/upload", files={"file": ("d.jsonl", b"\xff\xfe\x00bad")}).status_code == 400


def test_cli_run_writes_outputs(tmp_path, monkeypatch):
    from sftforge import pipeline

    src = tmp_path / "in.jsonl"
    src.write_text(jsonl(10), encoding="utf-8")
    real = pipeline.run

    def stubbed(raw, cfg=None, models=None, progress=None, load_errors=0):
        return real(raw, cfg, Models(cfg.models, embedder=HashEmbedder(), toxicity=StubToxicity()), progress, load_errors)

    monkeypatch.setattr(pipeline, "run", stubbed)
    out = tmp_path / "out"
    assert cli.main(["run", str(src), "--out", str(out), "--recorded-dir", str(tmp_path / "rec")]) == 0
    assert (out / "audit.json").exists() and (out / "train.jsonl").exists()
    index = json.loads((tmp_path / "rec" / "index.json").read_text(encoding="utf-8"))
    assert index["meta"]["label"] == "Recorded run" and (tmp_path / "rec" / "c0.json").exists()


def test_served_index_announces_the_backend(client):
    r = client.get("/")
    assert r.status_code == 200 and 'name="sftforge-server"' in r.text
