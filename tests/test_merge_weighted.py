from pathlib import Path
import json

import sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_root))
from ebpe.ensemble import build_merge_weighted_bpe_json


def _write_tok(p: Path, merges: list[tuple[str, str]], vocab: dict[str, int]):
    model = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [list(m) for m in merges],
        },
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model), encoding="utf-8")


def test_weighted_theta_selects_major_weight(tmp_path: Path):
    # Member 0 strongly weighted, proposes merge (a,b);
    # Member 1 low weight, does not include it. With theta=0.6, select.
    t0 = tmp_path / "t0.json"
    t1 = tmp_path / "t1.json"
    _write_tok(t0, merges=[("a", "b")], vocab={"a": 0, "b": 1, "ab": 2})
    _write_tok(t1, merges=[], vocab={"a": 0, "b": 1})

    merged = build_merge_weighted_bpe_json([t0, t1], weights=[10.0, 1.0], theta=0.6)
    merges = merged["model"]["merges"]
    assert ["a", "b"] in merges


def test_weighted_theta_filters_under_threshold(tmp_path: Path):
    # Member weights equal; theta=0.75 filters merges seen in only 1 member
    t0 = tmp_path / "t0.json"
    t1 = tmp_path / "t1.json"
    _write_tok(t0, merges=[("x", "y")], vocab={"x": 0, "y": 1, "xy": 2})
    _write_tok(t1, merges=[], vocab={"x": 0, "y": 1})

    merged = build_merge_weighted_bpe_json([t0, t1], weights=[1.0, 1.0], theta=0.75)
    merges = merged["model"]["merges"]
    assert ["x", "y"] not in merges
