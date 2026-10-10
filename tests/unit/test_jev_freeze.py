"""freeze-qset writes hash + ledger row without creating a git tag."""

from __future__ import annotations

from pathlib import Path

from jevlab.jev.audit import freeze_qset
from jevlab.jev.qset import load_qset
from jevlab.utils.paths import get_project_root


def test_freeze_qset_updates_config_and_ledger(tmp_path: Path) -> None:
    root = get_project_root()
    qset_src = root / "config" / "qset_v1.json"
    qset_copy = tmp_path / "qset_v1.json"
    qset_copy.write_text(qset_src.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = tmp_path / "jev.yaml"
    cfg.write_text(
        "route: systemone\nmodel_id: jev-1.13.0\nqset_hash: null\n",
        encoding="utf-8",
    )
    registry = tmp_path / "experiments.sqlite"
    result = freeze_qset(
        qset_path=qset_copy,
        config_path=cfg,
        registry_path=registry,
    )
    assert result["qset_hash"]
    assert "qset-v1-frozen" in result["git_tag_command"]
    assert "NOT created automatically" in result["note"]
    text = cfg.read_text(encoding="utf-8")
    assert result["qset_hash"] in text
    # Loader accepts frozen hash
    loaded = load_qset(qset_copy, expected_hash=result["qset_hash"])
    assert loaded.content_hash == result["qset_hash"]
