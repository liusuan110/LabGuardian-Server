from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.reference_service import ReferenceService

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "references"


class TestReferenceService:
    def test_list_references(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        refs = svc.list_references()
        assert len(refs) >= 1
        ref = next(r for r in refs if r["reference_id"] == "test_rc_v1")
        assert ref["name"] == "Test RC"
        assert ref["format"] == "logical_reference_v1"
        assert ref["component_count"] == 2
        assert ref["net_count"] == 3

    def test_load_reference(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        payload = svc.load_reference("test_rc_v1")
        assert payload["reference_id"] == "test_rc_v1"
        assert payload["format"] == "logical_reference_v1"
        assert len(payload["components"]) == 2
        assert len(payload["nets"]) == 3

    def test_load_reference_not_found(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        with pytest.raises(FileNotFoundError):
            svc.load_reference("non_existent")

    def test_load_reference_invalid_id(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        with pytest.raises(ValueError, match="非法 reference_id"):
            svc.load_reference("../secret")
        with pytest.raises(ValueError, match="非法 reference_id"):
            svc.load_reference("a/b")
        with pytest.raises(ValueError, match="非法 reference_id"):
            svc.load_reference("a\\b")
        with pytest.raises(ValueError, match="非法 reference_id"):
            svc.load_reference("with space")

    def test_load_reference_mismatched_id(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        # Create a temp file with mismatched reference_id
        tmp_path = FIXTURE_DIR / "mismatch_v1.json"
        tmp_path.write_text(
            json.dumps(
                {
                    "format": "logical_reference_v1",
                    "reference_id": "wrong_id",
                    "components": [{"ref_id": "R1", "type": "Resistor", "pins": [{"pin": "p1", "net": "N1"}]}],
                    "nets": [{"net": "N1"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            with pytest.raises(ValueError, match="reference_id 不匹配"):
                svc.load_reference("mismatch_v1")
        finally:
            tmp_path.unlink()

    def test_validate_reference_ok(self) -> None:
        svc = ReferenceService()
        svc.validate_reference(
            {
                "format": "logical_reference_v1",
                "reference_id": "ok",
                "components": [
                    {
                        "ref_id": "R1",
                        "type": "Resistor",
                        "pins": [{"pin": "p1", "net": "N1"}],
                    }
                ],
                "nets": [{"net": "N1"}],
            }
        )

    def test_validate_reference_missing_format(self) -> None:
        svc = ReferenceService()
        with pytest.raises(ValueError, match="format 必须是"):
            svc.validate_reference({"components": []})

    def test_validate_reference_empty_components(self) -> None:
        svc = ReferenceService()
        with pytest.raises(ValueError, match="components 必须是非空数组"):
            svc.validate_reference(
                {"format": "logical_reference_v1", "components": []}
            )

    def test_validate_reference_unknown_net(self) -> None:
        svc = ReferenceService()
        with pytest.raises(ValueError, match="引用了未定义的 net"):
            svc.validate_reference(
                {
                    "format": "logical_reference_v1",
                    "components": [
                        {
                            "ref_id": "R1",
                            "type": "Resistor",
                            "pins": [{"pin": "p1", "net": "UNKNOWN"}],
                        }
                    ],
                    "nets": [{"net": "N1"}],
                }
            )
