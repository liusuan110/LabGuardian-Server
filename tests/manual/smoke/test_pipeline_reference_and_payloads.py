"""
Pipeline reference / payload smoke test.

验证两件事:
1. `reference_circuit` 以内联 dict 传入时, S4 可以正确加载参考电路
2. 异步 worker 成功态返回的标准化 payload 可以被再次反序列化为 PipelineResult
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    # 这个 smoke 只需要最小 BaseModel/Field 能力, 用轻量 shim 让本地无依赖环境也能验证
    shim = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    value = getattr(self.__class__, name, None)
                setattr(self, name, value)

        def model_dump(self):
            return dict(self.__dict__)

    def _Field(default=None, **kwargs):
        return default

    shim.BaseModel = _BaseModel
    shim.Field = _Field
    sys.modules["pydantic"] = shim

from app.pipeline.stages.s3_topology import run_topology
from app.pipeline.stages.s4_validate import run_validate
from app.schemas.pipeline import PipelineResult


def main() -> int:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "netlist_v2"
    reference_path = fixture_dir / "reference_simple_v4.json"
    mapped_path = fixture_dir / "mapped_components_simple.json"

    with open(reference_path, "r", encoding="utf-8") as f:
        reference_payload = json.load(f)
    with open(mapped_path, "r", encoding="utf-8") as f:
        mapped_components = json.load(f)

    s3 = run_topology(mapped_components)
    s4 = run_validate(
        s3["topology_graph"],
        reference_circuit=reference_payload,
        components=mapped_components,
    )

    if not s4["is_correct"]:
        print("inline_reference_validate_failed")
        print(s4["comparison_report"])
        raise SystemExit(1)

    raw = {
        "stages": {
            "topology": {
                "duration_ms": 12.0,
                "component_count": s3["component_count"],
                "netlist_v2": s3["netlist_v2"],
            },
            "validate": {
                "duration_ms": 8.0,
                "progress": s4["progress"],
                "similarity": s4["similarity"],
                "diagnostics": s4["diagnostics"],
                "comparison_report": s4["comparison_report"],
                "risk_level": s4["risk_level"],
                "risk_reasons": s4["risk_reasons"],
            },
        },
        "total_duration_ms": 20.0,
    }
    normalized = PipelineResult.from_pipeline_run(
        job_id="job-inline",
        station_id="S01",
        raw=raw,
    )
    roundtrip = PipelineResult.from_pipeline_run(
        job_id="job-inline",
        station_id="S01",
        raw=normalized.model_dump(),
    )

    print(f"inline_reference_similarity={s4['similarity']:.2f}")
    print(f"normalized_station_id={normalized.station_id}")
    print(f"roundtrip_status={roundtrip.status}")
    print(f"roundtrip_stage_count={len(roundtrip.stages)}")

    if roundtrip.station_id != "S01" or len(roundtrip.stages) != 2:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
