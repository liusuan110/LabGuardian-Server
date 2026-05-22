from __future__ import annotations

from pathlib import Path

from app.agent.context_pack import build_context_pack
from app.agent.contracts import RuntimeEvidence
from app.agent.tools import CircuitLookupInput, circuit_lookup_tool
from app.core.config import PROJECT_ROOT
from app.services.circuit_kb_service import CircuitKbService, looks_like_circuit_query


def _top_circuit_id(query: str) -> str:
    hits = CircuitKbService().search(query=query, top_k=3)
    assert hits, f"expected circuit KB hit for query: {query}"
    return hits[0]["circuit"]["circuit_id"]


def test_paraphrased_waveform_query_hits_integrator() -> None:
    assert looks_like_circuit_query("输入方波以后为什么输出慢慢爬坡成三角波")
    assert _top_circuit_id("输入方波以后为什么输出慢慢爬坡成三角波") == "integrator_circuit"


def test_noise_chatter_query_hits_hysteresis_comparator() -> None:
    assert _top_circuit_id("输出在门限附近来回跳，怎么加两个门限防抖") == "comparator_hysteresis"


def test_subtract_common_mode_query_hits_bjt_differential_pair() -> None:
    assert _top_circuit_id("两个输入信号相减，还要抑制共模噪声") == "differential_amplifier"


def test_8050_audio_query_hits_common_emitter() -> None:
    assert _top_circuit_id("8050 做音频小信号放大，为什么从集电极取输出") == "common_emitter_amplifier"


def test_circuit_lookup_tool_searches_before_relevance_gate() -> None:
    result = circuit_lookup_tool(
        CircuitLookupInput(query="方波进去以后怎么变成三角波", top_k=3)
    )
    assert result.status == "ok"
    assert result.payload["circuits"][0]["circuit_id"] == "integrator_circuit"
    assert result.payload["circuits"][0]["matched_features"]


def test_context_pack_allows_circuit_tool_for_paraphrased_query() -> None:
    pack = build_context_pack(
        RuntimeEvidence(station_id="station-1"),
        query="比较器输出在阈值附近一直抖动怎么办",
    )
    assert any(tool.name == "circuit_lookup_tool" for tool in pack.allowed_tools)


def test_circuit_json_images_exist_and_have_annotations() -> None:
    service = CircuitKbService()
    circuits = service.list_circuits()
    assert circuits
    for circuit in circuits:
        image = circuit.get("image")
        assert image, circuit.get("circuit_id")
        assert (Path(PROJECT_ROOT) / image).exists(), image
        annotations = circuit.get("image_annotations") or {}
        assert annotations.get("visible_components"), circuit.get("circuit_id")


def test_common_emitter_json_matches_visible_8050_schematic() -> None:
    circuit = CircuitKbService().get_circuit("common_emitter_amplifier")
    assert circuit is not None
    refs = {component["ref"] for component in circuit["components"]}
    assert {"RP", "R", "RC", "CB", "CC", "RL", "VT"} <= refs
    assert "RE" not in refs
    assert "8050" in circuit["summary"]
