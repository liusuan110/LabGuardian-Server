from app.services.mrag_service import MragService
from app.services.teaching_kb_service import TeachingKbService
from app.services.vlm_service import VlmService


def _sample_pack():
    return MragService(teaching_kb_service=TeachingKbService()).build_pack(
        query="示波器 X10 档为什么读数要乘以 10",
        error_tags=["probe_mode_error"],
        structured_context={"error_codes": [], "risk_level": "safe"},
    )


def test_vlm_template_explains_rc_pack_without_model():
    service = VlmService(provider="template")

    result = service.explain_rc_pack(mrag_pack=_sample_pack())

    assert result["result_version"] == "vlm_explanation_v1"
    assert result["provider"] == "template"
    assert result["status"] == "completed"
    assert "X1/X10" in result["answer"]["conclusion"]
    assert result["answer"]["fix_steps"]


def test_vlm_prompt_keeps_recognition_out_of_vlm_scope():
    service = VlmService(provider="template")

    prompt = service.build_prompt(
        mrag_pack=_sample_pack(),
        user_query="为什么波形不对",
        has_current_image=True,
        has_reference_image=True,
    )

    assert "不要重新识别元件、孔位或网表" in prompt
    assert "一阶 RC" in prompt
    assert "当前实拍图：是" in prompt
    assert "参考图/波形：是" in prompt


def test_vlm_openai_compatible_failure_falls_back_to_template():
    service = VlmService(
        provider="openai_compatible",
        base_url="http://127.0.0.1:9/v1",
        model="local-vlm",
        timeout_s=0.01,
    )

    result = service.explain_rc_pack(mrag_pack=_sample_pack())

    assert result["provider"] == "template_fallback"
    assert result["status"] == "vlm_call_failed"
    assert result["answer"]["fix_steps"]


def test_vlm_openvino_missing_model_falls_back_to_template():
    service = VlmService(
        provider="openvino_genai",
        openvino_model_dir="/tmp/labguardian-missing-openvino-vlm",
    )

    result = service.explain_rc_pack(mrag_pack=_sample_pack())

    assert result["provider"] == "template_fallback"
    assert result["status"] == "openvino_call_failed"
    assert result["answer"]["fix_steps"]
