from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.core.config import settings


class VlmService:
    """Lightweight VLM boundary for first-order RC experiment explanations."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        openvino_model_dir: str | None = None,
        openvino_device: str | None = None,
        openvino_cache_dir: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self._provider = provider or settings.VLM_PROVIDER
        self._base_url = base_url or settings.VLM_BASE_URL
        self._model = model or settings.VLM_MODEL
        self._timeout_s = timeout_s or settings.VLM_TIMEOUT_S
        self._openvino_model_dir = openvino_model_dir or settings.VLM_OPENVINO_MODEL_DIR
        self._openvino_device = openvino_device or settings.VLM_OPENVINO_DEVICE
        self._openvino_cache_dir = openvino_cache_dir or settings.VLM_OPENVINO_CACHE_DIR
        self._max_new_tokens = max_new_tokens or settings.VLM_MAX_NEW_TOKENS
        self._openvino_pipeline: Any | None = None

    def explain_rc_pack(
        self,
        *,
        mrag_pack: dict[str, Any],
        user_query: str = "",
        current_image: str | None = None,
        reference_image: str | None = None,
    ) -> dict[str, Any]:
        prompt = self.build_prompt(
            mrag_pack=mrag_pack,
            user_query=user_query,
            has_current_image=bool(current_image),
            has_reference_image=bool(reference_image),
        )

        if self._provider == "openai_compatible" and self._base_url and self._model:
            return self._explain_with_openai_compatible(
                prompt=prompt,
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            )

        if self._provider == "openvino_genai" and self._openvino_model_dir:
            return self._explain_with_openvino_genai(
                prompt=prompt,
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            )

        return self._template_explanation(
            prompt=prompt,
            mrag_pack=mrag_pack,
            current_image=current_image,
            reference_image=reference_image,
        )

    def build_prompt(
        self,
        *,
        mrag_pack: dict[str, Any],
        user_query: str,
        has_current_image: bool,
        has_reference_image: bool,
    ) -> str:
        scene = mrag_pack.get("scene", {})
        fault_titles = [
            str(case.get("title", ""))
            for case in mrag_pack.get("fault_cases", [])
            if case.get("title")
        ]
        context = mrag_pack.get("structured_context", {})
        return "\n".join(
            [
                "你是一阶 RC 电路实验的离线板端 VLM 解释模块。",
                "不要重新识别元件、孔位或网表；这些事实已经由传统视觉和规则层给出。",
                f"实验场景：{scene.get('scene_name', '一阶 RC 电路实验')}",
                f"用户问题：{user_query or mrag_pack.get('query', '')}",
                f"错误标签：{', '.join(mrag_pack.get('error_tags', [])) or '无'}",
                f"规则错误码：{', '.join(context.get('error_codes', [])) or '无'}",
                f"候选错误知识：{'；'.join(fault_titles) or '无'}",
                f"是否有当前实拍图：{'是' if has_current_image else '否'}",
                f"是否有参考图/波形：{'是' if has_reference_image else '否'}",
                "请只输出：结论、依据、修改步骤。回答要短，优先使用结构化上下文和知识包。",
            ]
        )

    def _template_explanation(
        self,
        *,
        prompt: str,
        mrag_pack: dict[str, Any],
        current_image: str | None,
        reference_image: str | None,
    ) -> dict[str, Any]:
        fault_cases = mrag_pack.get("fault_cases", [])
        first_case = fault_cases[0] if fault_cases else {}
        title = first_case.get("title") or "一阶 RC 实验现象需要结合规则结果排查"
        reference_text = first_case.get("reference_text") or "当前没有命中的错误知识单元。"
        fix_steps = mrag_pack.get("fix_steps", [])[:4]

        return {
            "result_version": "vlm_explanation_v1",
            "provider": "template",
            "model": "",
            "status": "completed",
            "inputs": self._build_input_summary(
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            ),
            "prompt": prompt,
            "answer": {
                "conclusion": str(title),
                "evidence": str(reference_text),
                "fix_steps": fix_steps,
            },
            "raw_response": "",
        }

    def _explain_with_openai_compatible(
        self,
        *,
        prompt: str,
        mrag_pack: dict[str, Any],
        current_image: str | None,
        reference_image: str | None,
    ) -> dict[str, Any]:
        payload = self._build_openai_compatible_payload(
            prompt=prompt,
            current_image=current_image,
            reference_image=reference_image,
        )
        endpoint = f"{self._base_url.rstrip('/')}/chat/completions"
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            fallback = self._template_explanation(
                prompt=prompt,
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            )
            fallback["provider"] = "template_fallback"
            fallback["status"] = "vlm_call_failed"
            fallback["raw_response"] = str(exc)
            return fallback

        text = self._extract_openai_text(body)
        return {
            "result_version": "vlm_explanation_v1",
            "provider": "openai_compatible",
            "model": self._model or "",
            "status": "completed",
            "inputs": self._build_input_summary(
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            ),
            "prompt": prompt,
            "answer": {
                "conclusion": text,
                "evidence": "",
                "fix_steps": mrag_pack.get("fix_steps", [])[:4],
            },
            "raw_response": body,
        }

    def _explain_with_openvino_genai(
        self,
        *,
        prompt: str,
        mrag_pack: dict[str, Any],
        current_image: str | None,
        reference_image: str | None,
    ) -> dict[str, Any]:
        try:
            openvino_genai, _openvino = self._load_openvino_modules()
            pipe = self._get_openvino_pipeline(openvino_genai)
            generation_config = openvino_genai.GenerationConfig()
            generation_config.max_new_tokens = self._max_new_tokens
            images = self._read_openvino_images(
                current_image=current_image,
                reference_image=reference_image,
                openvino_module=_openvino,
            )
            if len(images) == 1:
                result = pipe.generate(prompt, image=images[0], generation_config=generation_config)
            elif images:
                result = pipe.generate(prompt, images=images, generation_config=generation_config)
            else:
                result = pipe.generate(prompt, generation_config=generation_config)
            text = self._extract_openvino_text(result)
        except Exception as exc:
            fallback = self._template_explanation(
                prompt=prompt,
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            )
            fallback["provider"] = "template_fallback"
            fallback["status"] = "openvino_call_failed"
            fallback["raw_response"] = str(exc)
            return fallback

        return {
            "result_version": "vlm_explanation_v1",
            "provider": "openvino_genai",
            "model": self._openvino_model_dir or "",
            "status": "completed",
            "inputs": self._build_input_summary(
                mrag_pack=mrag_pack,
                current_image=current_image,
                reference_image=reference_image,
            ),
            "prompt": prompt,
            "answer": {
                "conclusion": text,
                "evidence": "",
                "fix_steps": mrag_pack.get("fix_steps", [])[:4],
            },
            "raw_response": "",
        }

    def _load_openvino_modules(self) -> tuple[Any, Any]:
        import openvino
        import openvino_genai

        return openvino_genai, openvino

    def _get_openvino_pipeline(self, openvino_genai: Any) -> Any:
        if self._openvino_pipeline is not None:
            return self._openvino_pipeline
        model_dir = Path(str(self._openvino_model_dir or ""))
        if not model_dir.exists():
            raise FileNotFoundError(f"OpenVINO VLM model dir not found: {model_dir}")
        kwargs: dict[str, Any] = {}
        if self._openvino_cache_dir:
            kwargs["CACHE_DIR"] = str(self._openvino_cache_dir)
        self._openvino_pipeline = openvino_genai.VLMPipeline(
            str(model_dir),
            self._openvino_device,
            **kwargs,
        )
        return self._openvino_pipeline

    def _read_openvino_images(
        self,
        *,
        current_image: str | None,
        reference_image: str | None,
        openvino_module: Any,
    ) -> list[Any]:
        images: list[Any] = []
        for image in (current_image, reference_image):
            if not image:
                continue
            pil_image = self._read_pil_image(image)
            if pil_image is None:
                continue
            image_data = np.array(pil_image.convert("RGB"), dtype=np.uint8)
            image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1], 3)
            images.append(openvino_module.Tensor(image_data))
        return images

    def _read_pil_image(self, image: str) -> Image.Image | None:
        if image.startswith("data:"):
            try:
                _, encoded = image.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(encoded)))
            except Exception:
                return None
        if image.startswith(("http://", "https://")):
            return None
        path = Path(image)
        if not path.exists() or not path.is_file():
            return None
        return Image.open(path)

    def _extract_openvino_text(self, result: Any) -> str:
        texts = getattr(result, "texts", None)
        if isinstance(texts, list) and texts:
            return str(texts[0])
        text = getattr(result, "text", None)
        if text:
            return str(text)
        return str(result)

    def _build_openai_compatible_payload(
        self,
        *,
        prompt: str,
        current_image: str | None,
        reference_image: str | None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, image in (("当前实拍图", current_image), ("参考图或标准波形", reference_image)):
            if not image:
                continue
            image_url = self._to_image_url(image)
            if not image_url:
                continue
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": 512,
        }

    def _build_input_summary(
        self,
        *,
        mrag_pack: dict[str, Any],
        current_image: str | None,
        reference_image: str | None,
    ) -> dict[str, Any]:
        references = mrag_pack.get("references", {})
        return {
            "scene_id": mrag_pack.get("scene", {}).get("scene_id", ""),
            "error_tags": mrag_pack.get("error_tags", []),
            "fault_case_count": len(mrag_pack.get("fault_cases", [])),
            "has_current_image": bool(current_image),
            "has_reference_image": bool(reference_image),
            "reference_images": references.get("images", []),
            "reference_waveforms": references.get("waveforms", []),
            "reference_schematics": references.get("schematics", []),
        }

    def _to_image_url(self, image: str) -> str:
        if image.startswith(("data:", "http://", "https://")):
            return image
        path = Path(image)
        if not path.exists() or not path.is_file():
            return image
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_openai_text(self, body: dict[str, Any]) -> str:
        choices = body.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in parts if part)
        return str(content)
