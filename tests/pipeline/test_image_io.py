"""
T1: 图像解码测试 — 无模型依赖

验证 image_io.decode_images_b64() 和 decode_summary() 的:
- 正常解码
- 错误处理和日志
- 多视图标记
- 空输入处理
"""

from __future__ import annotations

import logging
import pytest

from tests.pipeline.fixtures import (
    make_blank_image,
    make_breadboard_image,
    image_to_b64,
    make_corrupted_b64,
)


class TestDecodeImagesB64:
    """T1: 图像解码测试."""

    def test_t1_1_single_valid_image(self):
        """T1.1: 单张有效图像 → decoded_view_count=1"""
        from app.pipeline.vision.image_io import decode_images_b64, decode_summary

        b64 = image_to_b64(make_blank_image())
        result = decode_images_b64([b64], logger=logging.getLogger(), stage_name="test")

        assert len(result) == 1
        assert result[0]["decoded"] is True
        assert result[0]["image"] is not None
        assert result[0]["view_id"] == "top"
        assert result[0]["error"] is None

        summary = decode_summary(result)
        assert summary["decoded_view_count"] == 1
        assert summary["available_view_ids"] == ["top"]
        assert summary["dropped_view_ids"] == []
        assert summary["decode_errors"] == {}

    def test_t1_2_three_valid_images(self):
        """T1.2: 3 张有效图像 → decoded_view_count=3"""
        from app.pipeline.vision.image_io import decode_images_b64, decode_summary

        images = [image_to_b64(make_blank_image()) for _ in range(3)]
        result = decode_images_b64(images, logger=logging.getLogger(), stage_name="test")

        assert len(result) == 3
        for item in result:
            assert item["decoded"] is True
            assert item["image"] is not None

        summary = decode_summary(result)
        assert summary["decoded_view_count"] == 3
        assert summary["available_view_ids"] == ["top", "left_front", "right_front"]

    def test_t1_3_empty_list(self):
        """T1.3: 空列表 → decoded_view_count=0"""
        from app.pipeline.vision.image_io import decode_images_b64, decode_summary

        result = decode_images_b64([], logger=logging.getLogger(), stage_name="test")
        assert len(result) == 0

        summary = decode_summary(result)
        assert summary["decoded_view_count"] == 0
        assert summary["available_view_ids"] == []

    def test_t1_4_one_corrupted(self):
        """T1.4: 3 张中 1 张损坏 → decoded_view_count=2, dropped_view_ids 记录损坏视图"""
        from app.pipeline.vision.image_io import decode_images_b64, decode_summary

        valid = image_to_b64(make_blank_image())
        corrupted = make_corrupted_b64()
        images = [valid, corrupted, valid]

        result = decode_images_b64(images, logger=logging.getLogger(), stage_name="test")

        assert len(result) == 3
        # view_ids 按顺序: top, left_front, right_front
        assert result[0]["decoded"] is True
        assert result[1]["decoded"] is False
        assert result[1]["view_id"] == "left_front"
        assert result[1]["error"] is not None
        assert result[2]["decoded"] is True

        summary = decode_summary(result)
        assert summary["decoded_view_count"] == 2
        assert summary["dropped_view_ids"] == ["left_front"]
        assert "left_front" in summary["decode_errors"]

    def test_t1_5_all_corrupted(self):
        """T1.5: 全部损坏 → decoded_view_count=0, decode_errors 记录所有"""
        from app.pipeline.vision.image_io import decode_images_b64, decode_summary

        images = [make_corrupted_b64() for _ in range(3)]
        result = decode_images_b64(images, logger=logging.getLogger(), stage_name="test")

        assert len(result) == 3
        for item in result:
            assert item["decoded"] is False
            assert item["image"] is None

        summary = decode_summary(result)
        assert summary["decoded_view_count"] == 0
        assert len(summary["dropped_view_ids"]) == 3
        assert len(summary["decode_errors"]) == 3

    def test_decode_view_id_assignment(self):
        """验证 view_id 按顺序正确分配: top, left_front, right_front, aux_view_1..."""
        from app.pipeline.vision.image_io import decode_images_b64

        images = [image_to_b64(make_blank_image()) for _ in range(5)]
        result = decode_images_b64(images, logger=logging.getLogger(), stage_name="test")

        assert result[0]["view_id"] == "top"
        assert result[1]["view_id"] == "left_front"
        assert result[2]["view_id"] == "right_front"
        assert result[3]["view_id"] == "aux_view_1"
        assert result[4]["view_id"] == "aux_view_2"

    def test_decode_preserves_image_data(self):
        """验证解码后的图像数据完整性."""
        from app.pipeline.vision.image_io import decode_images_b64
        import cv2

        original = make_blank_image(h=240, w=320, color=(100, 150, 200))
        b64 = image_to_b64(original)
        result = decode_images_b64([b64], logger=logging.getLogger(), stage_name="test")

        decoded = result[0]["image"]
        assert decoded is not None
        assert decoded.shape == original.shape
        assert cv2.imdecode is not None  # 确保 cv2 可用

    def test_decode_invalid_base64_characters(self):
        """测试无效 base64 字符."""
        from app.pipeline.vision.image_io import decode_images_b64

        result = decode_images_b64(["!!!not-base64!!!"], logger=logging.getLogger(), stage_name="test")
        assert len(result) == 1
        assert result[0]["decoded"] is False
        assert result[0]["view_id"] == "top"
        assert result[0]["error"] is not None
