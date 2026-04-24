from __future__ import annotations

import numpy as np

from app.pipeline.vision.roi_cropper import _expanded_bounds_from_bbox, crop_component_roi


class TestRoiCropper:
    def test_asymmetric_major_padding_is_supported(self):
        bounds = _expanded_bounds_from_bbox(
            bbox=(100, 100, 140, 120),
            image_shape=(300, 300),
            orientation=0.0,
            profile={
                "profile_name": "test_asymmetric",
                "major_pad_ratio": 0.0,
                "minor_pad_ratio": 0.0,
                "major_pad_before_ratio": 1.00,
                "major_pad_after_ratio": 0.25,
                "minor_pad_before_ratio": 0.10,
                "minor_pad_after_ratio": 0.10,
                "min_major_pad_px": 0,
                "min_minor_pad_px": 0,
                "min_major_span_px": 0,
                "min_minor_span_px": 0,
                "min_roi_w": 1,
                "min_roi_h": 1,
            },
        )

        assert bounds == (60, 98, 150, 122)

    def test_ceramic_cap_top_roi_expands_along_major_axis(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        bbox = (100, 120, 140, 160)

        roi, offset, meta = crop_component_roi(
            image,
            bbox,
            component_type="CapacitorCeramic",
            package_type="capacitor_ceramic_2pin",
            orientation=0.0,
            view_id="top",
        )

        assert roi is not None
        assert offset[0] <= bbox[0]
        assert offset[1] <= bbox[1]
        assert meta["profile_name"] == "ceramic_cap_body_with_extended_leads"
        assert meta["roi_size"][0] >= 120
        assert meta["roi_size"][1] >= 72
        assert meta["roi_size"][0] > meta["roi_size"][1]
