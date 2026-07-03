from __future__ import annotations

import pytest

from app.pipeline.vision import camera_locator
from app.pipeline.vision.camera_locator import VideoDeviceInfo


def test_resolve_video_device_index_by_usb_hint_prefers_capture_index_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        camera_locator,
        "list_video_devices",
        lambda **_: [
            VideoDeviceInfo(
                dev_node="/dev/video5",
                device_index=5,
                sysfs_name="video5",
                usb_device_path="/sys/devices/pci0000:00/.../1-6/video4linux/video5",
                capture_index=1,
            ),
            VideoDeviceInfo(
                dev_node="/dev/video4",
                device_index=4,
                sysfs_name="video4",
                usb_device_path="/sys/devices/pci0000:00/.../1-6/video4linux/video4",
                capture_index=0,
            ),
        ],
    )

    resolved = camera_locator.resolve_video_device_index_by_usb_hint(
        "usb-0000:00:14.0-6"
    )

    assert resolved == 4


def test_resolve_video_device_index_by_usb_hint_raises_on_missing_match(monkeypatch) -> None:
    monkeypatch.setattr(
        camera_locator,
        "list_video_devices",
        lambda **_: [
            VideoDeviceInfo(
                dev_node="/dev/video2",
                device_index=2,
                sysfs_name="video2",
                usb_device_path="/sys/devices/pci0000:00/.../1-3/video4linux/video2",
                capture_index=0,
            ),
        ],
    )

    with pytest.raises(ValueError, match="no /dev/video\\* matches usb hint"):
        camera_locator.resolve_video_device_index_by_usb_hint(
            "usb-0000:00:14.0-7"
        )
