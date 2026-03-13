import sys; sys.path.insert(0, '.')
import cv2, numpy as np
from app.pipeline.vision.calibrator import BreadboardCalibrator

img_path = r'D:\desktop\inter\LabGuardian\dataset\images\demo_self\valid\images\-_20260307102606_394_202_jpg.rf.50f22610ff8c8d5baf0027f46fa1eb98.jpg'
img = cv2.imread(img_path)
print(f"Image shape: {img.shape}")

cal = BreadboardCalibrator()
ok = cal.ensure_calibrated(img)
print(f"Calibrated: {ok}, landscape: {cal._landscape}")
print(f"Rows: {cal.rows}")
if cal._row_coords is not None:
    print(f"Row range: {cal._row_coords[0]:.1f} ~ {cal._row_coords[-1]:.1f}")
if cal._col_coords is not None:
    print(f"Col coords (main grid): {[f'{c:.0f}' for c in cal._col_coords]}")
print(f"Top rails: {cal._top_rails}")
print(f"Bot rails: {cal._bot_rails}")

pins = [
    ('Wire-top pin1', 27, 72),
    ('Wire-top pin2', 276, 117),
    ('Resistor pin1', 218, 189),
    ('Resistor pin2', 273, 210),
    ('LED pin1', 310, 241),
    ('LED pin2', 310, 292),
    ('Wire-bot pin1', 469, 318),
    ('Wire-bot pin2', 214, 471),
]
print()
for name, px, py in pins:
    logic = cal.frame_pixel_to_logic(px, py)
    print(f"  {name} ({px},{py}) -> {logic}")
