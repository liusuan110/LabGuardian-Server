from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from breadboard_detect import (
    build_connections,
    build_holes,
    build_regions,
    detect_board_quad,
    draw_holes,
    draw_quad,
    draw_region_and_connection_overlay,
    fit_grid,
    order_points,
    score_image,
    warp_board,
    write_csv,
)


def parse_corners(raw: str) -> np.ndarray:
    pts: list[list[float]] = []
    for part in raw.strip().split(";"):
        if not part.strip():
            continue
        x_str, y_str = part.split(",")
        pts.append([float(x_str), float(y_str)])
    if len(pts) != 4:
        raise ValueError("Expected 4 points: x1,y1;x2,y2;x3,y3;x4,y4")
    return np.asarray(pts, dtype=np.float32)


def pick_corners(image: np.ndarray) -> np.ndarray:
    points: list[tuple[int, int]] = []
    window = "pick_4_corners"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = image.copy()
        for idx, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(idx + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "Click 4 corners. Enter: confirm  Backspace: undo  R: reset  Esc: cancel",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = int(cv2.waitKey(30) & 0xFF)
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise RuntimeError("Corner selection cancelled")
        if key in (8, 127):
            if points:
                points.pop()
            continue
        if key in (ord("r"), ord("R")):
            points.clear()
            continue
        if key in (10, 13) and len(points) == 4:
            break

    cv2.destroyAllWindows()
    return np.asarray(points, dtype=np.float32)


def process_image(
    *,
    image_path: Path,
    out_dir: Path,
    prefix: str,
    main_columns: int,
    corners: np.ndarray | None,
    interactive: bool,
    auto_fallback: bool,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    used_mode = "manual"
    quad: np.ndarray
    if corners is not None:
        quad = corners
    elif interactive:
        quad = pick_corners(image)
    else:
        if not auto_fallback:
            raise ValueError("Provide --corners or --interactive, or set --auto-fallback.")
        used_mode = "auto"
        quad = detect_board_quad(image)

    quad = order_points(quad)
    warped, _, inverse_matrix = warp_board(image, quad)
    model = fit_grid(warped, main_columns)
    score_map = score_image(warped)
    holes = build_holes(model, inverse_matrix, score_map, main_columns)
    regions = build_regions(model, inverse_matrix, (warped.shape[1], warped.shape[0]), main_columns)
    connections = build_connections(model, inverse_matrix, main_columns)

    paths = {
        "corners": out_dir / f"{prefix}_corners.png",
        "warped": out_dir / f"{prefix}_warped.png",
        "annotated_warped": out_dir / f"{prefix}_annotated_warped.png",
        "annotated_original": out_dir / f"{prefix}_annotated_original.png",
        "connectivity_warped": out_dir / f"{prefix}_connectivity_warped.png",
        "connectivity_original": out_dir / f"{prefix}_connectivity_original.png",
        "csv": out_dir / f"{prefix}_holes.csv",
        "json": out_dir / f"{prefix}_holes.json",
    }

    cv2.imwrite(str(paths["corners"]), draw_quad(image, quad))
    cv2.imwrite(str(paths["warped"]), warped)
    cv2.imwrite(str(paths["annotated_warped"]), draw_holes(warped, holes, "warp", draw_labels=False))
    cv2.imwrite(str(paths["annotated_original"]), draw_holes(image, holes, "image", draw_labels=False))
    cv2.imwrite(str(paths["connectivity_warped"]), draw_region_and_connection_overlay(warped, holes, regions, connections, "warp", draw_labels=False))
    cv2.imwrite(str(paths["connectivity_original"]), draw_region_and_connection_overlay(image, holes, regions, connections, "image", draw_labels=False))
    write_csv(paths["csv"], holes)

    metadata: dict[str, Any] = {
        "image": str(image_path),
        "hole_count": len(holes),
        "main_columns": main_columns,
        "mode": used_mode,
        "quad_tl_tr_br_bl": [[round(float(x), 3), round(float(y), 3)] for x, y in order_points(quad)],
        "warped_size": {"width": warped.shape[1], "height": warped.shape[0]},
        "grid": model.__dict__,
        "regions": [asdict(region) for region in regions],
        "connections": [asdict(connection) for connection in connections],
        "holes": holes,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and annotate breadboard holes using manually selected corners.")
    parser.add_argument("--image", default="bread.png", type=Path, help="Input breadboard image.")
    parser.add_argument("--out-dir", default=Path("outputs"), type=Path, help="Directory for generated files.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to input stem.")
    parser.add_argument("--main-columns", default=63, type=int, help="Terminal-strip columns. Standard 830-point boards use 63.")
    parser.add_argument("--corners", default=None, help="Manual corners: x1,y1;x2,y2;x3,y3;x4,y4")
    parser.add_argument("--interactive", action="store_true", help="Pick corners by clicking on the image.")
    parser.add_argument("--auto-fallback", action="store_true", help="If no manual corners, fall back to auto board detection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prefix or args.image.stem
    corners = parse_corners(args.corners) if args.corners else None
    result = process_image(
        image_path=args.image,
        out_dir=args.out_dir,
        prefix=prefix,
        main_columns=args.main_columns,
        corners=corners,
        interactive=bool(args.interactive),
        auto_fallback=bool(args.auto_fallback),
    )
    print(f"Detected {result['hole_count']} holes.")
    print(f"mode: {result['mode']}")
    print(f"Board corners TL/TR/BR/BL: {result['quad_tl_tr_br_bl']}")
    for name, path in (result.get("paths") or {}).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

