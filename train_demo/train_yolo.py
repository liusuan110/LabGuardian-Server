import argparse
from pathlib import Path


def _resolve(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _train_detect(weights: str, data: str, epochs: int, imgsz: int, batch: int, device: str | None, project: str, name: str) -> None:
    from ultralytics import YOLO

    model = YOLO(_resolve(weights))
    model.train(
        task="detect",
        data=_resolve(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )


def _train_pose(pose_weights: str | None, weights_detect_backbone: str, data: str, epochs: int, imgsz: int, batch: int, device: str | None, project: str, name: str) -> None:
    from ultralytics import YOLO

    if pose_weights and Path(pose_weights).exists():
        model = YOLO(_resolve(pose_weights))
    else:
        pose_pretrained = Path(weights_detect_backbone).with_name("yolov8s-pose.pt")
        if pose_pretrained.exists():
            model = YOLO(_resolve(str(pose_pretrained)))
        else:
            model = YOLO("yolov8s-pose.yaml")
            model.load(_resolve(weights_detect_backbone))

    model.train(
        task="pose",
        data=_resolve(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detect", "pose", "both"], default="both")
    parser.add_argument("--weights", default=r"d:\train_demo\yolov8s.pt")
    parser.add_argument("--pose_weights", default=r"d:\train_demo\yolov8s-pose.pt")
    parser.add_argument("--data_detect", default=r"d:\train_demo\yolo_packed2_det\data.yaml")
    parser.add_argument("--data_pose", default=r"d:\train_demo\yolo_packed2_pose\data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name_detect", default="detect_components")
    parser.add_argument("--name_pose", default="pose_components")

    args = parser.parse_args()

    if args.task in ("detect", "both"):
        _train_detect(
            weights=args.weights,
            data=args.data_detect,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name_detect,
        )

    if args.task in ("pose", "both"):
        _train_pose(
            pose_weights=args.pose_weights,
            weights_detect_backbone=args.weights,
            data=args.data_pose,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name_pose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
