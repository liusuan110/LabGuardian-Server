"""Run YOLO component detection on user breadboard images."""
import os, sys, time, cv2
from ultralytics import YOLO

MODEL = r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\models\unit_v1_yolov8s_960\weights\best.pt"
OUT = r"D:\desktop\LabGuardian-Server-main\LabGuardian-Server-main\detect_results"
COLORS = {
    "Breadboard":(200,200,200), "IC":(255,0,0), "Line_area":(0,165,255),
    "capacitor":(0,255,0), "diode":(0,255,255), "led":(0,0,255),
    "pinned":(255,0,255), "potentiometer":(255,255,0), "resistor":(128,0,128),
}

os.makedirs(OUT, exist_ok=True)
model = YOLO(MODEL)
print(f"Classes: {model.names}")

for idx, p in enumerate(sys.argv[1:], 1):
    img = cv2.imread(p)
    if img is None:
        print(f"[SKIP] {p}")
        continue
    h, w = img.shape[:2]
    print(f"\n{'='*50}\nImage {idx}: {os.path.basename(p)} ({w}x{h})\n{'='*50}")
    t0 = time.time()
    results = model(img, conf=0.25, iou=0.5, imgsz=960, device="cpu", verbose=False)
    ms = (time.time()-t0)*1000
    print(f"  Inference: {ms:.0f}ms")
    boxes = results[0].boxes
    vis = img.copy()
    cc = {}
    for i in range(len(boxes)):
        cn = model.names[int(boxes.cls[i].item())]
        cf = float(boxes.conf[i].item())
        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        cc[cn] = cc.get(cn,0)+1
        color = COLORS.get(cn,(255,255,255))
        if cn == "pinned":
            cx,cy = (x1+x2)//2,(y1+y2)//2
            r = max(3, min(x2-x1,y2-y1)//3)
            cv2.circle(vis,(cx,cy),r,color,2)
            cv2.circle(vis,(cx,cy),2,(0,0,255),-1)
        elif cn in ("Breadboard","Line_area"):
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,1)
            cv2.putText(vis,f"{cn} {cf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        else:
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            label = f"{cn} {cf:.2f}"
            ls = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)[0]
            cv2.rectangle(vis,(x1,y1-ls[1]-10),(x1+ls[0],y1),color,-1)
            cv2.putText(vis,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    print(f"  Total: {sum(cc.values())}")
    for k,v in sorted(cc.items()):
        print(f"    {k:15s}: {v}")
    out = os.path.join(OUT, f"detect_{idx}.jpg")
    cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {out}")

print(f"\nDone! Results in: {OUT}")
