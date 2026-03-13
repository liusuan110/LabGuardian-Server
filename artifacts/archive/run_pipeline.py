"""Wrapper to run test_ua741_pipeline.py and capture all output to a file."""
import sys, os, traceback

os.environ["YOLO_DEVICE"] = "cpu"

ROOT = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(ROOT, "pipeline_log.txt")

# Redirect stdout+stderr to file
log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    stream=log_file, force=True)

try:
    from test_ua741_pipeline import run_full_pipeline
    img_path = os.path.join(ROOT, "detect_results", "scene5.jpg")
    out_dir = os.path.join(ROOT, "detect_results", "ua741_pipeline")
    run_full_pipeline(img_path, out_dir)
except Exception:
    traceback.print_exc(file=log_file)
finally:
    log_file.flush()
    log_file.close()
    # Restore stdout to print final message
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Done. Log: {log_path}")
