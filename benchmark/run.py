"""CLI and reusable benchmark(model, samples, ...) API."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from benchmark.dataset import load_manifest, prepare_dataset, read_image
from benchmark.metrics import evaluate_image, scores


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def benchmark(model, samples, output_dir, warmup_iterations=5, iou_threshold=0.5, ignore_overlap=0.5):
    """Any adapter with predict(BGR image) and synchronize() can be passed here."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    rows = []
    try:
        image = read_image(samples[0]["image"])
        for _ in range(warmup_iterations):
            model.predict(image)
            model.synchronize()
        started = time.perf_counter()
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(samples):
                row = {**sample, "predictions": [], "error": None, "latency_ms": None}
                begin = time.perf_counter()
                try:
                    image = read_image(sample["image"])
                    row["decode_ms"] = (time.perf_counter() - begin) * 1000
                    model.synchronize()
                    infer_start = time.perf_counter()
                    row["predictions"] = model.predict(image)
                    model.synchronize()
                    row["latency_ms"] = (time.perf_counter() - infer_start) * 1000
                    row["image_to_prediction_ms"] = (time.perf_counter() - begin) * 1000
                    row["evaluation"] = evaluate_image(sample["annotations"], row["predictions"],
                                                        iou_threshold, ignore_overlap)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["evaluation"] = evaluate_image(sample["annotations"], [], iou_threshold, ignore_overlap)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if (index + 1) % 10 == 0 or index == 0:
                    print(f"{index + 1}/{len(samples)} errors={sum(r['error'] is not None for r in rows)}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        monitor.join()
    latency = [r["latency_ms"] for r in rows if not r["error"]]
    totals = {k: sum(r["evaluation"][k] for r in rows) for k in ("tp", "fp", "fn")}
    summary = {**scores(**totals), "images": len(rows), "errors": sum(r["error"] is not None for r in rows),
               "latency_p50_ms": float(np.percentile(latency, 50)) if latency else None,
               "latency_p95_ms": float(np.percentile(latency, 95)) if latency else None,
               "successful_inference_images_per_s": len(latency) / (sum(latency) / 1000) if latency else 0,
               "benchmark_wall_seconds": elapsed, "sampled_process_peak_rss_mb": peak[0] / 1024**2,
               "iou_threshold": iou_threshold, "ignore_overlap": ignore_overlap,
               "protocol": "local polygon IoU maximum-cardinality matching; not official VinText scorer",
               "timing_scope": "latency: BGR array -> polygons incl adapter, preprocess, inference, postprocess; excludes decode, scoring, plotting; batch=1; warmup excluded",
               "memory_scope": "process RSS sampled every 50 ms after model load; not VRAM or exact peak"}
    write_json(output / "summary.json", summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", default="data/raw/datasets/vintext/test")
    prepare.add_argument("--manifest", default="data/splits/vintext_test.jsonl")
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/benchmark/ppocrv6_small_det.json")
    run.add_argument("--model")
    run.add_argument("--output")
    run.add_argument("--limit", type=int)
    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_dataset(args.root, args.manifest), indent=2))
        return
    from benchmark.report import make_report
    if args.command == "report":
        make_report(Path(args.run_dir))
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        config["model_name"] = args.model
        if not args.output:
            parser.error("Use --output with --model to keep model results separate")
    if args.output:
        config["output_dir"] = args.output
    samples = load_manifest(config["manifest"])
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        samples = samples[:args.limit]
    if not samples:
        parser.error("Empty manifest")
    output = Path(config["output_dir"])
    if (output / "predictions.jsonl").exists():
        parser.error("Run directory already has predictions; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    # Claim the output before model loading, so simultaneous launches cannot overwrite it.
    try:
        with (output / "run.claim").open("x", encoding="utf-8") as claim:
            claim.write("This run directory has been claimed; use a new output directory for retries.\n")
    except FileExistsError:
        parser.error("Run directory is already claimed; choose a new --output")
    write_json(output / "config.json", {**config, "limit": args.limit})
    versions = {p: importlib.metadata.version(p) for p in ("paddleocr", "paddlex", "paddlepaddle", "numpy", "shapely", "networkx", "matplotlib")}
    env = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
           "logical_cpus": psutil.cpu_count(), "packages": versions,
           "manifest_sha256": hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest(),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "git_status": subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout}
    write_json(output / "environment.json", env)
    from models.detection import load_detector
    start = time.perf_counter()
    model = load_detector(config["model_name"], config["device"],
                          predict_options=config.get("predict_options"), **config.get("model_options", {}))
    load_seconds = time.perf_counter() - start
    import os
    model_dir = Path(config.get("model_options", {}).get("model_dir") or
                     Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "official_models" / config["model_name"])
    env["model_files_sha256"] = {
        p.relative_to(model_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in model_dir.rglob("*") if p.is_file()}
    write_json(output / "environment.json", env)
    _, summary = benchmark(model, samples, output, config["warmup_iterations"],
                           config["iou_threshold"], config["ignore_overlap"])
    summary["model_load_seconds_including_download"] = load_seconds
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    make_report(output)


if __name__ == "__main__":
    main()
