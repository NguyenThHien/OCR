"""Regenerate charts and error analysis from saved predictions, without model inference."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from benchmark.dataset import load_manifest, read_image


def make_report(run_dir):
    run_dir = Path(run_dir)
    rows = load_manifest(run_dir / "predictions.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    charts = run_dir / "charts"
    charts.mkdir(exist_ok=True)
    groups = defaultdict(lambda: [0, 0])
    merged = 0
    near = 0
    missed_count = 0
    covered_gt = 0
    total_gt = 0
    for row in rows:
        matched = {m[0] for m in row["evaluation"]["matches"]}
        valid = [a for a in row["annotations"] if not a["ignore"]]
        total_gt += len(valid)
        for i, a in enumerate(row["annotations"]):
            if a["ignore"]:
                continue
            p = np.asarray(a["polygon"])
            short = min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1]))
            label = "small (<16 px)" if short < 16 else "medium (16-32 px)" if short < 32 else "large (>=32 px)"
            groups[label][0] += int(i in matched)
            groups[label][1] += 1
        if not row["error"]:
            covered_indices = set()
            for pred in row["predictions"]:
                p = Polygon(pred["polygon"])
                covered = [i for i, a in enumerate(valid)
                           if p.intersection(Polygon(a["polygon"])).area / Polygon(a["polygon"]).area > 0.5]
                covered_indices.update(covered)
                merged += int(len(covered) >= 2)
            covered_gt += len(covered_indices)
            near += sum(0.1 <= iou < summary["iou_threshold"] for iou in row["evaluation"]["best_ious"])
        missed_count += row["evaluation"]["fn"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(["Precision", "Recall", "H-mean"], [summary[k] for k in ("precision", "recall", "hmean")])
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Polygon detection @ IoU >= {summary['iou_threshold']}")
    latency = [r["latency_ms"] for r in rows if not r["error"]]
    axes[1].hist(latency, bins=30)
    axes[1].set_xlabel("Image-to-polygons latency (ms)")
    axes[1].set_title("Latency (warm-up excluded)")
    labels = [k for k in ("small (<16 px)", "medium (16-32 px)", "large (>=32 px)") if k in groups]
    axes[2].bar(labels, [groups[k][0] / groups[k][1] for k in labels])
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].set_title("Recall by original GT short edge")
    fig.tight_layout()
    fig.savefig(charts / "overview.png", dpi=160)
    fig.savefig(charts / "overview.svg")
    plt.close(fig)
    with (run_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "tp", "fp", "fn", "precision", "recall", "hmean", "latency_ms", "error"])
        for row in rows:
            writer.writerow([row["id"], *[row["evaluation"][k] for k in ("tp", "fp", "fn", "precision", "recall", "hmean")], row["latency_ms"], row["error"]])
    overlay_dir = run_dir / "error_examples"
    overlay_dir.mkdir(exist_ok=True)
    worst = sorted(rows, key=lambda r: (r["evaluation"]["hmean"], -(r["evaluation"]["fn"] + r["evaluation"]["fp"])))[:12]
    for row in worst:
        image = read_image(row["image"])
        matched = {m[0] for m in row["evaluation"]["matches"]}
        for i, a in enumerate(row["annotations"]):
            color = (128, 128, 128) if a["ignore"] else (0, 200, 0) if i in matched else (0, 0, 255)
            cv2.polylines(image, [np.asarray(a["polygon"], dtype=np.int32)], True, color, 2)
        for a in row["predictions"]:
            cv2.polylines(image, [np.asarray(a["polygon"], dtype=np.int32)], True, (255, 180, 0), 1)
        cv2.imwrite(str(overlay_dir / f"{row['id']}.jpg"), image)
    lines = ["# Benchmark VinText: kết quả thực tế", "",
             f"Số ảnh: {summary['images']}; ảnh lỗi: {summary['errors']}.", "",
             "| Precision | Recall | H-mean | p50 ms | p95 ms |",
             "| --- | --- | --- | --- | --- |",
             f"| {summary['precision']:.4f} | {summary['recall']:.4f} | {summary['hmean']:.4f} | {summary['latency_p50_ms']} | {summary['latency_p95_ms']} |", "",
             "![Biểu đồ](charts/overview.png)", "", "## Phân tích lỗi", "",
             "| Nhóm kích thước GT trên ảnh gốc | Khớp / tổng | Recall |", "| --- | --- | --- |"]
    for label in labels:
        hit, total = groups[label]
        lines.append(f"| {label} | {hit}/{total} | {hit / total:.4f} |")
    lines += ["", f"Có {merged} prediction phủ hơn 50% diện tích của ít nhất hai GT: dấu hiệu gộp từ thành dòng, cần xem overlay để xác nhận.",
              f"Có {covered_gt}/{total_gt} GT ({covered_gt / total_gt:.1%}) được một prediction bất kỳ phủ hơn 50% diện tích. Đây chỉ là chẩn đoán độ phủ, không phải recall chuẩn: box quá rộng cũng có thể đạt độ phủ cao.",
              f"Có {near} GT có best IoU trong [0.1, {summary['iou_threshold']}) trên ảnh chạy thành công; tổng FN là {missed_count} (kể cả ảnh lỗi). Đây là dấu hiệu sai hình học/gộp/tách, không đủ để kết luận một nguyên nhân duy nhất.", "",
              "## Hướng cải thiện cần kiểm chứng trên validation", "",
              "- Xem nhóm có recall thấp nhất; nếu chữ nhỏ bị bỏ sót, thử tăng kích thước ảnh đầu vào và đo lại chi phí latency.",
              "- Nếu overlay xác nhận gộp nhiều từ, ưu tiên thống nhất đơn vị nhãn và thử unclip_ratio trên validation trước khi fine-tuning.",
              "- Nếu nhiều FP, rà soát vùng chữ thiếu nhãn và thử box_thresh; nếu nhiều FN, kiểm tra threshold và dữ liệu khó.",
              "- Fine-tuning cần tập train/validation riêng; dữ liệu hiện có chỉ là test. Không chọn tham số tốt nhất bằng tập test này.", "",
              "## Giới hạn", "",
              summary["protocol"], "", summary["timing_scope"], "", summary["memory_scope"], "",
              "Không có CER/WER vì model chỉ định vị chữ. Không so trực tiếp H-mean này với bảng số liệu nội bộ PaddleOCR.", "",
              "Overlay: GT khớp xanh lá, GT bỏ sót đỏ, ignored xám, prediction xanh dương. Xem [CSV từng ảnh](per_image.csv) và thư mục error_examples."]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "diagnostics.json").write_text(json.dumps({"recall_by_size": dict(groups), "possible_merged_predictions": merged,
                                                        "gt_area_covered": covered_gt, "total_gt": total_gt,
                                                        "near_miss_gt": near}, indent=2), encoding="utf-8")
