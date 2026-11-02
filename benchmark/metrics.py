"""Polygon IoU, ignore regions and maximum-cardinality one-to-one matching.

Local diagnostic protocol, not the official VinText competition evaluator.
"""
import numpy as np
import networkx as nx
from shapely.geometry import Polygon


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "hmean": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate_image(annotations, predictions, iou_threshold=0.5, ignore_overlap=0.5):
    gt = [(i, Polygon(a["polygon"])) for i, a in enumerate(annotations) if not a["ignore"]]
    ignored = [Polygon(a["polygon"]) for a in annotations if a["ignore"]]
    pred, excluded = [], []
    for i, item in enumerate(predictions):
        p = Polygon(item["polygon"])
        if not p.is_valid or p.area <= 0:
            raise ValueError(f"Invalid predicted polygon {i}")
        if any(p.intersection(g).area / p.area > ignore_overlap for g in ignored):
            excluded.append(i)
        else:
            pred.append((i, p))
    ious = np.zeros((len(gt), len(pred)))
    for i, (_, g) in enumerate(gt):
        for j, (_, p) in enumerate(pred):
            intersection = g.intersection(p).area
            ious[i, j] = intersection / (g.area + p.area - intersection)
    matches = []
    if ious.size:
        graph = nx.Graph()
        for i, j in np.argwhere(ious >= iou_threshold):
            graph.add_edge(("gt", int(i)), ("pred", int(j)), weight=float(ious[i, j]))
        pairs = nx.max_weight_matching(graph, maxcardinality=True)
        for left, right in pairs:
            if left[0] == "pred":
                left, right = right, left
            i, j = left[1], right[1]
            matches.append((gt[i][0], pred[j][0], float(ious[i, j])))
        matches.sort()
    matched_gt, matched_pred = {m[0] for m in matches}, {m[1] for m in matches}
    missed = [i for i, _ in gt if i not in matched_gt]
    false = [i for i, _ in pred if i not in matched_pred]
    return {**scores(len(matches), len(false), len(missed)), "matches": matches,
            "missed_gt": missed, "false_predictions": false, "ignored_predictions": excluded,
            "best_ious": [float(ious[i].max()) if len(pred) else 0.0 for i in range(len(gt))]}
