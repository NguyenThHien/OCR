import pytest

from benchmark.dataset import read_annotations
from benchmark.metrics import evaluate_image


def box(x=0):
    return [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]


def test_duplicate_is_false_positive():
    result = evaluate_image([{"polygon": box(), "ignore": False}], [{"polygon": box()}, {"polygon": box()}])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)


def test_ignore_and_missed():
    result = evaluate_image([{"polygon": box(), "ignore": True}, {"polygon": box(20), "ignore": False}], [{"polygon": box()}])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


def test_empty_ground_truth_and_false_positive():
    assert evaluate_image([], [{"polygon": box()}])["fp"] == 1
    assert evaluate_image([], [])["hmean"] == 0


def test_parser_preserves_commas_and_ignore(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("0,0,10,0,10,10,0,10,Xin,chào\n0,0,10,0,10,10,0,10,###", encoding="utf-8-sig")
    items = read_annotations(path)
    assert items[0]["text"] == "Xin,chào"
    assert items[1]["ignore"]


def test_invalid_polygon_rejected(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("0,0,0,0,0,0,0,0,text", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid polygon"):
        read_annotations(path)


def test_matching_prioritizes_two_valid_pairs_over_one_high_iou():
    # IoUs: [[1, .538], [.538, .25]]; optimal valid matching is off diagonal.
    result = evaluate_image(
        [{"polygon": box(0), "ignore": False}, {"polygon": box(3), "ignore": False}],
        [{"polygon": box(0)}, {"polygon": box(-3)}],
    )
    assert result["tp"] == 2
