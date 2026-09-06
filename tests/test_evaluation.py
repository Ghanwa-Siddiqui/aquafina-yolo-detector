import pytest

from aquafina_detector.evaluate import operating_metrics, select_threshold, validate_predictions, coco_ap


def prediction(image_id, score=0.9, bbox=None):
    return {"image_id": image_id, "score": score, "category_id": 1,
            "bbox": bbox or [10, 10, 20, 40]}


def test_duplicate_detections_are_false_positives(dataset):
    result = operating_metrics(dataset, [prediction(0), prediction(0, 0.8)], 0.5)["overall"]
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 14)


def test_negative_image_rate_is_not_box_rate(dataset):
    result = operating_metrics(dataset, [prediction(1), prediction(1, 0.8)], 0.5)["overall"]
    assert result["fp_per_negative_image"] == pytest.approx(2 / 15)
    assert result["competitor_image_fpr"] == pytest.approx(1 / 15)


def test_selects_highest_recall_with_rejection(dataset):
    predictions = [prediction(i) for i in range(0, 30, 2)] + [prediction(1, 0.6)]
    result = select_threshold(dataset, predictions)
    assert result["status"] == "qualified"
    assert result["threshold"] == 0.9
    assert result["operating_report"]["overall"]["recall"] == 1


def test_empty_predictions_cannot_qualify(dataset):
    assert select_threshold(dataset, [])["status"] == "no_qualifying_threshold"
    assert coco_ap(dataset, []) == {"ap50": 0.0, "ap50_95": 0.0}


def test_bad_model_cannot_qualify(dataset):
    result = select_threshold(dataset, [prediction(0, 0.5), prediction(1, 0.99)])
    assert result["threshold"] is None


def test_test_set_cannot_select_threshold(dataset):
    dataset["info"]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_threshold(dataset, [prediction(0)])


def test_missing_competitors_is_unknown_not_perfect(dataset):
    for im in dataset["images"]:
        if im["subset"] == "competitor":
            im["subset"] = "ambiguous"
    assert select_threshold(dataset, [prediction(0)])["threshold"] is None


def test_unknown_prediction_rejected(dataset):
    with pytest.raises(ValueError):
        validate_predictions(dataset, [prediction(999)])


def test_sweep_matches_brute_force_with_ties(dataset):
    predictions = [prediction(i, 0.8 if i < 10 else 0.6) for i in range(0, 30, 2)]
    predictions += [prediction(1, 0.7), prediction(0, 0.6)]
    candidates = []
    for threshold in {p["score"] for p in predictions}:
        m = operating_metrics(dataset, predictions, threshold)["overall"]
        if m["precision"] >= 0.95 and m["competitor_image_fpr"] <= 0.05:
            candidates.append((m["recall"], m["precision"], -m["competitor_image_fpr"], threshold))
    assert select_threshold(dataset, predictions)["threshold"] == max(candidates)[3]


def test_real_coco_ap_perfect_and_wrong_boxes(dataset):
    pytest.importorskip("pycocotools")
    correct = [prediction(i) for i in range(0, 30, 2)]
    assert coco_ap(dataset, correct)["ap50_95"] == pytest.approx(1.0)
    wrong = [prediction(i, bbox=[70, 0, 10, 10]) for i in range(0, 30, 2)]
    assert coco_ap(dataset, wrong)["ap50"] == 0


def test_frozen_point_rejects_different_checkpoint(dataset, tmp_path):
    from aquafina_detector.common import sha256, write_json
    from aquafina_detector.evaluate import evaluate_files
    dataset["info"]["split"] = "test"
    ann, pred, point = (tmp_path / name for name in ("ann.json", "pred.json", "point.json"))
    write_json(ann, dataset)
    write_json(pred, [prediction(0)])
    meta = {"annotations_sha256": sha256(ann), "checkpoint_sha256": "new",
            "input_size": [640, 640], "nms": 0.65, "score_floor": 0.001, "upstream_commit": "pin"}
    write_json(str(pred) + ".meta.json", meta)
    write_json(point, {"status": "qualified", "threshold": 0.8,
                       "provenance": dict(meta, checkpoint_sha256="old")})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        evaluate_files(ann, pred, tmp_path / "out.json", point)
