import json

import pytest

from aquafina_detector.predict import predict_media, restore_boxes


def test_letterbox_restore_clips_to_original_image():
    assert restore_boxes([[64, 32, 640, 640]], 2, 200, 100) == [[32, 16, 200, 100]]


def test_saved_video_has_one_result_per_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("Local OpenCV has no MJPG encoder")
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    output = tmp_path / "out"
    predict_media(lambda frame: [], source, output)
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [r["frame_index"] for r in rows] == [0, 1, 2]
    assert rows[2]["timestamp_seconds"] == 0.2
    assert all(r["detections"] == [] for r in rows)
    assert (output / "annotated.mp4").stat().st_size > 0
