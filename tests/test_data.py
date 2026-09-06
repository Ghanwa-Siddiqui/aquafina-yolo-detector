import copy
from pathlib import Path

import pytest

from aquafina_detector.common import write_json
from aquafina_detector.data import grouped_split, prepare, validate, verify_prepared


def test_valid_negative_images_preserved(dataset):
    assert validate(dataset)["negative_images"] == 15
    splits = grouped_split(dataset)
    assert sum(len(p["images"]) for p in splits.values()) == 30
    assert all(validate(p)["negative_images"] > 0 for p in splits.values())


@pytest.mark.parametrize("mutation", [
    lambda d: d["categories"][0].update(id=0),
    lambda d: d["images"][1].update(id=0),
    lambda d: d["images"][0].update(file_name="../escape.png"),
    lambda d: d["images"][0].update(file_name="C:/escape.png"),
    lambda d: d["images"][0].update(group_id=""),
    lambda d: d["annotations"][0].update(bbox=[0, 0, -1, 20]),
    lambda d: d["annotations"][0].update(bbox=[0, 0, float("nan"), 20]),
    lambda d: d["annotations"][0].update(bbox=[90, 0, 20, 20]),
    lambda d: d["annotations"][0].update(image_id=1000),
    lambda d: d["annotations"][0].update(id=0),
    lambda d: d["annotations"][0].update(iscrowd=1),
    lambda d: d["images"][0].update(subset="competitor"),
])
def test_invalid_data_rejected(dataset, mutation):
    mutation(dataset)
    with pytest.raises(ValueError):
        validate(dataset)


def test_groups_and_duplicate_components_never_leak(dataset):
    hashes = {im["id"]: str(im["id"]) for im in dataset["images"]}
    hashes[4] = hashes[0]
    result = grouped_split(dataset, hashes)
    assert result == grouped_split(dataset, hashes)
    owners, image_split = {}, {}
    for split, data in result.items():
        for im in data["images"]:
            assert owners.setdefault(im["group_id"], split) == split
            image_split[im["id"]] = split
    assert image_split[0] == image_split[4]


def test_insufficient_independent_groups(dataset):
    for im in dataset["images"]:
        im["group_id"] = "one-video"
    with pytest.raises(ValueError, match="three independent"):
        grouped_split(dataset)


def test_prepare_real_images_and_immutable_output(dataset, tmp_path):
    from PIL import Image
    images = tmp_path / "raw"
    images.mkdir()
    for im in dataset["images"]:
        Image.new("RGB", (100, 80), (im["id"] * 7, 0, 0)).save(images / im["file_name"])
    annotations = tmp_path / "annotations.json"
    write_json(annotations, dataset)
    output = tmp_path / "prepared"
    manifest = prepare(annotations, images, output)
    assert sum(p["summary"]["images"] for p in manifest["splits"].values()) == 30
    assert len(list(output.rglob("*.png"))) == 30
    assert set(verify_prepared(output)) == {"train", "val", "test"}
    with pytest.raises(FileExistsError):
        prepare(annotations, images, output)
    damaged = next(output.rglob("*.png"))
    Image.new("RGB", (100, 80), (255, 255, 255)).save(damaged)
    with pytest.raises(ValueError, match="changed since manifest"):
        verify_prepared(output)


def test_no_competitor_coverage_rejected(dataset):
    for im in dataset["images"]:
        if im["subset"] == "competitor":
            im["subset"] = "empty"
    with pytest.raises(ValueError, match="Cannot produce"):
        grouped_split(dataset)
