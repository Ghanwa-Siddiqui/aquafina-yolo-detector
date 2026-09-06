import pytest


@pytest.fixture
def dataset():
    data = {"info": {"split": "val"}, "categories": [{"id": 1, "name": "aquafina_bottle"}],
            "images": [], "annotations": []}
    for i in range(30):
        positive = i % 2 == 0
        data["images"].append({"id": i, "file_name": f"{i}.png", "width": 100, "height": 80,
                               "group_id": f"session-{i // 2}", "scene": "desk",
                               "subset": "visible_brand" if positive else "competitor"})
        if positive:
            data["annotations"].append({"id": i + 1, "image_id": i, "category_id": 1,
                                         "bbox": [10, 10, 20, 40], "area": 800, "iscrowd": 0})
    return data
