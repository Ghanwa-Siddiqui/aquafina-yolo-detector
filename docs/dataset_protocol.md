# Dataset protocol

## What counts as positive

Annotate the **full visible extent of each bottle** when the Aquafina brand can be
identified visually. Include diverse label designs, sizes, orientations, lighting,
occlusion and backgrounds. A known Aquafina bottle with a completely hidden label
is not positive under this project's visible-evidence definition. Keep ambiguous
views as rejection examples. Do not infer brand solely from a blue cap or shape.

Annotate every qualifying bottle, including in mixed-brand scenes. Other bottles
receive no boxes. Use the same rule across training, validation and test. Exclude
unreviewable/corrupt images rather than invent labels. Rasterize orientation before
annotation so exported dimensions and pixel boxes match the image file exactly.

## Required COCO export

The canonical category list is exactly:

```json
[{"id": 1, "name": "aquafina_bottle"}]
```

Every image requires unique integer `id`, portable relative `file_name`, integer
`width`/`height`, nonempty `group_id`, `scene` and one `subset`:

| subset | Content | Positive annotations |
|---|---|---|
| `visible_brand` | Identifiable Aquafina, no competitor bottles | At least one |
| `mixed_brand` | Identifiable Aquafina with other brands | At least one, Aquafina only |
| `competitor` | Other identifiable bottle brands; no qualifying Aquafina | None |
| `ambiguous` | Bottle branding cannot be identified; no qualifying Aquafina | None |
| `empty` | No bottles | None |

`scene` is a consistent descriptive label such as `desk`, `shelf` or `outdoor`.
Use `group_id` for capture session/video/source sequence, not each frame. Keep
near-duplicates, bursts, crops and re-encoded copies in the same group. For a shared
source image with variants, give every variant the same group. Do not split videos
randomly by frame. Exact byte duplicates automatically merge groups; this does not
detect visual near-duplicates. Review these manually.

Each annotation requires unique positive integer `id` (start at 1), `image_id`, `category_id: 1` and
`bbox: [x, y, width, height]` in **original pixels**. Coordinates must remain inside
the image and width/height must be positive. Use `iscrowd: 0`. `area`, when supplied,
must equal bbox width × height; preparation fills area and iscrowd when omitted.
Crowds and more than 50 positive boxes per image are rejected explicitly.

Example negative image entry (also include it in `images`, even without annotations):

```json
{"id": 7, "file_name": "session02/competitor.jpg", "width": 1280, "height": 720,
 "group_id": "session02", "scene": "desk", "subset": "competitor"}
```

## Collection and splits

Prefer a broad range of independent sessions over many adjacent frames. Collect
lookalike brands and similar blue/white labels in backgrounds also used for
Aquafina. Include clear negatives, multiple bottles, reflections, blur and small
objects. Background and capture source must not become a shortcut for the brand.

Preparation validates every decoded image and bbox, hashes files and assigns whole
connected groups to approximately 70/15/15 splits. It optimizes subset/scene balance
using seed 42. Every split must contain both positives and competitor-only images;
an impossible allocation fails. Large groups can prevent exact ratios. Review the
saved manifest and collect more groups if important scenes/subsets are missing.

Outputs use YOLOX's expected directories (`train2017`, `val2017`, `test2017`) and
`annotations/train.json`, `val.json`, `test.json`. Negative image entries remain
present. Prepared version directories cannot be overwritten. Keep raw data and
prepared versions in Drive; copying images is intentional and needs Drive space.

For subsequent hard-negative collection, keep the original validation/test
membership fixed. Add new independent training groups to a new dataset version;
do not blindly rerun the random splitter over an expanded dataset. The initial
splitter is not an incremental split editor. Review and preserve the existing
manifest when preparing later versions.

## Interpretation

This one-class detector can learn brand cues but cannot guarantee perfect rejection
or identify a hidden label. Report performance separately for visible-brand,
mixed-brand, competitor, ambiguous and empty subsets where present. Report sample
counts; a low FPR on a handful of images is weak evidence. Threshold selection does
not impose a minimum recall: always examine recall before declaring the model useful.
