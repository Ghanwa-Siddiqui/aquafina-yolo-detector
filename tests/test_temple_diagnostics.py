import copy
import json
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from test_temple import temple
from aquafina_detector.temple import audit_temple, snapshot_raw
from aquafina_detector.temple_diagnostics import (
    compare_boxes, duplicate_conflict_report, render_duplicate_group, save_duplicate_conflict_report,
)


def box(name='Aquafina', coords=(10, 10, 30, 50)):
    return {'class_name': name, 'bbox_xyxy': list(coords)}


@pytest.mark.parametrize('right,finding', [
    ([box(coords=(10.00001, 10, 30, 50))], 'coordinate_rounding_candidate'),
    ([box(coords=(20, 10, 40, 50))], 'different_boxes'),
    ([], 'missing_objects'), ([box('Deer')], 'different_class_labels'),
])
def test_box_diagnoses(right, finding):
    result = compare_boxes([box()], right, 100, 80)
    assert finding in result['findings']
    if result['matches']:
        assert 0 <= result['matches'][0]['iou'] <= 1


def test_matching_deltas_iou_order_and_ambiguity():
    result = compare_boxes([box()], [box(coords=(20, 10, 40, 50))], 100, 80)
    assert result['matches'][0]['iou'] == pytest.approx(1/3)
    assert result['matches'][0]['delta_xyxy_pixels_right_minus_left'] == [10, 0, 10, 0]
    assert compare_boxes([box(), box()], [box(), box()], 100, 80)['matches'][0]['ambiguous_match']
    other = box('Deer', (60, 10, 80, 50))
    assert not compare_boxes([box(), other], [other, box()], 100, 80)['findings']


def test_filename_only_differences_are_not_annotation_conflicts(temple):
    root, expected = temple
    (root / 'JPEGImages/img5.jpg').write_bytes((root / 'JPEGImages/img0.jpg').read_bytes())
    audit = audit_temple(root, expected)
    report = duplicate_conflict_report(audit)
    assert report['total_conflicting_groups'] == 0
    assert report['groups'][0]['comparisons'][0]['filename_only_annotation_difference']


def test_37_conflicts_all_nine_cross_split_panels_and_immutability(temple, tmp_path):
    root, expected = temple
    validation = ['img4', 'img5']
    for n in range(37):
        keys = [f'dup{n:02d}a', f'dup{n:02d}b']
        Image.new('RGB', (100, 80), (n*6, 200, 17)).save(root / 'JPEGImages' / (keys[0] + '.png'))
        (root / 'JPEGImages' / (keys[1] + '.png')).write_bytes((root / 'JPEGImages' / (keys[0] + '.png')).read_bytes())
        for j, key in enumerate(keys):
            tree = ET.parse(root / 'Annotations/img0.xml')
            tree.getroot().find('filename').text = key + '.png'
            tree.getroot().find('object/name').text = 'Aquafina' if j == 0 else 'Deer'
            tree.write(root / 'Annotations' / (key + '.xml'))
            (root / 'Labels' / (key + '.txt')).write_text(f'{j} 0.2 0.375 0.2 0.5\n')
        if n < 9:
            validation.append(keys[1])
    (root / 'ImageSets/Main/val.txt').write_text('\n'.join(validation))
    expected.update(images=80, xml=80, txt=83, train=69, val=11)
    expected.pop('darknet_instances')
    audit = audit_temple(root, expected)
    before, state = snapshot_raw(root), copy.deepcopy(audit)
    report = duplicate_conflict_report(audit)
    assert report['summary'] == {'cross_split': 9, 'train_only': 28, 'validation_only': 0}
    assert report['total_conflicting_groups'] == 37
    assert report['annotation_selection'] is None
    panels = [render_duplicate_group(audit, g) for g in report['groups'] if g['scope'] == 'cross_split']
    assert len(panels) == 9 and all(p.size == (1280, 540) for p in panels)
    member = report['groups'][0]['members'][0]
    assert member['original_darknet_lines'] and member['darknet'][0]['normalized_cxcywh']
    assert member['xml'][0]['bbox_xyxy'] == [10, 10, 30, 50]
    assert member['xml_darknet_comparison']['matches'][0]['iou'] == 1
    output = tmp_path / 'diagnostics/duplicate_conflict_report.json'
    save_duplicate_conflict_report(audit, report, output, tmp_path / 'processed')
    assert json.loads(output.read_text()) == report
    assert not (tmp_path / 'processed').exists()
    assert snapshot_raw(root) == before and audit == state


def test_validation_only_conflict_and_report_destination_guards(temple, tmp_path):
    root, expected = temple
    (root / 'JPEGImages/img5.jpg').write_bytes((root / 'JPEGImages/img4.jpg').read_bytes())
    audit = audit_temple(root, expected)
    report = duplicate_conflict_report(audit)
    assert report['summary']['validation_only'] == 1
    audit.audit['original_archive'] = str(tmp_path / 'drive/raw/temple/data.tar.gz')
    for destination in (root, tmp_path / 'processed', tmp_path / 'drive/raw/temple'):
        with pytest.raises(ValueError, match='separate'):
            save_duplicate_conflict_report(audit, report, destination / 'duplicate_conflict_report.json', tmp_path / 'processed')


def test_stale_source_rejected(temple):
    root, expected = temple
    (root / 'JPEGImages/img5.jpg').write_bytes((root / 'JPEGImages/img0.jpg').read_bytes())
    audit = audit_temple(root, expected)
    (root / 'Labels/img0.txt').write_text('changed')
    with pytest.raises(ValueError, match='changed after audit'):
        duplicate_conflict_report(audit)
