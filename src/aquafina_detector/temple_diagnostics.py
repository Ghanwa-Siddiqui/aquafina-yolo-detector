"""Read-only duplicate inspection; report export never changes audit decisions."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from .common import contained, sha256, write_json
from .temple import CLASSES, PROCESSED_ROOT


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def compare_boxes(left, right, width, height):
    """Geometry-first greedy pairing for inspection only; disclose ties/unmatched boxes.

    No class restriction: otherwise identical boxes with changed classes must be
    visible. Pair all possible boxes, even IoU=0, and expose unmatched indices.
    This heuristic cannot establish object identity or approve an annotation.
    """
    edges = sorted((-_iou(a['bbox_xyxy'], b['bbox_xyxy']),
                    sum(abs(x-y) for x, y in zip(a['bbox_xyxy'], b['bbox_xyxy'])), i, j)
                   for i, a in enumerate(left) for j, b in enumerate(right))
    used_left, used_right, matches = set(), set(), []
    for score, distance, i, j in edges:
        if i in used_left or j in used_right:
            continue
        ties = [(x, y) for s, d, x, y in edges if s == score and d == distance
                and x not in used_left and y not in used_right and (x == i or y == j)]
        used_left.add(i)
        used_right.add(j)
        delta = [b-a for a, b in zip(left[i]['bbox_xyxy'], right[j]['bbox_xyxy'])]
        matches.append({'left_index': i, 'right_index': j, 'iou': -score,
                        'delta_xyxy_pixels_right_minus_left': delta,
                        'delta_xyxy_normalized': [v/s for v, s in zip(delta, (width, height, width, height))],
                        'class_changed': left[i]['class_name'].casefold() != right[j]['class_name'].casefold(),
                        'ambiguous_match': len(ties) > 1})
    findings = []
    if len(left) != len(right):
        findings.append('missing_objects')
    if any(m['class_changed'] for m in matches):
        findings.append('different_class_labels')
    changed = [m for m in matches if any(m['delta_xyxy_normalized'])]
    if changed:
        findings.append('coordinate_rounding_candidate' if all(
            max(abs(v) for v in m['delta_xyxy_normalized']) <= 1e-6 for m in changed) else 'different_boxes')
    return {'matches': matches, 'unmatched_left': sorted(set(range(len(left))) - used_left),
            'unmatched_right': sorted(set(range(len(right))) - used_right), 'findings': findings,
            'matching_method': 'descending IoU, ascending L1 distance, index tie-break; diagnostic only'}


def _read_verified(audit, relative):
    path = contained(audit.root, relative)
    if sha256(path) != audit.snapshot[relative]:
        raise ValueError('Staged source changed after audit: ' + relative)
    return path.read_text(encoding='utf-8-sig')


def duplicate_conflict_report(audit):
    """Inspect every exact-image group, including groups blocked for other reasons."""
    paths = defaultdict(list)
    for relative in audit.snapshot:
        path = Path(relative)
        if path.parts[0] in ('Annotations', 'Labels'):
            paths[(path.parts[0], path.stem)].append(relative)
    groups = defaultdict(list)
    for key, record in audit.records.items():
        groups[record['sha256']].append(key)
    validation = set(audit.audit['splits']['original_validation_ids'])
    result = []
    for digest, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        members = []
        for key in sorted(ids):
            record = audit.records[key]
            member = {'image_id': key, 'assigned_split': 'val' if key in validation else 'train',
                      'file_name': record['file_name'], 'width': record['width'], 'height': record['height'],
                      'paired_annotations_verified': record['paired_annotations_verified'],
                      'darknet': [], 'xml': [], 'parse_errors': []}
            for folder in ('Annotations', 'Labels'):
                matches = paths[(folder, key)]
                if len(matches) != 1:
                    member['parse_errors'].append(f'{folder}: expected one source file, got {len(matches)}')
                    continue
                source = _read_verified(audit, matches[0])
                member[folder.lower() + '_path'] = matches[0]
                if folder == 'Labels':
                    member['original_darknet_lines'] = source.splitlines()
                    for line_number, line in enumerate(source.splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            parts = line.split()
                            if len(parts) != 5:
                                raise ValueError('Expected five fields')
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                                raise ValueError('Non-finite coordinate')
                            corners = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
                            member['darknet'].append({'line_number': line_number, 'original_line': line,
                                'class_id': class_id, 'class_name': CLASSES.get(class_id, str(class_id)),
                                'normalized_cxcywh': [cx, cy, w, h], 'normalized_xyxy': corners,
                                'bbox_xyxy': [v*s for v, s in zip(corners, (record['width'], record['height'])*2)]})
                        except ValueError as exc:
                            member['parse_errors'].append(f'Darknet line {line_number}: {exc}')
                else:
                    try:
                        tree = ET.fromstring(source)
                        member['xml_filename'] = tree.findtext('filename')
                        for index, obj in enumerate(tree.findall('object')):
                            box = [float(obj.findtext('bndbox/' + name)) for name in ('xmin', 'ymin', 'xmax', 'ymax')]
                            if not all(math.isfinite(v) for v in box):
                                raise ValueError('Non-finite XML coordinate')
                            member['xml'].append({'object_index': index, 'class_name': obj.findtext('name', '').strip(),
                                                  'bbox_xyxy': box})
                    except (ET.ParseError, ValueError, TypeError) as exc:
                        member['parse_errors'].append(f'XML: {exc}')
            member['xml_darknet_comparison'] = compare_boxes(member['xml'], member['darknet'], record['width'], record['height'])
            members.append(member)
        comparisons, findings = [], set()
        for a, b in combinations(members, 2):
            pair = {'left_image_id': a['image_id'], 'right_image_id': b['image_id']}
            for source in ('xml', 'darknet'):
                pair[source] = compare_boxes(a[source], b[source], a['width'], a['height'])
                findings.update(pair[source]['findings'])
            pair['xml_filename_differs'] = a.get('xml_filename') != b.get('xml_filename')
            if pair['xml_filename_differs'] and not any(pair[s]['findings'] for s in ('xml', 'darknet')):
                pair['filename_only_annotation_difference'] = True
            comparisons.append(pair)
        splits = {m['assigned_split'] for m in members}
        scope = 'cross_split' if len(splits) > 1 else ('validation_only' if 'val' in splits else 'train_only')
        reasons = [i['kind'] for i in audit.anomalies['issues'] if i.get('sha256') == digest]
        conflict = bool(findings or any(m['parse_errors'] for m in members))
        result.append({'sha256': digest, 'scope': scope, 'annotation_conflict': conflict,
                       'audit_issue_kinds': reasons, 'findings': sorted(findings),
                       'members': members, 'comparisons': comparisons})
    counts = Counter(g['scope'] for g in result if g['annotation_conflict'])
    return {'schema': 1, 'source': 'TempleRAIL', 'archive_md5': audit.audit.get('archive_md5'),
            'summary': {scope: counts[scope] for scope in ('cross_split', 'train_only', 'validation_only')},
            'total_conflicting_groups': sum(counts.values()), 'groups': result,
            'rounding_policy': 'Differences <=1e-6 normalized are rounding candidates, not proven harmless; no audit tolerance changes.',
            'filename_policy': 'Filename-only differences are already ignored by annotation equivalence and cannot explain a conflict alone.',
            'audit_status_unchanged': audit.audit['status'], 'annotation_selection': None,
            'processed_dataset_written': False}


def render_duplicate_group(audit, group):
    """One side-by-side panel per file: cyan Darknet, magenta XML, numbered boxes."""
    from PIL import Image, ImageDraw
    panels = []
    for member in group['members']:
        path = contained(audit.root / 'JPEGImages', member['file_name'])
        if sha256(path) != group['sha256']:
            raise ValueError('Duplicate image changed after audit')
        with Image.open(path) as source:
            picture = source.convert('RGB')
        picture.thumbnail((640, 480))
        panel = Image.new('RGB', (640, 540), 'white')
        panel.paste(picture, (0, 60))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 5), f"{member['image_id']} | {member['assigned_split']}\ncyan=Darknet; magenta=XML (box indices)", fill='black')
        for source, color in (('darknet', 'cyan'), ('xml', 'magenta')):
            for index, obj in enumerate(member[source]):
                box = [v*s for v, s in zip(obj['bbox_xyxy'], (picture.width/member['width'], picture.height/member['height'])*2)]
                box[1] += 60
                box[3] += 60
                draw.rectangle(box, outline=color, width=2 if source == 'xml' else 4)
                draw.text((box[0], box[1] + (12 if source == 'xml' else 0)), f"{source}[{index}] {obj['class_name']}", fill=color)
        panels.append(panel)
    canvas = Image.new('RGB', (640*len(panels), 540), 'white')
    for index, panel in enumerate(panels):
        canvas.paste(panel, (640*index, 0))
    return canvas


def save_duplicate_conflict_report(audit, report, path, processed_root=PROCESSED_ROOT):
    """Only write the diagnostic JSON, outside raw/staged/processed directories."""
    path = Path(path).resolve()
    protected = [audit.root.resolve(), Path(processed_root).resolve()]
    if audit.audit.get('original_archive'):
        protected.append(Path(audit.audit['original_archive']).resolve().parent)
    if path.name != 'duplicate_conflict_report.json' or any(path.is_relative_to(root) for root in protected):
        raise ValueError('Diagnostic report must be separate from raw and processed data')
    write_json(path, report)
