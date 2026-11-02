"""Conservative source-guided removal of isolated extraction speckles."""
import math
import cv2
import numpy as np
from PIL import Image


def clean_layer(image, source_bbox, source_size):
    image = image.convert('RGBA')
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    w, h = image.size
    sw, sh = source_size
    box = [source_bbox[0]*w/sw, source_bbox[1]*h/sh,
           source_bbox[2]*w/sw, source_bbox[3]*h/sh]
    pad = max(3, math.ceil(max(box[2]-box[0], box[3]-box[1])*.1))
    l, t = max(0, math.floor(box[0])-pad), max(0, math.floor(box[1])-pad)
    r, b = min(w, math.ceil(box[2])+pad), min(h, math.ceil(box[3])+pad)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8), connectivity=8)
    # Threshold anchors only, preserving connected soft edges at their original alpha.
    anchors = set(np.unique(labels[t:b, l:r][alpha[t:b, l:r] >= 8])) - {0}
    report = {'source_bbox_scaled': box, 'anchor_alpha': 8, 'padding': pad,
              'removed_components': [], 'review_reasons': []}
    if not anchors:
        report['review_reasons'].append('no_confident_region_near_source_bbox')
        return image, report
    anchor_area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in anchors)
    limit = min(64, max(4, int(anchor_area*.005)))
    report['speckle_area_limit'] = limit
    nearby = set(np.unique(labels[t:b, l:r])) - {0}
    remove = np.zeros(count, dtype=bool)
    for i in range(1, count):
        x, y, width, height, area = map(int, stats[i])
        if i in nearby:
            if x < l or y < t or x+width > r or y+height > b:
                report['review_reasons'].append('connected_region_extends_beyond_source_bbox')
            continue
        if area <= limit:
            remove[i] = True
            report['removed_components'].append({'bbox': [x,y,x+width,y+height], 'area': area})
        else:
            report['review_reasons'].append('substantial_detached_region_outside_source_bbox')
    report['review_reasons'] = sorted(set(report['review_reasons']))
    pixels[remove[labels]] = 0
    return Image.fromarray(pixels), report
