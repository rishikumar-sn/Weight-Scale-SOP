"""CPU-bounded CIELAB region color predictions from unmodified source pixels."""
from collections import Counter
import math
import cv2
import numpy as np

COLOR_NOTE = ("Red includes maroon, dark red and related reds; blue and green include related shades. "
              "Dark colored stones may be black, purple, grey or other dark shades. "
              "Counts refer to detected regions; a continuous area may contain several stones.")
SWATCHES = {
    "Red": [(190,25,40), (110,15,30), (230,50,65)],
    "Pink": [(245,150,185), (230,85,145), (250,195,210)],
    "Green": [(20,130,65), (50,185,100), (15,65,35)],
    "Blue": [(35,90,190), (55,155,230), (15,35,100)],
    "Yellow": [(235,215,40), (235,225,130)],
    "Orange": [(240,125,25), (210,90,35)],
    "Purple": [(125,45,160), (80,30,95), (175,125,205)],
    "Brown": [(115,70,35), (155,100,60)], "Cyan": [(30,185,185)],
}
_NAMES = [name for name, shades in SWATCHES.items() for _ in shades]
_RGB = np.array([shade for shades in SWATCHES.values() for shade in shades], np.float32) / 255
_LAB = cv2.cvtColor(_RGB[None], cv2.COLOR_RGB2LAB)[0]


def original_color_crop(original, x, y, shape):
    """Map virtual padded tray coordinates to the source without resampling."""
    h, w = shape
    crop = np.zeros((h, w, 3), dtype=original.dtype)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(original.shape[1], x+w), min(original.shape[0], y+h)
    if x2 > x1 and y2 > y1:
        crop[y1-y:y2-y, x1-x:x2-x] = original[y1:y2, x1:x2]
    return crop


def _cluster(lab):
    # Deterministic farthest-point initial labels; no global random seed changes.
    k = min(4, len(np.unique(lab.round(1), axis=0)))
    centers = [lab[np.argmin(lab[:,0])]]
    for _ in range(1, k):
        distance = np.min(((lab[:,None] - np.array(centers)[None]) ** 2).sum(2), axis=1)
        centers.append(lab[np.argmax(distance)])
    centers = np.asarray(centers, dtype=np.float32)
    # NumPy Lloyd iterations avoid consuming OpenCV's process-wide RNG, which
    # is also used by parts of the existing segmentation workflow.
    for _ in range(15):
        labels = np.argmin(((lab[:, None] - centers[None]) ** 2).sum(2), axis=1)
        updated = np.asarray([
            lab[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ], dtype=np.float32)
        movement = np.max(np.linalg.norm(updated - centers, axis=1))
        centers = updated
        if movement <= .5:
            break
    labels = np.argmin(((lab[:, None] - centers[None]) ** 2).sum(2), axis=1)
    return centers, np.bincount(labels, minlength=k) / len(lab)



def predict_gemstone_color(source_bgr, mask, max_samples=1536):
    if mask.shape != source_bgr.shape[:2]:
        raise ValueError("Color mask must match original source dimensions")
    selected = mask > 0
    interior = cv2.erode(selected.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    if np.count_nonzero(interior) >= max(16, np.count_nonzero(selected) * .4):
        selected = interior
    pixels = source_bgr[selected]
    total = len(pixels)
    result = dict(color="Uncertain", confidence=0.0, sample_count=min(total,max_samples), reason="Insufficient color evidence")
    if total < 12:
        return result
    pixels = pixels[np.linspace(0,total-1,min(total,max_samples),dtype=int)]
    lab = cv2.cvtColor(pixels[:,None].astype(np.float32)/255, cv2.COLOR_BGR2LAB)[:,0]
    centers, fractions = _cluster(lab)
    chroma = np.linalg.norm(centers[:,1:],axis=1)
    chromatic = (chroma >= 12) & (centers[:,0] >= 7) & (centers[:,0] < 97)
    evidence = float(fractions[chromatic].sum())
    votes, distances = Counter(), []
    if evidence >= .12:
        for center, fraction in zip(centers[chromatic],fractions[chromatic]):
            delta = _LAB - center
            delta[:,0] *= .65
            distance = np.linalg.norm(delta,axis=1)
            nearest = int(np.argmin(distance))
            votes[_NAMES[nearest]] += float(fraction)
            distances.append((float(fraction),float(distance[nearest])))
        ranked = votes.most_common()
        name, support = ranked[0]
        agreement = support/evidence
        margin = (support - (ranked[1][1] if len(ranked)>1 else 0))/evidence
        distance = sum(w*d for w,d in distances)/evidence
        confidence = agreement * min(1.,evidence/.35) * max(0.,1-distance/65)
        if agreement >= .6 and margin >= .2 and distance < 38 and confidence >= .45:
            result.update(color=name,confidence=round(confidence,3),reason="Dominant chromatic cluster distribution")
        else:
            result.update(confidence=round(confidence,3),reason="Mixed or weak chromatic evidence")
    else:
        neutral = chroma < 12
        dark = float(fractions[neutral & (centers[:,0]>=7) & (centers[:,0]<32)].sum())
        light = float(fractions[neutral & (centers[:,0]>=65) & (centers[:,0]<98)].sum())
        grey = float(fractions[neutral & (centers[:,0]>=32) & (centers[:,0]<65)].sum())
        if max(dark,light,grey) >= .7:
            name,strength = max((("Black",dark),("White/colorless",light),("Grey",grey)),key=lambda v:v[1])
            result.update(color=name,confidence=round(strength*.75,3),reason="Predominantly neutral region; dark shades are ambiguous")
    result["clusters"] = [dict(lab=[round(float(v),2) for v in c],fraction=round(float(f),3)) for c,f in zip(centers,fractions)]
    return result


def describe_regions(source_bgr, regions, jewel_mask, scale=None):
    """Project masks back to native pixels without modifying segmentation."""
    h,w = source_bgr.shape[:2]
    entries = []
    jewel_area = max(1,np.count_nonzero(cv2.resize(jewel_mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)))
    for i,region in enumerate(regions,1):
        native = cv2.resize(region["mask"].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        area = int(np.count_nonzero(native))
        prediction = predict_gemstone_color(source_bgr,native)
        area_mm2 = None
        if scale and float(scale.get("mm_per_pixel_x",0))>0 and float(scale.get("mm_per_pixel_y",0))>0:
            area_mm2 = area * float(scale["mm_per_pixel_x"]) * float(scale["mm_per_pixel_y"])
        diameter = math.sqrt(4*area_mm2/math.pi) if area_mm2 is not None else None
        ratio = area/jewel_area
        size = ("small" if diameter<2 else "medium" if diameter<4 else "large") if diameter is not None else ("small" if ratio<.01 else "medium" if ratio<.05 else "large")
        component_count, _, stats, _ = cv2.connectedComponentsWithStats((native > 0).astype(np.uint8))
        connected_fraction = float(stats[1:, cv2.CC_STAT_AREA].max()) / max(1, area) if component_count > 1 else 0
        continuous = connected_fraction >= .85 and (area_mm2>=25 if area_mm2 is not None else ratio>=.15)
        entries.append(dict(region_id=i,**prediction,size=size,area_px=area,area_mm2=round(area_mm2,3) if area_mm2 is not None else None,
            reportable=area >= 12 and (area_mm2 >= .5 if area_mm2 is not None else ratio >= .0005),
            size_basis="equivalent diameter: small <2 mm, medium 2-4 mm, large >=4 mm" if diameter is not None else "relative mask area: small <1%, medium 1-5%, large >=5% of jewellery",continuous_area=continuous))
    return entries


def stone_description(entries, label="jewellery", found=True):
    """Keep print insights brief: ignore tiny regions and uncertain colors."""
    entries = [entry for entry in entries if entry.get("reportable", True)] if found else []
    if not entries:
        return dict(region_count=0, regions=[], summary="", color_note="")
    sizes = Counter(entry["size"] for entry in entries)
    size_text = ", ".join(f"{sizes[size]} {size}" for size in ("small", "medium", "large") if sizes[size])
    text = f"This {label} has {len(entries)} detected stone regions ({size_text})."
    # Area-weighted votes describe the visibly prevalent colors, rather than
    # listing every shade or allowing numerous tiny flecks to dominate.
    votes = Counter()
    for entry in entries:
        if entry["color"] == "Uncertain" or entry["confidence"] < .55:
            continue
        color = "dark colored" if entry["color"] in {"Black", "Purple", "Grey"} else entry["color"].lower()
        votes[color] += entry.get("area_mm2") or entry["area_px"]
    total = sum(votes.values())
    dominant = [color for color, area in votes.most_common(2) if area >= total * .2]
    if dominant:
        text += " Most present stone colors: " + " and ".join(dominant) + "."
    if any(entry["continuous_area"] for entry in entries):
        text += " A large continuous stone area is present; its individual stones cannot be counted reliably."
    size_note = (
        " Sizes use equivalent face-up diameter: small <2 mm, medium 2 to <4 mm, large >=4 mm."
        if all(entry["area_mm2"] is not None for entry in entries) else
        " Sizes use relative mask area without calibration: small <1%, medium 1 to <5%, large >=5% of jewellery."
    )
    note = (COLOR_NOTE if dominant else "Counts refer to detected regions, rather than individual stones.") + size_note
    return dict(region_count=len(entries), regions=entries, summary=text, color_note=note)
