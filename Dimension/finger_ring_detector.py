"""Fit visible finger-ring openings and band edges in original image coordinates."""

import cv2
import numpy as np


class RingDimensionsUnavailable(RuntimeError):
    """The capture does not support a reliable ring boundary measurement."""


def _points(ellipse, offsets):
    (cx, cy), (da, db), angle = ellipse
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ca, sa = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    x, y = da / 2 * np.cos(theta), db / 2 * np.sin(theta)
    # Unit outward normals, rather than rays, also work on tilted bands.
    nx, ny = np.cos(theta) / da, np.sin(theta) / db
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    xs = cx + ca * x - sa * y + np.asarray(offsets)[:, None] * (ca * nx - sa * ny)
    ys = cy + sa * x + ca * y + np.asarray(offsets)[:, None] * (sa * nx + ca * ny)
    return xs.astype(np.float32), ys.astype(np.float32)


def _fit_supported(points, seed, tolerance):
    """Trim attached decoration/shadow points without moving to another object."""
    if len(points) < 90:
        return None

    def inliers(ellipse):
        (cx, cy), (da, db), angle = ellipse
        if min(da, db) < min(seed[1]) * .8 or max(da, db) > max(seed[1]) * 1.7:
            return np.zeros(len(points), dtype=bool)
        if np.linalg.norm(np.array(ellipse[0]) - seed[0]) > min(seed[1]) * .10:
            return np.zeros(len(points), dtype=bool)
        ca, sa = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        p = points - (cx, cy)
        radial = np.hypot((ca * p[:, 0] + sa * p[:, 1]) / (da / 2),
                          (-sa * p[:, 0] + ca * p[:, 1]) / (db / 2))
        return np.abs(radial - 1) * min(da, db) / 2 < tolerance

    rng = np.random.default_rng(0)
    keep = inliers(seed)
    for _ in range(80):
        sample = points[rng.choice(len(points), 12, replace=False)]
        candidate = cv2.fitEllipse(sample.astype(np.float32).reshape(-1, 1, 2))
        candidate_keep = inliers(candidate)
        if candidate_keep.sum() > keep.sum():
            keep = candidate_keep
    for _ in range(3):
        if keep.sum() < 90:
            return None
        ellipse = cv2.fitEllipse(points[keep].astype(np.float32).reshape(-1, 1, 2))
        keep = inliers(ellipse)
    if keep.mean() < .70:
        return None
    return ellipse, float(keep.mean())


def visible_ring_ellipses(img, scale=None, *, allow_open=False, return_evidence=False):
    """Fit observed edges; optionally seed an open band across its missing gap.

    Closing and convex envelopes only propose ellipses. All measurements and
    angular-support checks use the original image, preserving real band gaps.
    """
    if img is None or img.size == 0:
        raise ValueError('Empty image supplied to finger-ring detection.')
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError('Finger-ring scale must be finite and positive.')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), .7)
    min_dim = min(gray.shape)
    min_radius = max(10., min_dim * .018) if scale is None else max(5., 2.5 / scale)
    max_radius = min_dim * .49 if scale is None else min(min_dim * .49, 18 / scale)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(hsv, (2, 35, 25), (48, 255, 255))
    masks = [warm]
    masks += [cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, c)
              for block in (51, 101, 201) for c in (4, 10)]
    if allow_open:
        # These masks propose a closed opening only. Measurements below always
        # sample the untouched gray image, never the artificially joined gap.
        proposals = []
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_proposal = mask.copy()
            for contour in contours:
                if cv2.contourArea(contour) < 80:
                    continue
                # Close the gap with the body's convex envelope to obtain a
                # seed. The added hull pixels never become measurement points.
                cv2.drawContours(hull_proposal, [cv2.convexHull(contour)], -1, 255, 2)
            proposals.append(hull_proposal)
        masks = proposals + [cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (max(5, round(min_dim * fraction)) | 1,) * 2))
                 for mask in masks for fraction in (.08, .12, .18, .24)]
    texture = gray.astype(float) - cv2.GaussianBlur(gray, (0, 0), 3).astype(float)
    candidates = []
    seeds = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] < 0 or len(contour) < 40:
                continue
            area = cv2.contourArea(contour)
            if area < np.pi * min_radius ** 2:
                continue
            seed = cv2.fitEllipse(contour)
            (cx, cy), (da, db), angle = seed
            if not min_radius * 2 <= max(da, db) <= max_radius * 2:
                continue
            if min(da, db) / max(da, db) < .55:
                continue
            if abs(area / (np.pi * da * db / 4) - 1) > .12:
                continue
            # The opening must be smooth background, not a textured ornament.
            interior = np.zeros_like(gray)
            small = (seed[0], (da * .70, db * .70), angle)
            cv2.ellipse(interior, small, 255, -1)
            values = gray[interior > 0]
            outside_x, outside_y = _points(seed, [max(da, db) * .4])
            outside_valid = ((outside_x >= 0) & (outside_x < gray.shape[1] - 1)
                             & (outside_y >= 0) & (outside_y < gray.shape[0] - 1))
            outside = cv2.remap(gray, outside_x, outside_y, cv2.INTER_LINEAR)[outside_valid]
            if outside.size < 20:
                # Tightly cropped exports may put the normal samples outside
                # the image; use the actual border pixels as background evidence.
                outside = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
            if values.size == 0 or np.std(texture[interior > 0]) > 5 or abs(np.median(values) - np.median(outside)) > 30:
                continue
            if np.mean(warm[interior > 0] > 0) > .15:
                continue
            if any(np.linalg.norm(np.array(seed[0]) - old[0]) < 2 and
                   np.max(np.abs(np.array(seed[1]) - old[1])) < 3 for old in seeds):
                continue
            seeds.append(seed)
            radius = min(da, db) / 2
            tolerance = max(1.5, radius * .035)
            # First align the inside edge. Negative gradient leaves the light
            # opening and enters the band. Search locally to avoid the far wall.
            offsets = np.arange(-max(3., radius * .08), max(3., radius * .08) + .25, .5)
            xs, ys = _points(seed, offsets)
            profile = cv2.remap(smooth, xs, ys, cv2.INTER_LINEAR).astype(float)
            gradient = np.gradient(profile, .5, axis=0)
            best = np.argmin(gradient, axis=0)
            strength = -gradient[best, np.arange(180)]
            valid = strength > 3
            if valid.mean() < .65:
                continue
            points = np.column_stack((xs[best, np.arange(180)], ys[best, np.arange(180)]))
            fitted = _fit_supported(points[valid], seed, tolerance)
            if fitted is None:
                continue
            inner, inner_support = fitted
            # Find the transition back to background outside the band, excluding
            # bright stripes whose far side falls dark again.
            offsets = np.arange(2., max(10., max(da, db) * .30), .5)
            xs, ys = _points(inner, offsets)
            profile = cv2.remap(smooth, xs, ys, cv2.INTER_LINEAR).astype(float)
            gradient = np.gradient(profile, .5, axis=0)
            background = np.median(values)
            recovered = profile >= background - 18
            sustained = np.minimum.accumulate(recovered[::-1], axis=0)[::-1]
            # Evaluate recovery a few pixels after the edge peak. Requiring it
            # at the peak itself places OD on the far edge of the shadow halo.
            sustained = np.vstack((sustained[8:], np.repeat(sustained[-1:], 8, axis=0)))
            merit = np.where(sustained, gradient, -100.)
            best = np.argmax(merit, axis=0)
            valid = merit[best, np.arange(180)] > 3
            if valid.mean() < .65:
                continue
            points = np.column_stack((xs[best, np.arange(180)], ys[best, np.arange(180)]))
            # Use only visible arcs; do not fit the gemstone setting as OD.
            fitted = _fit_supported(points[valid], inner, tolerance * 1.5)
            if fitted is None:
                continue
            outer, outer_support = fitted
            if min(np.array(outer[1]) - inner[1]) < 2 or np.linalg.norm(np.array(outer[0]) - inner[0]) > radius * (.18 if allow_open else .12):
                continue
            # Different ellipse orientations can cross despite ordered axis
            # lengths. The complete opening must remain inside the outer band.
            ix, iy = _points(inner, [0])
            angle = np.deg2rad(outer[2])
            dx, dy = ix - outer[0][0], iy - outer[0][1]
            distance = np.hypot((np.cos(angle) * dx + np.sin(angle) * dy) / (outer[1][0] / 2),
                                (-np.sin(angle) * dx + np.cos(angle) * dy) / (outer[1][1] / 2))
            if np.max(distance) >= 1:
                continue
            if scale is not None:
                od, id_ = sum(outer[1]) * scale / 2, sum(inner[1]) * scale / 2
                if not (8 <= od <= 50 and 5 <= id_ <= 36 and .2 <= (od - id_) / 2 <= 10):
                    continue
            center_distance = np.linalg.norm(np.array(outer[0]) - inner[0])
            score = (inner_support + outer_support + valid.mean()
                     + min(area / (np.pi * max_radius ** 2), 1.)
                     - center_distance / max(radius * .1, 1.))
            evidence = dict(is_open_ring_estimate=False,
                            opening_description="Visible; ellipse fitted",
                            measurement_note="Ellipse long axis; assumes a circular band")
            if allow_open:
                # Confirm support on the FINAL ellipse, in all angular sectors.
                # A large missing arc, obstruction or almost edge-on band must
                # not be converted into an apparently measured complete ring.
                offsets = np.arange(-2., 2.01, .5)
                xs, ys = _points(inner, offsets)
                profile = cv2.remap(smooth, xs, ys, cv2.INTER_LINEAR).astype(float)
                support = -np.min(np.gradient(profile, .5, axis=0), axis=0) > 3
                coverage = float(support.mean())
                doubled = np.tile(~support, 2)
                runs = np.diff(np.flatnonzero(np.diff(np.r_[False, doubled, False])))
                gap = min(180, int(runs[::2].max(initial=0))) * 2
                # At least three quarters of the inner outline must be visible.
                # Require a genuine gap, not a fully visible band rejected for
                # another reason by the ordinary detector.
                if coverage < .75 or not 8 <= gap <= 80 or outer_support < .75:
                    continue
                # No part of the inferred ellipse may leave the captured crop.
                ex, ey = _points(inner, [0])
                if (ex.min() < 1 or ey.min() < 1 or ex.max() >= img.shape[1]-1
                        or ey.max() >= img.shape[0]-1):
                    continue
                evidence = dict(is_open_ring_estimate=True,
                    opening_description="Open ring; missing arc estimated",
                    measurement_note="Visible arc fitted; missing gap estimated",
                    visible_arc_fraction=coverage, largest_missing_arc_degrees=gap,
                    estimate_note="Current opening only; adjustable ring fit can change")
            candidates.append((score, outer, inner, evidence))
    if not candidates:
        message = ('Not enough consistent inner-band arc is visible for an approximate size. '
                   'Place the ring flat with most of its band visible.' if allow_open else
                   'Finger-ring opening or both band boundaries are not clearly visible. '
                   'Place the ring flat with its center hole clear.')
        raise RingDimensionsUnavailable(message)
    _, outer, inner, evidence = max(candidates, key=lambda item: item[0])
    return (outer, inner, evidence) if return_evidence else (outer, inner)


def measure_ring_ellipses(img, scale=None):
    """Prefer directly visible boundaries; approximate only supported open bands."""
    try:
        return visible_ring_ellipses(img, scale, return_evidence=True)
    except RingDimensionsUnavailable:
        return visible_ring_ellipses(img, scale, allow_open=True, return_evidence=True)
