# Marker-padding split correction

A jewellery strand can pass beside the AprilTag and remain fully visible in the
camera image, while the marker's enlarged exclusion rectangle erases it from the
foreground mask. Connected-component separation can then report one chain twice.

`backend/app/analysis/marker_continuity.py` runs after foreground separation and
before model verification/classification. It considers only two or more accepted
components whose foreground pixels reach the marker erase boundary. A single
already-connected item is unchanged merely because it is near the marker.

For connectivity evidence, the function maps the original camera image into the
working crop and requires every surviving foreground pixel to match exactly.
It uses the detected tag quadrilateral, extended outward by 15% of the tag size on each side (1.30× about its centre), plus a
clearance of at least 2 pixels (4% of the longest detected tag side). That area
remains protected. The smaller polygon is used only to inspect visible pixels
outside the tag and its quiet zone. The existing conservative measurement erase
rectangle, its 8% padding, and the foreground builder's 12-pixel dilation remain
unchanged.

A connection must be wholly in the extra marker erase band, outside the protected
polygon and inside the tray's safety inset. Evidence is raw colour (HSV saturation
above 45, value above 28) or local luminance at least 12 levels below a 31-pixel
median background. A 3-pixel morphological opening removes isolated noise. There
is no foreground dilation or closing: a real gap must remain a gap.

Exactly two accepted components on a proven visible path are combined into one
item. Only their original masks are united. No recovered bridge pixels or marker
pixels are added to classification or measurement masks. Unrelated items keep
their exact original crops, masks, areas and boxes. Model verification and item
numbering use the corrected set, so overlap evidence is computed after merging.

Three or more components on a path, conflicting pairings, or two unresolved
components running into the protected tag border require repositioning. Missing
or mismatched source geometry when multiple components touch the erase boundary
also requires repositioning. The capture is rejected with zero classified items
and the message: "Move jewellery away from the marker and recapture." The review
marks the affected objects. Confirmation and analysis cannot start on a rejected
capture.

This is a bounded image heuristic. Thin or reflective strands can lack enough
visible evidence; objects physically touching in the inspected band can appear
connected, as elsewhere in the existing connected-component pipeline. It cannot
infer ownership behind the marker. The quiet-zone allowance assumes the current
printed marker; live camera/lighting and changed marker prints need validation.

Restart the backend to load the change. No frontend build or model download is
needed. Saved captures are not rewritten. See
`testing/marker-continuity-20261111/report.md` in the installation root for replay
results and the corrected review image.
