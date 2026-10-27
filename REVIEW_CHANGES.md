# Capture review and surface reflection changes — 24 October 2026

Capture review now draws red A-numbered boxes for accepted non-jewellery
predictions from anomolies.onnx (foreign_obj, plastic_packet, paper). A location
card identifies each object relative to the tray. Boxes retain raw camera
coordinates before the display crop; inference inputs and detection thresholds
are unchanged. Rejected reviews retain the jewellery classification labels.

Stone analysis displays this notice in both the result image and UI for affected
jewellery: "This necklace has a highly reflective surface; possible stones are
present." Other jewellery uses "This jewel". The notice is independent of
confirmed stone detections and does not increase detected stone area or estimated
weight. Existing analysis routing, including disabled ring stone analysis, is
unchanged.

The surface check uses original, pre-normalization jewellery pixels, independent
of glare-removal settings. Highlights need HSV value >=170, saturation <=85,
and either local brightness contrast >=12 or value >=220. Local means exclude
pixels outside the jewellery mask. The notice requires >=12 highlight pixels
and at least 1% surface coverage, plus one of:

- >=5% overall highlight coverage;
- >=2% coverage and >=18% supported local highlight density;
- >=32% supported local density and a connected highlight region covering
  >=0.5% of the jewellery (minimum 6 pixels).

These visual heuristics allow reflective metal to trigger the notice; they
indicate possible stones, not confirmed transparent stones.

Validation:

- Existing stone-area calculator validation passed.
- Six original saved captures replayed through fresh testbed/anomaly inference:
  five clear and the plastic capture rejected. Real classification retained the
  clean confirmation and plastic rejection flows; rejected confirmation stayed
  blocked. See ../testing/review-anomaly-replay.json.
- Original-coordinate boxes verified using real plastic detection; all three
  anomaly classes get boxes and jewellery is excluded.
- Empty, matte, single pixel, isolated pixels, tiny-mask, small hotspot on a
  large jewel, and outside-mask cases do not trigger a notice. Substantial local
  highlights and broad reflections do.
- Four real necklace/earring items from two saved sessions replayed through the
  stone-analysis service, including image-panel and returned UI-data assertions.
- Production frontend TypeScript/Vite build passed.
- No external connections attempted during model replay. Saved session images,
  manifests, and the application database were not changed. Hardware testing
  remains for the manual test.

Previews and details: ../testing/review-changes/.
Repeat: .venv/bin/python scripts/verify_review_changes.py
Restart the application/backend and refresh the browser before manual testing.
Existing saved exports retain their previous appearance; new captures use the
updated rendering and notice.
