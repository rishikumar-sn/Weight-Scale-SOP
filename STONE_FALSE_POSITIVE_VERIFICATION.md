# Stone false-positive verification — 14 November 2026

Stone analysis now verifies final regions against unnormalized raw camera pixels
after growth and instance separation, before rendering, area and weight features.
The working image, candidate order, RNG state, accepted boundaries and jewellery
area denominator are preserved. The verifier removes whole false regions; it
does not expand masks or recover missed faces.

The service supplies the separate raw item crop for platform sampling. Legacy
sessions without that crop use the existing aligned full-capture color source.
`analyze_image_bgr` accepts an optional aligned `verification_image_bgr` and falls
back to `original_color_bgr`, then the existing crop. Pure-white composited
outside pixels are excluded from the platform reference. Without enough real
platform samples, platform-gap rejection is disabled.

Final checks cover dim clipped warm-metal recesses, uniform neutral platform
gaps, unsupported neutral hardware and large muted structures whose mixed color
vote contains only black/white. Bright/varied facets, small mounted white faces,
glossy black facets, coherent colored faces and explicit learned matches are
protected by the evidence checks. These are conservative image heuristics, not
a material-identification model or guaranteed accuracy on unseen jewellery.

Every rejected region records its reason, area and raw evidence under
`jewels[].stone_analysis_v2.material_verification`. Earlier neutral/gold color
gates also record their reasons. Debug masks now reflect the final verified
regions, including empty results.

Thread classification, segmentation, projection and exclusion are unchanged.
The change does not claim to fix cord pixels missed by the thread model.

Evidence, frozen baseline code, paired replays, protected-mask checks and
limitations: `../testing/stone-false-positives-20261114/report.md`.

New analyses use this behavior after the backend loads the changed code. Saved
historical reports and captures are not rewritten.
