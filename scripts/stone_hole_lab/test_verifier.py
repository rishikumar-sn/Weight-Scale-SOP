from copy import deepcopy

from verifier import evaluate


def candidates():
    return [dict(color='White/Colorless', bbox=[x, 20, 18, 12],
                 refinement_diagnostics={'metal_surround_ratio': .9},
                 metrics=dict(valid=True, core_pixels=80, local_background_pixels=300,
                              local_delta_e_median=4, local_delta_e_p90=6,
                              raw_lightness_std=2, reference_background_p90=1,
                              reference_delta_e_median=4, reference_delta_e_p90=6))
            for x in (20, 45, 70)]


def test_matching_repeated_link_holes_rejected():
    assert all(d['decision'] == 'reject_background_hole'
               for d in evaluate(candidates(), {'valid': True}))


def test_missing_or_misaligned_reference_cannot_reject():
    assert all(d['decision'] == 'uncertain'
               for d in evaluate(candidates(), {'valid': False}))


def test_gold_enclosure_alone_cannot_reject_real_white_surfaces():
    rows = candidates()
    for r in rows:
        r['metrics']['local_delta_e_median'] = 20
        r['metrics']['reference_delta_e_median'] = 20
    assert all(d['decision'] == 'keep' for d in evaluate(rows, {'valid': True}))


def test_colored_stones_preserved_even_if_diagnostic_scores_match():
    rows = candidates()
    for r in rows:
        r['color'] = 'Green'
    assert all(d['decision'] == 'keep' for d in evaluate(rows, {'valid': True}))


def test_no_group_reinforcement_from_unmatched_regions():
    rows = candidates()
    rows[1]['metrics']['reference_delta_e_p90'] = 25
    rows[2]['metrics']['reference_delta_e_p90'] = 25
    assert not any(d['decision'] == 'reject_background_hole' for d in evaluate(rows, {'valid': True}))


def test_bad_photometric_fit_or_texture_retains_regions_and_input():
    for key, value in [('reference_background_p90', 10), ('raw_lightness_std', 9)]:
        rows = candidates()
        for row in rows:
            row['metrics'][key] = value
        before = deepcopy(rows)
        assert not any(d['decision'] == 'reject_background_hole' for d in evaluate(rows, {'valid': True}))
        assert rows == before


def test_separate_jewels_cannot_reinforce_each_other():
    rows = candidates()
    for i, row in enumerate(rows):
        row['jewel_id'] = i + 1
    assert not any(d['decision'] == 'reject_background_hole'
                   for d in evaluate(rows, {'valid': True}))


def test_learned_stones_are_preserved_despite_background_like_scores():
    rows = candidates()
    for row in rows:
        row['learned_matches'] = ['user-confirmed-white']
    assert not any(d['decision'] == 'reject_background_hole'
                   for d in evaluate(rows, {'valid': True}))
