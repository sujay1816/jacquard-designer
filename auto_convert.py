"""
Auto-convert — generate, score, and retry until the result is defensible.

The gap this closes is not intelligence, it is FEEDBACK. Every failure this
product has shipped looked identical from the inside: a clean, valid, 1-bit
BMP that verify_bmp passed and _design_warnings had nothing to say about,
whose design had nevertheless been ruined. The app never compared its output
to the input, so it never noticed, and a person had to spot it by eye.

Reviewing real failures, the app almost always had the right answer available
and simply never tried it:

  * a butta at 200 pins closed every gap; 340 pins was fine
  * a pencil scan lost 71% accuracy on the colour path; the line-art path was
    already implemented
  * motifs thickened under reduction at coverage 0.16; 0.28 was better
  * automatic fabric enhancement fragmented designs 4.7x; not enhancing was
    better

Each of those is a parameter that already exists. So this module tries them,
scores every candidate with fidelity_report, and keeps the winner — turning a
silent wrong answer into either a right answer or an honest refusal.

Deterministic by construction: the candidate list is fixed and ordered, and
scoring is arithmetic, so the same image and constraints always select the
same settings. No model is involved.
"""
import numpy as np

from fidelity import fidelity_report
from vision_engine import detect_colors_smart

# Verdict ranking used to compare candidates.
_VERDICT_RANK = {'ok': 0, 'warn': 1, 'fail': 2}

# Pin counts offered when the caller has not fixed one. Ordered low to high so
# ties resolve toward the cheaper loom setup.
CANDIDATE_PINS = (240, 360, 480, 600, 720, 960, 1200)


def _score(report):
    """
    Lower is better. Verdict dominates, then structural error, then ink error.

    Structure is weighted above coverage deliberately: lighter linework with
    the right topology still reads as the design, while heavy linework with
    closed gaps does not, however well its ink matches.
    """
    s_white = max(report.get('source_white_regions') or 1, 1)
    t_white = report.get('output_white_regions') or 0
    ratio = t_white / s_white
    structure_err = abs(np.log(max(ratio, 1e-3)))          # symmetric about 1.0
    ink_err = abs(report.get('ink_drift_pct', 0)) / 100.0
    isolated = min(report.get('isolated_cells', 0) / 500.0, 1.0)
    return (_VERDICT_RANK.get(report.get('verdict', 'fail'), 2),
            round(2.0 * structure_err + ink_err + isolated, 4))


def _candidate_settings():
    """
    Settings to try, in preference order.

    Each entry is a kwargs dict for detect_colors_smart. The first is the
    current default; the rest are the alternatives that have historically been
    the right answer when the default was wrong.
    """
    return [
        {},                                    # defaults (line-art auto-detect)
        {'lineart': False},                    # force the colour path
        {'thin_rescue': False},                # no thin-feature rescue
        {'despeckle': 0},                      # keep every speck
        {'lineart': False, 'superpixels': True},
    ]


def auto_convert(image, pins=None, n_colors=2, cards=None, max_candidates=16):
    """
    Convert an image, checking the result and retrying when it is poor.

    image  : PIL image as uploaded
    pins   : required pin count for this job, or None to search
    n_colors : number of colours to detect

    Returns a decision record:
        best        : {'pins', 'cards', 'settings', 'label_map', 'report', 'score'}
        alternatives: other candidates, best first, without label maps
        attempts    : how many were evaluated
        verdict     : the winning report's verdict
        summary     : one line for a person
        advice      : what to do when the winner is still not good

    The label map is returned rather than a BMP so the caller keeps control of
    shuttle assignment and weave, which are craft decisions this must not make.
    """
    def evaluate(p, c, settings):
        try:
            _, _, lm, _ = detect_colors_smart(image, n_colors, p, c, **settings)
            report = fidelity_report(image, np.asarray(lm) > 0)
        except Exception:
            return None
        return {'pins': p, 'cards': c, 'settings': settings, 'label_map': lm,
                'report': report, 'score': _score(report)}

    def cards_for(p):
        return int(cards) if cards else int(round(
            p * image.size[1] / max(image.size[0], 1)))

    results, tried = [], 0

    # Two phases rather than a full grid. A full sweep of every pin count
    # against every setting was both slow (22s) and truncated by the candidate
    # budget, which biased selection toward the low pin counts evaluated first
    # — it picked 360 pins for a design that converts perfectly at 859.
    #
    # Phase 1 finds the best SETTINGS at a mid pin count; phase 2 sweeps pin
    # counts using them. Settings and resolution are close to independent here,
    # so this costs a dozen evaluations instead of thirty-five and never runs
    # out of budget before reaching the high end.
    probe_pins = int(pins) if pins else 480
    probe_cards = cards_for(probe_pins)
    for settings in _candidate_settings():
        if tried >= max_candidates:
            break
        tried += 1
        r = evaluate(probe_pins, probe_cards, settings)
        if r:
            results.append(r)

    if not results:
        return {'best': None, 'alternatives': [], 'attempts': tried,
                'verdict': 'fail', 'summary': 'Conversion failed on every setting.',
                'advice': ['The image could not be read. Try a different file.']}

    best_settings = min(results, key=lambda r: r['score'])['settings']

    if not pins:
        for p in CANDIDATE_PINS:
            if p == probe_pins or tried >= max_candidates:
                continue
            tried += 1
            r = evaluate(p, cards_for(p), best_settings)
            if r:
                results.append(r)

    if not results:
        return {'best': None, 'alternatives': [], 'attempts': tried,
                'verdict': 'fail', 'summary': 'Conversion failed on every setting.',
                'advice': ['The image could not be read. Try a different file.']}

    results.sort(key=lambda r: r['score'])
    best = results[0]
    rep = best['report']

    alts = []
    for r in results[1:]:
        if any(a['pins'] == r['pins'] for a in alts):
            continue                                   # one entry per pin count
        alts.append({'pins': r['pins'], 'cards': r['cards'],
                     'settings': r['settings'], 'verdict': r['report']['verdict'],
                     'ink_drift_pct': r['report']['ink_drift_pct'],
                     'gaps': r['report']['output_white_regions'],
                     'score': r['score']})
        if len(alts) >= 4:
            break

    # Fidelity measures FAITHFULNESS, not ADEQUACY. Upscaling a tiny source
    # reproduces it perfectly and scores OK, while carrying almost no design:
    # a 90x60 thumbnail rendered at 600 pins is a faithful copy of nothing
    # much. Source adequacy is a separate question and has to be asked
    # separately.
    verdict = rep['verdict']
    source_note = None
    try:
        from loom_utils import source_resolution_check
        chk = source_resolution_check(image, best['pins'])
        tps = chk.get('threads_per_stroke')
        if tps is None:
            # No measurable strokes — usually a source so small its ink has
            # dissolved into grey. Fall back to raw resolution: asking for more
            # than twice the source width cannot add design that is not there.
            tps = (image.size[0] / max(best['pins'], 1)) * 2.0
        if tps < 1.0:
            source_note = (
                f"The source is only {image.size[0]}px wide, so its strokes are "
                f"thinner than one thread at {best['pins']} pins. The output is "
                f"faithful to the file but the file itself carries little "
                f"detail — a higher-resolution source is needed, not different "
                f"settings.")
            if verdict == 'ok':
                verdict = 'warn'
    except Exception:
        pass

    summary = (f"{best['pins']} x {best['cards']}: {verdict.upper()}, "
               f"{rep['ink_drift_pct']:+.0f}% thread, "
               f"{rep['output_white_regions']} gaps against "
               f"{rep['source_white_regions']} in the source.")

    advice = list(rep.get('messages', []))
    if source_note:
        advice.insert(0, source_note)
    if verdict != 'ok':
        better = [a for a in alts if _VERDICT_RANK[a['verdict']] < _VERDICT_RANK[verdict]]
        if better:
            b = better[0]
            advice.append(
                f"{b['pins']} pins would convert cleanly if the job allows it.")
        elif pins:
            advice.append(
                f"No setting converts this image cleanly at {pins} pins. A "
                f"higher-resolution source is the only real fix.")

    return {'best': best, 'alternatives': alts, 'attempts': tried,
            'verdict': verdict, 'summary': summary, 'advice': advice}
