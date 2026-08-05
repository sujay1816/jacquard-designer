"""
Tests for the assistant validator — the trust boundary between model output
and the loom.

Runs entirely offline. No API key, no network. Every case here is a thing a
model could plausibly propose that must not reach the generator.

Run:  python tools/test_assistant.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant_engine import (validate_patch, advisories, interpret_response,
                              interpret_analysis)

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  pass  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


BASE = {'shuttle_count': 2, 'detected_colors': 3, 'pins': 360, 'cards': 360}


def main():
    print("\nShuttle budget — the constraint that matters most")

    # The headline case: weaver asks for a colour the loom cannot weave.
    r = validate_patch({'color_assignments': {
        '0': 'background', '1': 'zari', '2': 'meena1', '3': 'meena2'}},
        {'shuttle_count': 2, 'detected_colors': 4})
    check("3 threads on a 2-shuttle loom is refused",
          'color_assignments' not in r['patch'], r)
    check("refusal explains the shuttle budget",
          any('shuttle' in m.lower() for m in r['rejected']), r['rejected'])

    r = validate_patch({'color_assignments': {
        '0': 'background', '1': 'zari', '2': 'meena1'}},
        {'shuttle_count': 2, 'detected_colors': 3})
    check("2 threads on a 2-shuttle loom is allowed",
          r['patch'].get('color_assignments') and not r['rejected'], r)

    r = validate_patch({'color_assignments': {'0': 'background', '1': 'zari'}},
                       {'shuttle_count': 1, 'detected_colors': 2})
    check("background does not consume a shuttle",
          'color_assignments' in r['patch'], r)

    # Raising shuttle_count in the same patch must be honoured before the
    # budget check, or "use 3 shuttles and add green" would wrongly fail.
    r = validate_patch({'shuttle_count': 3, 'color_assignments': {
        '0': 'background', '1': 'zari', '2': 'meena1', '3': 'meena2'}},
        {'shuttle_count': 2, 'detected_colors': 4})
    check("raising shuttle count in the same turn is honoured",
          r['patch'].get('shuttle_count') == 3
          and 'color_assignments' in r['patch'], r)

    print("\nHallucinated and malformed input")

    r = validate_patch({'color_assignments': {'7': 'zari'}}, BASE)
    check("undetected colour index is refused",
          'color_assignments' not in r['patch'], r)

    r = validate_patch({'color_assignments': {'1': 'silk'}}, BASE)
    check("invented shuttle name is refused",
          'color_assignments' not in r['patch'], r)

    r = validate_patch({'magic_mode': True, 'pins': 400}, BASE)
    check("unknown field is dropped",
          'magic_mode' not in r['patch'] and r['patch'].get('pins') == 400, r)

    r = validate_patch({'satin_settings': 'lots'}, BASE)
    check("malformed weave settings rejected, not crashed",
          'satin_settings' not in r['patch'], r)

    check("non-dict patch does not crash",
          validate_patch(None, BASE)['patch'] == {})
    check("missing state does not crash",
          isinstance(validate_patch({'pins': 400}, None)['patch'], dict))

    print("\nRange clamping")

    r = validate_patch({'pins': 99999}, BASE)
    check("absurd pin count is clamped", r['patch']['pins'] == 2640, r)
    check("clamping is reported to the weaver", bool(r['rejected']), r)

    r = validate_patch({'shuttle_count': 9}, BASE)
    check("shuttle count clamped to 4", r['patch']['shuttle_count'] == 4, r)

    r = validate_patch({'satin_settings': {'zari': {'n': 99}}}, BASE)
    check("satin count clamped to 16",
          r['patch']['satin_settings']['zari']['n'] == 16, r)

    r = validate_patch({'stroke_thickness': 0}, BASE)
    check("outline thickness clamped up to 1",
          r['patch']['stroke_thickness'] == 1, r)

    r = validate_patch({'rani_weave': 'velvet'}, BASE)
    check("invalid rani weave refused", 'rani_weave' not in r['patch'], r)

    r = validate_patch({'rani_weave': 'TWILL'}, BASE)
    check("rani weave is case-insensitive",
          r['patch'].get('rani_weave') == 'twill', r)

    print("\nAdvisories are warnings, not blocks")

    p = validate_patch({'satin_settings': {'zari': {'n': 14}}}, BASE)['patch']
    notes = advisories(p, BASE)
    check("long float earns a warning", any('float' in n for n in notes), notes)
    check("but the change still applies",
          p['satin_settings']['zari']['n'] == 14, p)

    print("\nResponse parsing")

    fake = {'content': [
        {'type': 'text', 'text': 'Made the gold finer.'},
        {'type': 'tool_use', 'name': 'update_settings',
         'input': {'satin_settings': {'zari': {'n': 5}},
                   'explanation': 'Satin 5 on zari.'}},
    ]}
    out = interpret_response(fake, BASE)
    check("tool call is extracted and validated",
          out['patch']['satin_settings']['zari']['n'] == 5, out)
    check("prose reply is preserved", 'gold' in out['reply'], out)

    out = interpret_response({'content': [
        {'type': 'text', 'text': 'That needs a spare shuttle.'}]}, BASE)
    check("text-only reply yields an empty patch",
          out['patch'] == {} and out['ok'], out)

    check("empty response does not crash",
          interpret_response({}, BASE)['patch'] == {})

    print("\nDesign analysis — colour grouping")

    C4 = [(240, 220, 210), (200, 140, 150), (150, 70, 90), (60, 20, 50)]

    def analysis(assignments, **extra):
        inp = {'assignments': assignments, 'explanation': 'x'}
        inp.update(extra)
        return {'content': [{'type': 'tool_use', 'name': 'group_colours', 'input': inp}]}

    # The whole point: tonal shades merged onto one shuttle.
    r = interpret_analysis(analysis(
        {'0': 'zari', '1': 'zari', '2': 'meena1', '3': 'background'}), C4, 2)
    check("tonal shades may share a shuttle",
          r['assignments'].get('0') == 'zari' and r['assignments'].get('1') == 'zari', r)
    check("no rejections for a valid grouping", not r['rejected'], r['rejected'])

    # Over-budget must fail even though every colour is assigned.
    r = interpret_analysis(analysis(
        {'0': 'zari', '1': 'meena1', '2': 'meena2', '3': 'background'}), C4, 2)
    check("three threads on a two-shuttle loom is refused",
          r['assignments'] == {}, r)

    # A dropped colour would silently vanish from the design.
    r = interpret_analysis(analysis({'0': 'zari', '1': 'zari', '3': 'background'}), C4, 2)
    check("incomplete grouping is refused", r['assignments'] == {}, r)
    check("missing colour is named",
          any('2' in m for m in r['rejected']), r['rejected'])

    # The ground must be unambiguous.
    r = interpret_analysis(analysis(
        {'0': 'background', '1': 'zari', '2': 'zari', '3': 'background'}), C4, 2)
    check("two backgrounds refused", r['assignments'] == {}, r)
    r = interpret_analysis(analysis(
        {'0': 'zari', '1': 'zari', '2': 'zari', '3': 'meena1'}), C4, 2)
    check("no background refused", r['assignments'] == {}, r)

    # Hallucinated index beyond what was detected.
    r = interpret_analysis(analysis(
        {'0': 'zari', '1': 'zari', '2': 'meena1', '3': 'background', '9': 'meena2'}), C4, 3)
    check("undetected colour index refused", r['assignments'] == {}, r)

    # Confidence is surfaced so a weaver knows when to check.
    r = interpret_analysis(analysis(
        {'0': 'zari', '1': 'zari', '2': 'meena1', '3': 'background'},
        confidence='low'), C4, 2)
    check("low confidence is passed through", r['confidence'] == 'low', r)

    check("empty analysis response does not crash",
          interpret_analysis({}, C4, 2)['assignments'] == {})

    print(f"\n{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
