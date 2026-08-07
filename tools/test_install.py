"""
Install-health tests: every page, every route, every dependency.

This suite exists because of a failure the whole test corpus missed. A weaver
uploaded a motif to Butta Studio and got:

    Preview failed: No module named 'skimage'

No test caught it, and none could have: every suite here runs in an environment
where the package is installed. The tests verified the code, never the install.

So this checks the things that only break on someone else's machine — that the
dependency list matches what is imported, that a missing package is reported at
startup rather than at the moment of use, and that when it does surface it
names the pip command rather than a Python module.

Run:  python tools/test_install.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app                                     # noqa: E402
import deps                                                 # noqa: E402

PASS = FAIL = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  pass  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def main():
    c = flask_app.app.test_client()

    print('\nThe dependency list matches what the code imports')
    reqs = open(os.path.join(ROOT, 'requirements.txt'), encoding='utf-8').read().lower()
    for mod, pip_name, _ in deps.REQUIRED:
        base = re.split(r'[><=]', pip_name)[0].strip().lower()
        check(f'{mod} is declared in requirements.txt', base in reqs, pip_name)

    # skimage/scikit-image, sklearn/scikit-learn, PIL/pillow — the import name
    # and the install name differ, which is exactly why "No module named
    # 'skimage'" is not an instruction anyone can follow.
    check('the table maps import names to install names',
          dict((m, p) for m, p, _ in deps.REQUIRED)['skimage'].startswith('scikit-image'))
    check('every required package has a stated purpose',
          all(purpose for _, _, purpose in deps.REQUIRED))

    print('\nEvery module the app imports is covered')
    declared = {m for m, _, _ in deps.REQUIRED} | {m for m, _, _ in deps.OPTIONAL}
    stdlib_ok = {'io', 'os', 're', 'sys', 'json', 'math', 'time', 'uuid', 'zlib',
                 'base64', 'struct', 'pickle', 'queue', 'zipfile', 'hashlib',
                 'inspect', 'colorsys', 'threading', 'urllib', 'importlib',
                 'abc', 'dataclasses', 'typing', 'functools', 'collections',
                 'itertools', 'random', 'shutil', 'tempfile', 'traceback',
                 'warnings', 'webbrowser', 'subprocess', 'datetime', 'copy',
                 'string', 'glob', 'textwrap', 'unicodedata', 'secrets'}
    local = {f[:-3] for f in os.listdir(ROOT) if f.endswith('.py')} | {'llm', 'templates'}
    import ast
    found, uncovered = set(), set()
    for fn in os.listdir(ROOT):
        if not fn.endswith('.py'):
            continue
        try:
            tree = ast.parse(open(os.path.join(ROOT, fn), encoding='utf-8').read())
        except SyntaxError:
            continue
        # Parsed, not pattern-matched: a regex over lines also matches prose
        # inside a docstring — "from under a second to nine" was read as an
        # import of a package called `under`.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                found.add(node.module.split('.')[0])
    for mod in found:
        if mod in declared or mod in stdlib_ok or mod in local:
            continue
        uncovered.add(mod)
    check('no third-party import is missing from the dependency table',
          not uncovered, sorted(uncovered))

    print('\nA missing package is reported, not discovered')
    absent = {'ok': False,
              'missing': [{'module': 'skimage', 'install': 'scikit-image>=0.21.0',
                           'purpose': 'Butta Studio'}],
              'missing_optional': []}
    text = deps.report(absent, colour=False)
    check('the startup report names the package', 'skimage' in text, text)
    check('and the command that installs it',
          'pip install "scikit-image>=0.21.0"' in text, text)
    check('and what stops working without it', 'Butta Studio' in text, text)
    healthy = {'ok': True, 'missing': [], 'missing_optional': []}
    check('a complete install prints nothing at startup',
          deps.report(healthy) is None, deps.report(healthy))
    # An optional package missing is worth one line, not a warning block.
    opt = {'ok': True, 'missing': [],
           'missing_optional': [{'module': 'pillow_heif', 'install': 'pillow-heif',
                                 'purpose': 'HEIC uploads'}]}
    check('a missing optional package is mentioned, not alarming about',
          'Optional' in deps.report(opt, colour=False)
          and 'required packages are missing' not in deps.report(opt, colour=False))

    print('\nWhen it does surface, it is an instruction')
    # The message the weaver actually saw. It is true and useless: the thing to
    # install is called scikit-image, which is a different word.
    msg = flask_app._dep_message("Preview failed: No module named 'skimage'")
    check('the raw module error is gone', "No module named" not in msg, msg)
    check('the pip name is given', 'scikit-image' in msg, msg)
    check('and a runnable command', 'pip install' in msg, msg)
    check('and a restart instruction', 'restart' in msg.lower(), msg)
    check('the original context is kept', 'Preview failed' in msg, msg)
    check('a submodule resolves to its parent package',
          'scikit-image' in flask_app._dep_message("No module named 'skimage.measure'"))
    check('an unknown module still yields something useful',
          'pip install' in flask_app._dep_message("No module named 'weirdlib'"))
    # Ordinary errors must pass through untouched.
    check('an ordinary error is left alone',
          flask_app._dep_message('Image is too large (max 50 MB).')
          == 'Image is too large (max 50 MB).')

    print('\nThe health endpoint')
    h = c.get('/api/health').get_json()
    check('reports install health', h['success'] and 'ok' in h, h)
    check('lists missing packages', isinstance(h['missing'], list), h)
    check('lists missing page files', isinstance(h['templates_missing'], list), h)
    check('this install is complete', h['ok'] and not h['templates_missing'], h)

    print('\nEvery page carries the warning banner')
    # One copy in the shared nav — a per-page copy is how border.html ends up
    # with the check and butta.html without it.
    for page in ('/', '/butta', '/border', '/edit', '/agent', '/trace'):
        body = c.get(page).get_data(as_text=True)
        check(f'{page} checks its install', 'jqDepBanner' in body and '/api/health' in body)

    print('\nNo page renders an error')
    for page in ('/', '/butta', '/border', '/edit', '/agent', '/trace'):
        r = c.get(page)
        body = r.get_data(as_text=True)
        leaks = [w for w in ('Traceback', 'jinja2.exceptions', 'UndefinedError',
                             'TemplateNotFound', 'Internal Server Error')
                 if w in body]
        check(f'{page} renders cleanly', r.status_code == 200 and not leaks,
              (r.status_code, leaks))

    print('\nNo API route 500s or returns HTML on bad input')
    rules = [r for r in flask_app.app.url_map.iter_rules()
             if str(r).startswith('/api') and '<' not in str(r)]
    problems = []
    for rule in rules:
        for m in sorted(rule.methods - {'HEAD', 'OPTIONS'}):
            resp = (c.get(str(rule)) if m == 'GET'
                    else c.post(str(rule), data={},
                                content_type='multipart/form-data'))
            ct = resp.headers.get('Content-Type', '')
            if resp.status_code >= 500:
                problems.append((m, str(rule), resp.status_code))
            elif 'json' not in ct and 'event-stream' not in ct:
                problems.append((m, str(rule), ct))
    check(f'all {len(rules)} api routes answer cleanly', not problems, problems[:5])

    print('\nReal messages reach the person')
    r = c.post('/api/detect-colors', data={}, content_type='multipart/form-data')
    check('a route explains what was wrong', r.get_json().get('error'), r.get_json())
    home = c.get('/').get_data(as_text=True)
    # The page threw the body away and showed "Server error 400: BAD REQUEST",
    # which would have hidden the pip instruction too.
    check('the page reads the body before the status code',
          'the only part the person can act on' in home)

    print(f'\n{PASS} passed, {FAIL} failed\n')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
