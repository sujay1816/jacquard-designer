"""
Dependency check.

The failure this exists to prevent: a weaver uploads a motif to Butta Studio and
gets "Preview failed: No module named 'skimage'". The package is in
requirements.txt — the install simply did not finish, or finished before that
line was added. But because the import sits inside the function that does the
reduction, nothing complains until someone clicks the button, and then the
message is a Python module name rather than an instruction.

Three things follow from that:

  * The check runs at STARTUP, so an incomplete install is reported once,
    clearly, before anyone uses the app — not per-page, per-click, forever.
  * Every message names the pip command that fixes it. "No module named
    skimage" is not actionable; "pip install scikit-image" is, and the two
    strings are not even the same word.
  * Pages that need a missing package say which feature is unavailable, rather
    than failing at the moment of use with an internal error.

Import names and install names differ often enough (skimage/scikit-image,
sklearn/scikit-learn, PIL/pillow, cv2/opencv-python) that mapping them is the
whole point of the table below.
"""
import importlib

# (import name, pip name, what stops working without it)
REQUIRED = [
    ('flask',   'flask>=2.3.0',         'the web server itself'),
    ('PIL',     'pillow>=10.0.0',       'reading and writing images'),
    ('numpy',   'numpy>=1.24.0',        'every image operation'),
    ('sklearn', 'scikit-learn>=1.3.0',  'colour detection'),
    ('scipy',   'scipy>=1.11.0',        'linework cleaning and float checks'),
    ('skimage', 'scikit-image>=0.21.0', 'Butta Studio, Border Studio and thread reduction'),
    ('cairosvg', 'cairosvg>=2.7',       'generating motifs in the Assistant and Generator'),
]

OPTIONAL = [
    ('pillow_heif', 'pillow-heif>=0.15.0', 'HEIC/HEIF uploads from iPhones'),
]


def _probe(table):
    missing = []
    for mod, pip_name, purpose in table:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append({'module': mod, 'install': pip_name, 'purpose': purpose})
    return missing


def check():
    """Returns {'ok', 'missing', 'missing_optional'} without raising."""
    missing = _probe(REQUIRED)
    return {'ok': not missing,
            'missing': missing,
            'missing_optional': _probe(OPTIONAL)}


def install_command(missing):
    """The single pip line that fixes everything in `missing`."""
    return 'pip install ' + ' '.join(f'"{m["install"]}"' for m in missing)


def report(result=None, colour=True):
    """A startup message an operator can act on without reading the source."""
    result = result or check()
    if result['ok'] and not result['missing_optional']:
        return None

    bold = '\033[1m' if colour else ''
    warn = '\033[33m' if colour else ''
    off = '\033[0m' if colour else ''
    lines = []

    if result['missing']:
        lines.append(f'{warn}{bold}Some required packages are missing.{off}')
        lines.append('')
        for m in result['missing']:
            lines.append(f"  {m['module']:<10} needed for {m['purpose']}")
        lines.append('')
        lines.append(f"  Fix:  {bold}{install_command(result['missing'])}{off}")
        lines.append('  Or:   pip install -r requirements.txt')
        lines.append('')
        lines.append('  The app will still start, but the features above will fail')
        lines.append('  when you use them.')

    for m in result['missing_optional']:
        lines.append(f"  Optional: {m['module']} is not installed — "
                     f"{m['purpose']} will not work.")
        lines.append(f"            pip install {m['install']}")

    return '\n'.join(lines)


def guard(module, feature=''):
    """
    Raise a message a person can act on, instead of a bare ModuleNotFoundError.

    Call at the top of any function whose work needs an optional-at-import-time
    package. The raw exception names the import ('skimage'); this names the
    thing to install ('scikit-image'), which is a different word.
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        entry = next((r for r in REQUIRED + OPTIONAL if r[0] == module), None)
        pip_name = entry[1] if entry else module
        what = feature or (entry[2] if entry else 'this feature')
        raise RuntimeError(
            f'{what} needs a package that is not installed. '
            f'Run:  pip install "{pip_name}"   '
            f'(or: pip install -r requirements.txt), then restart the app.')
