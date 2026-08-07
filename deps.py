"""
Dependency and install health.

Two distinct failures live here, and telling them apart is the whole job.

  1. The package is not installed. `pip install X` fixes it.
  2. The package IS installed but cannot load, because it binds to a native
     library pip does not ship. cairosvg on Windows is the standard case:
     `pip install cairosvg` reports "Requirement already satisfied" while
     `import cairosvg` fails, because cairocffi is looking for libcairo-2.dll
     and Windows has no system Cairo.

An earlier version of this file treated both as "missing" and told people to
run pip. For case 2 that advice is not merely unhelpful, it is a loop: pip
confirms the package is there, the app keeps insisting it is absent, and
nothing in either message explains the contradiction.

The second distinction that matters is BLAST RADIUS. cairosvg only draws
generated motifs. A weaver uploading a saree photo to the Generator, converting
it and downloading BMPs never touches it — yet they were shown "This install is
incomplete" across the top of a page that worked perfectly. A warning that
overstates the damage gets ignored, and then the one that matters is ignored
with it.
"""
import importlib
import platform
import sys

# (import name, pip name, what it does, tier)
#   'core'    — nothing works without it
#   'feature' — one part of the app stops working; the rest is fine
REQUIRED = [
    ('flask',    'flask>=2.3.0',         'the web server itself', 'core'),
    ('PIL',      'pillow>=10.0.0',       'reading and writing images', 'core'),
    ('numpy',    'numpy>=1.24.0',        'every image operation', 'core'),
    ('sklearn',  'scikit-learn>=1.3.0',  'colour detection', 'core'),
    ('scipy',    'scipy>=1.11.0',        'linework cleaning and float checks', 'core'),
    ('skimage',  'scikit-image>=0.21.0',
     'Butta Studio, Border Studio and thread reduction', 'core'),
    ('cairosvg', 'cairosvg>=2.7',
     'drawing generated motifs (Assistant designs and the motif library)',
     'feature'),
]

OPTIONAL = [
    ('pillow_heif', 'pillow-heif>=0.15.0', 'HEIC/HEIF uploads from iPhones', 'feature'),
]

# Packages wrapping a native library pip does not install. When one of these
# fails to import, pip is the wrong answer.
NATIVE = {
    'cairosvg': {
        'Windows': [
            'pip installs cairosvg but NOT Cairo itself, which is a Windows',
            'system library. Install the GTK3 runtime, which provides it:',
            '',
            '1. Download the installer from',
            '   github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases',
            '2. Run it and tick "Set up PATH environment variable"',
            '3. Close and reopen your terminal, then restart the app',
            '',
            'If you use conda instead:  conda install -c conda-forge cairo',
        ],
        'Darwin': ['brew install cairo', 'then restart the app'],
        'Linux': ['sudo apt install libcairo2',
                  "(or your distribution's cairo package), then restart"],
    },
}


def _entry(mod, pip_name, purpose, tier, kind, err):
    e = {'module': mod, 'install': pip_name, 'purpose': purpose,
         'tier': tier, 'kind': kind, 'detail': str(err)[:200]}
    e['fix'] = _fix_lines(mod, pip_name, kind)
    e['headline'] = (f'{mod} is not installed' if kind == 'not_installed'
                     else f'{mod} is installed but cannot load')
    return e


def _fix_lines(mod, pip_name, kind):
    if kind == 'not_installed':
        return [f'pip install "{pip_name}"',
                'or: pip install -r requirements.txt',
                'then restart the app']
    info = NATIVE.get(mod)
    if not info:
        return [f'Reinstall it:  pip install --force-reinstall "{pip_name}"']
    return info.get(platform.system()) or info.get('Linux') or []


def _probe(table):
    """
    Try each import and classify the failure.

    ImportError means the package is absent. Anything else — OSError is the
    usual one — means Python found the package and the package could not load
    its own dependency: a different problem with a different fix.
    """
    results = []
    for mod, pip_name, purpose, tier in table:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            # A native binding can raise ImportError too, while naming the
            # library it could not find, so check the text as well as the type.
            text = str(e).lower()
            native = mod in NATIVE and any(
                w in text for w in ('library', 'dll', '.so', 'cairo', 'symbol'))
            results.append(_entry(mod, pip_name, purpose, tier,
                                  'broken' if native else 'not_installed', e))
        except Exception as e:
            results.append(_entry(mod, pip_name, purpose, tier, 'broken', e))
    return results


def check():
    """Install health. Never raises."""
    missing = _probe(REQUIRED)
    core = [m for m in missing if m['tier'] == 'core']
    return {'ok': not missing,
            'core_ok': not core,
            'missing': missing,
            'missing_core': core,
            'missing_features': [m for m in missing if m['tier'] != 'core'],
            'missing_optional': _probe(OPTIONAL),
            'platform': platform.system(),
            'python': sys.version.split()[0]}


def install_command(missing):
    """
    A pip line for the entries pip can actually fix.

    None when every problem is a native-library one — offering a pip command
    there is exactly what sent someone round in a circle.
    """
    fixable = [m for m in missing if m.get('kind') == 'not_installed']
    if not fixable:
        return None
    return 'pip install ' + ' '.join(f'"{m["install"]}"' for m in fixable)


def report(result=None, colour=True):
    """A startup message saying what is wrong, and what still works."""
    result = result or check()
    if result['ok'] and not result['missing_optional']:
        return None

    bold = '\033[1m' if colour else ''
    warn = '\033[33m' if colour else ''
    off = '\033[0m' if colour else ''
    out = []

    if result['missing_core']:
        out.append(f'{warn}{bold}The app cannot run properly.{off}')
    elif result['missing_features'] or result['missing_optional']:
        out.append(f'{warn}One feature is unavailable. Everything else works.{off}')

    for m in result['missing'] + result['missing_optional']:
        out.append('')
        out.append(f"  {bold}{m['headline']}{off}")
        out.append(f"  Needed for: {m['purpose']}")
        if m['kind'] == 'broken':
            # Without this line the person runs pip, is told the package is
            # already there, and has no idea why the app disagrees.
            out.append('  pip will NOT fix this — the package is present; the')
            out.append('  native library it depends on is not.')
        out.append('')
        for line in m['fix']:
            out.append(f'    {line}' if line else '')

    cmd = install_command(result['missing'])
    if cmd and len(result['missing']) > 1:
        out.append('')
        out.append(f'  All at once:  {bold}{cmd}{off}')
    return '\n'.join(out)


def guard(module, feature=''):
    """
    Raise something a person can act on, instead of a bare import error.

    Call at the top of any function whose work needs a lazily imported package.
    """
    try:
        return importlib.import_module(module)
    except Exception as e:
        entry = next((r for r in REQUIRED + OPTIONAL if r[0] == module), None)
        pip_name = entry[1] if entry else module
        what = feature or (entry[2] if entry else 'this feature')
        kind = ('broken' if (module in NATIVE or not isinstance(e, ImportError))
                else 'not_installed')
        fix = ' '.join(x for x in _fix_lines(module, pip_name, kind) if x)
        raise RuntimeError(
            f'{what} is unavailable: {module} could not be loaded. {fix}')
