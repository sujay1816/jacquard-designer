"""
Jacquard Designer — Launcher
Double-click this file or run: python run.py
"""
import os, webbrowser, time, threading

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # Prevent joblib/OpenMP from spawning parallel workers.
    # Required on macOS (avoids 10-30s KMeans hang on first detect) and Windows alike.
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    os.environ['OMP_NUM_THREADS']    = '1'
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from app import app, NAV_BUILD, EXPECTED_TEMPLATES, missing_templates

    # Startup self-check.
    #
    # The commonest deployment mistake is copying only the files that changed,
    # which leaves new templates and modules behind. The app then either serves
    # the old interface or fails with an unrelated-looking error much later.
    # Printing the build and the page list here makes a partial copy obvious in
    # the first two seconds instead of after an hour of confusion.
    # Dependency check BEFORE the template check. A missing package is the
    # commoner failure and the more confusing one: the app starts, every page
    # loads, and then one button fails with a Python module name. Reporting it
    # once at startup costs nothing and saves that entire discovery.
    import deps
    dep_result = deps.check()
    dep_report = deps.report(dep_result)

    missing = missing_templates()
    pages = [str(r) for r in app.url_map.iter_rules()
             if not str(r).startswith('/api') and 'static' not in str(r)
             and str(r) != '/static/<path:filename>']

    print("=" * 58)
    print(" JACQUARD DESIGNER")
    print(f" build {NAV_BUILD}")
    print("=" * 58)
    if dep_report:
        print(dep_report)
        print("=" * 58)
    if missing:
        print(" INCOMPLETE INSTALL — these template files are missing:")
        for t in missing:
            print(f"   - templates/{t}")
        print()
        print(" Copy the WHOLE project folder, not just changed files.")
        print("=" * 58)
    else:
        print(f" {len(EXPECTED_TEMPLATES)} templates present · "
              f"{len(pages)} pages:")
        print("   " + "  ".join(sorted(pages)))
        print("=" * 58)
    print(" Starting... please wait")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
