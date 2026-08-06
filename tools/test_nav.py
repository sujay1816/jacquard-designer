"""
Navigation consistency tests.

The menu previously drifted: five templates each carried their own copy with a
different class system, edit.html had no nav element and omitted its own link,
and border_id.html highlighted the wrong item. These tests fail if that
happens again.

Run:  python tools/test_nav.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app

PAGES = {
    '/': 'Generator',
    '/butta': 'Butta',
    '/border': 'Border',
    '/edit': 'BMP Editor',
    '/agent': 'Assistant',
    '/trace': 'Tracing Guide',
}
REMOVED = ['/border-id']

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  pass  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def links_in(html):
    return re.findall(r'<a href="(/[a-z]*)" *class="jq-nav-link', html)


def active_in(html):
    return re.findall(r'class="jq-nav-link is-active"[^>]*>([^<]+)<', html)


def main():
    c = flask_app.app.test_client()
    rendered = {}

    print("\nEvery page renders")
    for path in PAGES:
        r = c.get(path)
        rendered[path] = r.data.decode('utf-8', 'ignore')
        check(f"{path} returns 200", r.status_code == 200, r.status_code)

    print("\nIdentical menu everywhere")
    sets = {p: tuple(links_in(h)) for p, h in rendered.items()}
    check("all pages expose the same links", len(set(sets.values())) == 1, sets)
    check("menu is exactly the five live pages",
          set(sets['/']) == set(PAGES), sets['/'])

    print("\nActive state matches the route")
    for path, label in PAGES.items():
        act = active_in(rendered[path])
        check(f"{path} highlights {label}",
              len(act) == 1 and label in act[0], act)

    print("\nSingle source of styling")
    for path, html in rendered.items():
        stale = re.findall(r'class="(?:nav-link|nav-btn|topnav|hbtn nav)"', html)
        check(f"{path} has no legacy nav classes", not stale, stale)
    nav_partial = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'templates', '_nav.html')
    check("shared partial exists", os.path.exists(nav_partial))
    for path in PAGES:
        src = os.path.join(os.path.dirname(nav_partial),
                           ('index' if path == '/' else path.strip('/')) + '.html')
        body = open(src, encoding='utf-8').read()
        check(f"{os.path.basename(src)} includes the partial",
              "include '_nav.html'" in body)

    print("\nRemoved pages stay removed")
    for path in REMOVED:
        check(f"{path} is gone", c.get(path).status_code == 404,
              c.get(path).status_code)
        for p, html in rendered.items():
            check(f"{p} does not link to {path}", f'href="{path}"' not in html)

    print("\nBorder ID capability survived the merge")
    rules = {str(r) for r in flask_app.app.url_map.iter_rules()}
    check("fine-detail API is retained", '/api/border-id-generate' in rules)
    border = rendered['/border']
    check("Border Studio calls it for fine mode",
          '/api/border-id-generate' in border)
    check("Border Studio exposes the mode switcher", 'setMode(' in border)

    print(f"\n{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
