"""
Jacquard Designer — Launcher
Double-click this file or run: python run.py
"""
import subprocess, sys, os, webbrowser, time, threading

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("="*50)
    print(" JACQUARD DESIGNER")
    print(" Starting... please wait")
    print("="*50)
    threading.Thread(target=open_browser, daemon=True).start()
    from app import app
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
