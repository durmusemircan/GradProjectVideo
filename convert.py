from PyQt5 import uic

with open("video_app.py", "w", encoding="utf-8") as fout:
    uic.compileUi("app.ui", fout)