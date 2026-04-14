Set WshShell = CreateObject("WScript.Shell")
' 仮想環境のpythonw.exeと、実行したいapp.pyのパスを指定します
WshShell.Run "C:\Users\tesiy\PycharmProjects\tmp\whisper\venv\Scripts\pythonw.exe C:\Users\tesiy\PycharmProjects\tmp\whisper\app.py", 0
Set WshShell = Nothing