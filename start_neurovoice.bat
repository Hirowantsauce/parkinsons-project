@echo off
cd /d D:\AG\Parkinsons\Codes
echo Starting NeuroVoice server... wait for the "Web Server Running" message,
echo then open http://127.0.0.1:8080 in your browser.
python server.py
pause
