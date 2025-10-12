@echo off
setlocal

REM Launch the Umbra model viewer desktop utility
python -m umbra.tools.model_viewer %*

endlocal
