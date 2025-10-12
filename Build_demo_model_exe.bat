@echo off
setlocal
REM Launch the Umbra model viewer desktop utility (demo executable)
python -m umbra.tools.model_viewer %*
endlocal
