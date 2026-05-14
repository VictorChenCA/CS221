@echo off
setlocal
cd /d "%~dp0\.."
set "N=%~1"
if "%N%"=="" set "N=10"
node bot\spawn.js %N%
endlocal
