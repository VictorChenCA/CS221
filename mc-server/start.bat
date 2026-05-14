@echo off
setlocal
cd /d "%~dp0"
if not defined JAVA_HOME (
  for /f "delims=" %%i in ('where /r "C:\Program Files\Eclipse Adoptium" java.exe 2^>nul') do (
    set "JAVA_HOME=%%~dpi.."
    goto :found
  )
)
:found
"%JAVA_HOME%\bin\java.exe" -Xmx6G -Xms2G -jar paper.jar nogui
endlocal
