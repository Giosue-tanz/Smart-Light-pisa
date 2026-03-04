@echo off
echo Avvio di OSM Web Wizard...
echo Assicurati di avere una connessione internet attiva.

:: Imposta percorsi relativi basati sulla posizione di questo script
set BASE_DIR=%~dp0
set SUMO_HOME=%BASE_DIR%sumo_tools\sumo-1.22.0

:: Verifica che SUMO esista
if not exist "%SUMO_HOME%" (
    echo ERRORE: Non trovo la cartella di SUMO in: %SUMO_HOME%
    pause
    exit /b
)

:: Aggiungi bin al PATH
set PATH=%SUMO_HOME%\bin;%PATH%
set SUMO_HOME=%SUMO_HOME%

echo SUMO_HOME impostata a: %SUMO_HOME%

:: Avvia il wizard
python "%SUMO_HOME%\tools\osmWebWizard.py"

pause
