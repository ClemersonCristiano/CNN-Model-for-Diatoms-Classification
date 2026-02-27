@echo off
setlocal

echo.
echo +---------------------------------------------+
echo    CNN Model for Diatoms Classification
echo    Setup Script - Windows
echo +---------------------------------------------+
echo.

REM --- Verificar se Python esta instalado ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale o Python 3.8+ em https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python encontrado:
python --version
echo.

REM --- Verificar tkinter (no Windows ja vem com o Python) ---
python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] tkinter nao encontrado.
    echo [AVISO] Certifique-se de que instalou o Python pelo instalador oficial em https://www.python.org/
    echo [AVISO] Na instalacao, marque a opcao "tcl/tk and IDLE".
    pause
    exit /b 1
) else (
    echo [OK] tkinter disponivel.
)

echo.

REM --- Ambiente virtual ---
if not exist "venv" (
    echo [INFO] Criando ambiente virtual...
    python -m venv venv
    echo [OK] Ambiente virtual criado em .\venv
) else (
    echo [OK] Ambiente virtual ja existe em .\venv
)

echo.

REM --- Instalar dependencias ---
echo [INFO] Ativando ambiente virtual e instalando dependencias...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt

echo.
echo +---------------------------------------------+
echo    Setup concluido com sucesso!
echo.
echo    Para ativar o ambiente virtual:
echo      venv\Scripts\activate
echo.
echo    Para rodar o pipeline de pre-processamento:
echo      cd pipeline_tratamento
echo      python main.py
echo +---------------------------------------------+
echo.

pause
