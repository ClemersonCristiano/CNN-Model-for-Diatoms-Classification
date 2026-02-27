#!/bin/bash

set -e

echo ""
echo "+---------------------------------------------+"
echo "   CNN Model for Diatoms Classification        "
echo "   Setup Script                                "
echo "+---------------------------------------------+"
echo ""

# --- Dependências de sistema (apenas Linux) ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "[INFO] Sistema Linux detectado."
    echo "[INFO] Verificando dependência de sistema: python3-tk..."

    if python3 -c "import tkinter" &>/dev/null; then
        echo "[OK] tkinter já está instalado."
    else
        echo "[INFO] Instalando python3-tk via apt-get..."
        sudo apt-get update -qq
        sudo apt-get install -y python3-tk
        echo "[OK] python3-tk instalado com sucesso."
    fi
else
    echo "[INFO] Sistema não-Linux detectado. tkinter já deve estar disponível."
fi

echo ""

# --- Ambiente virtual ---
if [ ! -d "venv" ]; then
    echo "[INFO] Criando ambiente virtual..."
    python3 -m venv venv
    echo "[OK] Ambiente virtual criado em ./venv"
else
    echo "[OK] Ambiente virtual já existe em ./venv"
fi

echo ""

# --- Ativação e instalação das dependências Python ---
echo "[INFO] Ativando ambiente virtual e instalando dependências..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "+---------------------------------------------+"
echo "   Setup concluído com sucesso!"
echo ""
echo "   Para ativar o ambiente virtual:"
echo "     source venv/bin/activate"
echo ""
echo "   Para rodar o pipeline de pré-processamento:"
echo "     cd pipeline_tratamento && python main.py"
echo "+---------------------------------------------+"
echo ""
