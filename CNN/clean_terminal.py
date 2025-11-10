import os
def clean_terminal():
    """Limpa a tela do terminal, compatível com Windows, macOS e Linux."""
    # Para Windows, o comando é 'cls'. Para macOS e Linux, é 'clear'.
    os.system('cls' if os.name == 'nt' else 'clear')