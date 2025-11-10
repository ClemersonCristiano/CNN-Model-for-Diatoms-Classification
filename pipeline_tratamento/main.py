from limpar_terminal import limpar_terminal
from tratar_Pasta import tratar_pasta
from tratar_img_cortada import tratar_img_cortada
from recortar_img import recortar_img
from estado_selecao import EstadoSelecao
from time import sleep
        
estado = EstadoSelecao()
        
def main():
    """Função principal que controla o loop do programa."""
    while True:
        
        limpar_terminal()
        
        print("\n+------------ MENU PRINCIPAL ------------+")
        print("Escolha uma opção de tratamento de imagem:")

        escolha = input(">>> Digite a letra correspondente: [s] = Recortar Imagem | [d] = Tratar imagem já recortada | [p] = Tratar todas as imagens em uma pasta | [q] = Sair: ").lower()

        if escolha == 'q':
            print("\nEncerrando o programa...")
            break

        if escolha == 's':
            recortar_img()
    
        elif escolha == 'd':
            tratar_img_cortada()
                
        elif escolha == 'p':
            tratar_pasta()

        else:
            print("[AVISO] Opção inválida. Tente novamente.")
            sleep(1)

    print("\nProcesso concluído. Encerrando...")
    
    estado.ref_point.clear()
    estado.is_drawing = False
    estado.current_mouse_pos = (0, 0)

if __name__ == "__main__":
    main()