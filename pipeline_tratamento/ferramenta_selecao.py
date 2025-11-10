from time import sleep
from mouse_callback import mouse_callback
from processa_img import processa_img
from mover_img_original import mover_img_original
from menu_genero import menu_genero
from estado_selecao import EstadoSelecao
import cv2

estado = EstadoSelecao()

def ferramenta_selecao(CAMINHO_IMAGEM_ENTRADA , PASTA_SAIDA, img_original, LARGURA_MAX_JANELA, nome_arquivo, estado):
    """Inicia a interface gráfica para selecionar múltiplas ROIs."""
        
    clone = img_original.copy()
    
    h_orig, w_orig = img_original.shape[:2]
    
    ratio = LARGURA_MAX_JANELA / w_orig if w_orig > LARGURA_MAX_JANELA else 1.0
    
    image_display = cv2.resize(img_original, (int(w_orig * ratio), int(h_orig * ratio)), interpolation=cv2.INTER_AREA)
    
    clone_display = image_display.copy()

    window_name = "Ferramenta de recorte - Pressione [y] para salvar, [r] para resetar, [q] para voltar ao menu"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, estado)

    print("\n--- MODO DE SELEÇÃO ATIVADO ---")
    print("1. Clique e arraste para desenhar um retângulo.")
    print("2. Pressione [y] para salvar. A etiqueta será pedida no terminal.")
    print("3. Pressione [r] para limpar")
    print("4. Pressione [q] para voltar ao menu.")
    print("--------------------")

    while True:
        
        temp_display = image_display.copy()
        
        if estado.is_drawing:
            cv2.rectangle(temp_display, estado.ref_point[0], estado.current_mouse_pos, (0, 255, 0), 2)
            
        elif len(estado.ref_point) == 2:
            cv2.rectangle(temp_display, estado.ref_point[0], estado.ref_point[1], (0, 255, 0), 2)
            
        cv2.imshow(window_name, temp_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y") and len(estado.ref_point) == 2:
            
            start_point_orig = (int(estado.ref_point[0][0] / ratio), int(estado.ref_point[0][1] / ratio))
            end_point_orig = (int(estado.ref_point[1][0] / ratio), int(estado.ref_point[1][1] / ratio))
            x1, y1 = min(start_point_orig[0], end_point_orig[0]), min(start_point_orig[1], end_point_orig[1])
            x2, y2 = max(start_point_orig[0], end_point_orig[0]), max(start_point_orig[1], end_point_orig[1])
            roi = clone[y1:y2, x1:x2]
            
            cv2.destroyWindow(window_name)
            
            modo = "Individual"
            
            genero = menu_genero()
            
            if not genero:
                print("[AVISO] Gênero nao selecionado. Voltando...")
                continue
            
            imgSalva = processa_img(roi, nome_arquivo, PASTA_SAIDA, modo, genero)
            
            if imgSalva:
                                
                estado.ref_point.clear()
                image_display = clone_display.copy()
                
                while True:
                    
                    mover_img_original(imgSalva, CAMINHO_IMAGEM_ENTRADA, nome_arquivo)
                    
                    print("=" * 200)
                        
                    resposta = int(input("\n >>> Deseja continuar seleção na mesma imagem ou abrir uma nova imagem?\n >>> [1] - Continuar / [2] - Voltar ao menu: "))
                    
                    if resposta == 1:
                        print("Abrindo ferramenta de recorte novamente...")
                        sleep(1)
                        cv2.namedWindow(window_name)
                        cv2.setMouseCallback(window_name, mouse_callback, estado)
                        break
                    
                    elif resposta == 2:
                        print("\nVoltando ao menu principal...")
                        sleep(1)
                        return
                    
                    else:
                        print("[AVISO] Opção inválida. Por favor, digite 1 ou 2.")
                                    
            estado.ref_point.clear()
            image_display = clone_display.copy()

        if key == ord("r"):
            estado.ref_point.clear()
            image_display = clone_display.copy()
            
        elif key == ord("q"):
            estado.ref_point.clear()
            image_display = clone_display.copy()
            cv2.destroyWindow(window_name)
            print("\nVoltando ao menu principal...")
            sleep(1)
            break
        
    estado.ref_point.clear()
    image_display = clone_display.copy()
    cv2.destroyAllWindows()