from estado_selecao import EstadoSelecao
import cv2

estado = EstadoSelecao()

def mouse_callback(event, x, y, flags, estado):
    if event == cv2.EVENT_LBUTTONDOWN:
        estado.ref_point = [(x, y)]
        estado.is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        estado.current_mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        estado.ref_point.append((x, y))
        estado.is_drawing = False
        