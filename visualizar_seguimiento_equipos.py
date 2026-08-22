import cv2
import pandas as pd
import numpy as np
from collections import Counter
import os

def clasificar_rol_por_color(recorte):
    """
    K-Means ultra robusto enfocado en los jugadores de campo.
    Agrupa los colores fluorescentes/fosforitos en una sola etiqueta para no confundirlos.
    """
    if recorte is None or recorte.size == 0:
        return "Desconocido"
        
    img = cv2.resize(recorte, (20, 20))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixeles = img_hsv.reshape((-1, 3)).astype(np.float32)
    
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, etiquetas, centros = cv2.kmeans(pixeles, 2, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    for centro in centros:
        h, s, v = centro[0], centro[1], centro[2]
        
        # Filtro unificado para ropa fosforita (Árbitro amarillo y Portero verde)
        if 20 <= h <= 80 and s > 120 and v > 110:
            return "Fosforito"
            
        # Filtro de césped de fondo
        if 35 <= h <= 85 and s > 30:
            continue
            
        # Jugadores de campo principales
        if (h <= 10 or h >= 165) and s > 70:
            return "Equipo_1"  # Rojo
        if s < 65 and v > 140:
            return "Equipo_2"  # Blanco
            
    return "Desconocido"

def filtrar_fuera_del_campo(caja, ancho_vid, alto_vid):
    x1, y1, x2, y2 = caja
    centro_y = (y1 + y2) // 2
    if centro_y < int(alto_vid * 0.22):
        return True
    return False

def procesar_y_limpiar_dataset(video_original, csv_datos, video_salida, csv_salida_limpio):
    print("Iniciando Consolidación de datos...")
    df = pd.read_csv(csv_datos)
    
    # Asignación manual de roles según listas base
    ARBITROS_BASE = [11, 15, 122, 123, 342, 449, 595, 602, 1115, 2020, 2585, 2763, 2697, 2912]
    PORTERO_2_BASE = [273]
    PORTERO_1_BASE = [204, 3067]
    IDS_FUERA_DEL_CAMPO = [2, 20, 259, 287, 344, 416, 510, 707, 792, 334, 400, 452, 457, 605, 617, 613, 706, 756, 721, 824, 870, 912, 883, 1217, 1105, 1328, 1349, 1445, 1528, 1542, 1550, 1554, 1593, 1588, 1678, 1679, 1799, 1839, 1851, 1854, 1961, 2003, 2074, 2333, 2417, 2405, 2419, 2460, 2511, 2530, 2535, 2536, 2643, 2781]

    filas_csv_limpio = []

    for _, row in df.iterrows():
        idx = int(row['id'])
        if idx in IDS_FUERA_DEL_CAMPO or idx in ARBITROS_BASE:
            continue

        caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(',')))
        centro_x = (caja[0] + caja[2]) // 2
        pies_y = caja[3]

        # Asignación rápida de rol
        if idx in PORTERO_1_BASE:
            rol = "Equipo_1"
        elif idx in PORTERO_2_BASE:
            rol = "Equipo_2"
        else:
            rol = "Equipo_1" if idx % 2 == 0 else "Equipo_2" # Asignación de prueba si no hay fotos

        filas_csv_limpio.append({
            'frame': row['frame'],
            'id_jugador': idx,
            'rol_equipo': rol,
            'pos_x': centro_x,
            'pos_y': pies_y,
            'bbox': row['coords_caja'],
            'balon_x': row['balon_x'],
            'balon_y': row['balon_y']
        })

    df_limpio = pd.DataFrame(filas_csv_limpio)
    df_limpio.to_csv(csv_salida_limpio, index=False)
    print("CSV filtrado correctamente sin renderizar vídeo.")

if __name__ == "__main__":
    procesar_y_limpiar_dataset('Partido1.mp4', 'posiciones_partido1_raw.csv', 
                               'Partido1_IA.mp4', 'posiciones_partido1_filtrado.csv')