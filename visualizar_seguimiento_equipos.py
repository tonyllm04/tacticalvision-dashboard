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
    print("🚀 Iniciando algoritmo de Consolidación por Descarte Cromático...")
    df = pd.read_csv(csv_datos)
    
    cap = cv2.VideoCapture(video_original)
    w_video = int(cap.get(3))
    h_video = int(cap.get(4))
    fps = cap.get(5) or 30
    
    roles_maestros = {}
    votos_color_id = {}

    # --------------------------------------------------------------------------
    # CONFIGURACIÓN DE IDs DE CONTROL (Portero Verde = 2697)
    # --------------------------------------------------------------------------
    ARBITROS_BASE = [11, 15, 122, 123, 342, 449, 595, 602, 1115, 2020, 2585, 2763, 2697, 2912]
    PORTERO_2_BASE = [273]  # Portero Verde (Equipo 2)
    PORTERO_1_BASE = [204, 3067]   # Porteros oscuros (Equipo 1)
    JUGADORES_CAMPO_FORZADOS = {19: "Equipo_1"}

    # Lista de rescate inicial para ignorar la eliminación por altura
    PORTEROS_Y_RESCATADOS = PORTERO_2_BASE + PORTERO_1_BASE

    IDS_FUERA_DEL_CAMPO = [2, 20, 259, 287, 344, 416, 510, 707, 792, 334, 400, 452, 457, 605, 617, 613, 706, 756, 721, 824, 870, 912, 883, 1217, 1105, 1328, 1349, 1445, 1528, 1542, 1506, 1550, 1554, 1593, 1588, 1678, 1679, 1799, 1839, 1851, 1854, 1961, 2003, 2074, 2333, 2417, 2405, 2419, 2460, 2511, 2530, 2535, 2536, 2643, 2781]

    # --------------------------------------------------------------------------
    # PASO 1: Análisis Cromático Global (Solo si existen las imágenes)
    # --------------------------------------------------------------------------
    print("📊 Fase 1: Extrayendo histogramas de color por ID...")

    for _, row in df.iterrows():
        idx = int(row['id'])
        camiseta_path = row['camiseta_path']
        caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(',')))

        if idx not in PORTEROS_Y_RESCATADOS:
            if filtrar_fuera_del_campo(caja, w_video, h_video):
                continue
            
        if os.path.exists(str(camiseta_path)):
            recorte = cv2.imread(camiseta_path)
            h_rec, w_rec = recorte.shape[:2]
            pecho = recorte[int(h_rec*0.25):int(h_rec*0.75), int(w_rec*0.25):int(w_rec*0.75)]
            
            rol_detectado = clasificar_rol_por_color(pecho)
            if rol_detectado != "Desconocido":
                if idx not in votos_color_id:
                    votos_color_id[idx] = []
                votos_color_id[idx].append(rol_detectado)

    # --------------------------------------------------------------------------
    # PASO 2: Reglas de Consolidación (Forzado de Casos Críticos Garantizado)
    # --------------------------------------------------------------------------
    print("🔒 Fase 2: Consolidando roles...")
    
    # INYECCIÓN DIRECTA DE SEGURIDAD
    for arb in ARBITROS_BASE:
        roles_maestros[arb] = "Arbitro"
    for p2 in PORTERO_2_BASE:
        roles_maestros[p2] = "Equipo_2"
    for p1 in PORTERO_1_BASE:
        roles_maestros[p1] = "Equipo_1"
    for idx_f, rol_f in JUGADORES_CAMPO_FORZADOS.items():
        roles_maestros[idx_f] = rol_f
        
    # Consolidación normal por votos
    for idx, votos in votos_color_id.items():
        if idx in roles_maestros:
            continue # Evitar pisar los porteros/árbitros ya forzados estáticamente
            
        if votos:
            rol_ganador = Counter(votos).most_common(1)[0][0]
            
            if rol_ganador == "Fosforito":
                # --- AQUÍ ESTÁ EL CAMBIO CRÍTICO ---
                # Si es fosforito pero NO está en la lista de árbitros conocidos, 
                # obligatoriamente es el Portero Verde (Equipo 2) que ha mutado de ID.
                if idx not in ARBITROS_BASE:
                    roles_maestros[idx] = "Equipo_2"
                    # Lo rescatamos dinámicamente para que no sea eliminado por filtros de altura
                    PORTEROS_Y_RESCATADOS.append(idx)
                else:
                    roles_maestros[idx] = "Arbitro"
            else:
                roles_maestros[idx] = rol_ganador

    # Aseguramos de forma redundante que no se hayan sobreescrito las bases estáticas
    for arb in ARBITROS_BASE: 
        roles_maestros[arb] = "Arbitro"
    for p2 in PORTERO_2_BASE: 
        roles_maestros[p2] = "Equipo_2"
    for p1 in PORTERO_1_BASE: 
        roles_maestros[p1] = "Equipo_1"

    # --------------------------------------------------------------------------
    # PASO 3: Definición Estricta de Oclusión para Puntos de Penalti
    # --------------------------------------------------------------------------
    PUNTOS_PENALTI_MANUALES = [
        {"x": 308, "y": 295},  
        {"x": 1610, "y": 295}  
    ]
    RADIO_EXCLUSION_PENALTI = 15  

    # --------------------------------------------------------------------------
    # PASO 4: Renderizado Final
    # --------------------------------------------------------------------------
    print("🎥 Fase 3: Renderizando vídeo final...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_salida, fourcc, fps, (w_video, h_video))
    
    lista_frames = sorted(df['frame'].unique())
    colores_bgr = {
        "Equipo_1": (0, 0, 255),            # Rojo
        "Equipo_2": (255, 255, 255),        # Blanco
        "Arbitro": (0, 255, 255),           # Amarillo
        "Desconocido": (140, 140, 140)
    }

    filas_csv_limpio = []
    total_filtrados_penalti = 0

    for f_num in lista_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
        ret, frame = cap.read()
        if not ret: break
        
        df_frame = df[df['frame'] == f_num]

        balon_x, balon_y = -1, -1
        if not df_frame.empty:
            val_x = df_frame.iloc[0].get('balon_x', -1)
            val_y = df_frame.iloc[0].get('balon_y', -1)
            
            if pd.notna(val_x) and pd.notna(val_y):
                bx_temp = int(float(val_x))
                by_temp = int(float(val_y))
                
                es_punto_penalti = False
                if bx_temp != -1 and by_temp != -1:
                    for punto in PUNTOS_PENALTI_MANUALES:
                        distancia = np.sqrt((bx_temp - punto["x"])**2 + (by_temp - punto["y"])**2)
                        if distancia <= RADIO_EXCLUSION_PENALTI:
                            es_punto_penalti = True
                            break
                
                if es_punto_penalti:
                    total_filtrados_penalti += 1
                else:
                    balon_x = bx_temp
                    balon_y = by_temp

        # Dibujar Balón
        if balon_x != -1 and balon_y != -1:
            cv2.circle(frame, (balon_x, balon_y), 7, (0, 128, 255), -1)
            cv2.circle(frame, (balon_x, balon_y), 0, (0, 0, 0), 1)
            cv2.putText(frame, "BALON", (balon_x - 18, balon_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1, cv2.LINE_AA)

        for _, row in df_frame.iterrows():
            idx = int(row['id'])

            # Eliminar público / IDs de fuera del campo
            if idx in IDS_FUERA_DEL_CAMPO:
                continue

            rol = roles_maestros.get(idx, "Desconocido")
            
            # Los árbitros reales no se pintan en el vídeo
            if rol == "Arbitro":
                continue

            caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(',')))
            
            # Evitamos borrar a los porteros (originales o rescatados dinámicamente) por la altura del campo
            if idx not in PORTEROS_Y_RESCATADOS:
                if filtrar_fuera_del_campo(caja, w_video, h_video):
                    continue
                
            centro_x = (caja[0] + caja[2]) // 2
            pies_y = caja[3]

            filas_csv_limpio.append({
                'frame': f_num,
                'id_jugador': idx,
                'rol_equipo': rol,
                'pos_x': centro_x,
                'pos_y': pies_y,
                'bbox': f"({caja[0]}, {caja[1]}, {caja[2]}, {caja[3]})",
                'balon_x': balon_x,
                'balon_y': balon_y
            })

            # Pintar la caja del jugador/portero
            color_caja = colores_bgr.get(rol, (140, 140, 140))
            cv2.rectangle(frame, (caja[0], caja[1]), (caja[2], caja[3]), color_caja, 2)
            cv2.rectangle(frame, (caja[0], caja[1] - 16), (caja[0] + 110, caja[1]), (0, 0, 0), -1)
            cv2.putText(frame, f"ID {idx}: {rol}", (caja[0] + 4, caja[1] - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_caja, 1, cv2.LINE_AA)
                        
        out.write(frame)
        
    cap.release()
    out.release()

    df_limpio = pd.DataFrame(filas_csv_limpio)
    df_limpio.to_csv(csv_salida_limpio, index=False)

    print(f"\n[INFO] Se han descartado {total_filtrados_penalti} detecciones en los puntos de penalti.")
    print(f"Proceso completado con éxito. Vídeo limpio: {video_salida}")
    print(f"CSV de estadísticas limpio: {csv_salida_limpio} (Filas finales: {len(df_limpio)})")

if __name__ == "__main__":
    procesar_y_limpiar_dataset('Partido1.mp4', 'posiciones_partido1_raw.csv', 
                               'Partido1_IA.mp4', 'posiciones_partido1_filtrado.csv')