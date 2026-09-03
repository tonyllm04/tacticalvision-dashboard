import cv2
import pandas as pd
import numpy as np
from collections import Counter
import os

def es_dentro_terreno_juego(caja, ancho_vid, alto_vid):
    x1, y1, x2, y2 = caja
    pie_y = y2
    pie_x = (x1 + x2) // 2
    # Filtrar bordes externos
    return (alto_vid * 0.12 < pie_y < alto_vid * 0.98) and (ancho_vid * 0.02 < pie_x < ancho_vid * 0.98)

def clasificar_rol_por_color_adaptativo(recorte):
    """
    Clasificación cromática con corrección de iluminación (CLAHE)
    para mitigar sombras en el césped y torso.
    """
    if recorte is None or recorte.size == 0:
        return "Desconocido"
        
    img = cv2.resize(recorte, (20, 20))
    
    # Corrección de sombras/brillo con CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    cl = clahe.apply(l)
    img_corregida = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    img_hsv = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2HSV)
    pixeles = img_hsv.reshape((-1, 3)).astype(np.float32)
    
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centros = cv2.kmeans(pixeles, 2, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    for centro in centros:
        h, s, v = centro[0], centro[1], centro[2]
        
        # Filtro de césped (descartar verde del fondo)
        if 35 <= h <= 85 and s > 30:
            continue
            
        # Árbitros / Porteros con equipación fosforita
        if 20 <= h <= 80 and s > 110 and v > 100:
            return "Fosforito"
            
        # Equipo 1 (Tonos rojos / oscuros cálidos)
        if (h <= 12 or h >= 160) and s > 60:
            return "Equipo_1"
            
        # Equipo 2 (Tonos claros / blancos)
        if s < 60 and v > 130:
            return "Equipo_2"
            
    return "Desconocido"

def filtrar_fuera_del_campo(caja, ancho_vid, alto_vid):
    x1, y1, x2, y2 = caja
    centro_y = (y1 + y2) // 2
    # Ajuste dinámico de margen superior
    return centro_y < int(alto_vid * 0.15)

def procesar_y_limpiar_dataset(video_original, csv_datos, video_salida, csv_salida_limpio):
    print("Iniciando algoritmo de Consolidación Adaptativa...")
    df = pd.read_csv(csv_datos)
    
    cap = cv2.VideoCapture(video_original)
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # --- CAMBIO 2: Estructuras dinámicas (Sin IDs hardcodeados) ---
    roles_maestros = {}
    votos_color_id = {}
    posiciones_promedio_id = {}

    # --------------------------------------------------------------------------
    # PASO 1: Análisis Cromático y Espacial por ID
    # --------------------------------------------------------------------------
    print("Fase 1: Extrayendo histogramas y posiciones por ID...")

    for _, row in df.iterrows():
        idx = int(row['id'])
        camiseta_path = str(row['camiseta_path'])
        caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(',')))

        if not es_dentro_terreno_juego(caja, w_video, h_video):
            continue

        # Guardar posición para diferenciar roles según ubicación en el campo
        if idx not in posiciones_promedio_id:
            posiciones_promedio_id[idx] = []
        posiciones_promedio_id[idx].append((caja[0] + caja[2]) / 2)
            
        if os.path.exists(camiseta_path):
            recorte = cv2.imread(camiseta_path)
            if recorte is not None:
                h_rec, w_rec = recorte.shape[:2]
                pecho = recorte[int(h_rec*0.20):int(h_rec*0.80), int(w_rec*0.20):int(w_rec*0.80)]
                
                rol_detectado = clasificar_rol_por_color_adaptativo(pecho)
                if rol_detectado != "Desconocido":
                    if idx not in votos_color_id:
                        votos_color_id[idx] = []
                    votos_color_id[idx].append(rol_detectado)

    # --------------------------------------------------------------------------
    # PASO 2: Reglas de Consolidación Dinámica
    # --------------------------------------------------------------------------
    print("Fase 2: Consolidando roles dinámicamente...")
    
    for idx, votos in votos_color_id.items():
        if votos:
            rol_ganador = Counter(votos).most_common(1)[0][0]
            
            # --- CAMBIO 3: Desambiguación Dinámica de "Fosforito" ---
            if rol_ganador == "Fosforito":
                pos_x_media = np.mean(posiciones_promedio_id.get(idx, [w_video / 2]))
                # Si pasa la mayor parte del tiempo cerca de las porterías (extremos X) es Portero
                if pos_x_media < w_video * 0.12 or pos_x_media > w_video * 0.88:
                    roles_maestros[idx] = "Equipo_2"
                else:
                    roles_maestros[idx] = "Arbitro"
            else:
                roles_maestros[idx] = rol_ganador

    # --------------------------------------------------------------------------
    # PASO 3: Definición de Oclusión para Puntos de Penalti
    # --------------------------------------------------------------------------
    PUNTOS_PENALTI_MANUALES = [
        {"x": int(w_video * 0.16), "y": int(h_video * 0.27)}, 
        {"x": int(w_video * 0.84), "y": int(h_video * 0.27)} 
    ]
    RADIO_EXCLUSION_PENALTI = 15  

    # --------------------------------------------------------------------------
    # PASO 4: Renderizado Final
    # --------------------------------------------------------------------------
    print("Fase 3: Renderizando vídeo final...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_salida, fourcc, fps, (w_video, h_video))
    
    lista_frames = sorted(df['frame'].unique())
    colores_bgr = {
        "Equipo_1": (0, 0, 255),       # Rojo
        "Equipo_2": (255, 255, 255),   # Blanco
        "Arbitro": (0, 255, 255),      # Amarillo
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
            rol = roles_maestros.get(idx, "Desconocido")
            
            # Filtrar árbitros en el resultado visual si se prefiere no dibujarlos
            if rol == "Arbitro":
                continue

            caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(',')))
            
            # --- CAMBIO 4: Filtrado dinámico por geometría real del campo ---
            if not es_dentro_terreno_juego(caja, w_video, h_video):
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

    print(f"\n[INFO] Se han descartado {total_filtrados_penalti} detecciones en puntos de penalti.")
    print(f"Proceso completado. Vídeo: {video_salida} | CSV: {csv_salida_limpio}")

if __name__ == "__main__":
    procesar_y_limpiar_dataset(
        'Partido2.mp4', 
        'posiciones_partido2_raw.csv', 
        'Partido2_IA.mp4', 
        'posiciones_partido2_filtrado.csv'
    )