import cv2
import pandas as pd
import numpy as np
from collections import Counter
import os
from sklearn.cluster import KMeans

def es_dentro_terreno_juego(caja, ancho_vid, alto_vid):
    x1, y1, x2, y2 = caja
    pie_y = y2
    pie_x = (x1 + x2) // 2
    return (alto_vid * 0.12 < pie_y < alto_vid * 0.98) and (ancho_vid * 0.02 < pie_x < ancho_vid * 0.98)

def clasificar_equipos_kmeans(df_detecciones, cap):
    """
    Agrupa dinámicamente los colores dominantes de los jugadores en 2 clústeres K-Means
    """
    jugadores = df_detecciones[df_detecciones['rol_equipo'] == 'Jugador'] if 'rol_equipo' in df_detecciones.columns else df_detecciones
    if jugadores.empty:
        return df_detecciones

    caracteristicas_color = []
    indices = []

    for idx, row in jugadores.iterrows():
        f_num = int(row['frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
        ret, frame = cap.read()
        if ret:
            caja = list(map(int, row['bbox'].replace('(', '').replace(')', '').split(','))) if isinstance(row['bbox'], str) else [row['x1'], row['y1'], row['x2'], row['y2']]
            x1, y1, x2, y2 = caja
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                h, w, _ = crop.shape
                torso = crop[int(h*0.3):int(h*0.7), :]
                color_promedio = torso.mean(axis=(0,1))
                caracteristicas_color.append(color_promedio)
                indices.append(idx)

    if len(caracteristicas_color) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(caracteristicas_color)
        etiquetas = kmeans.labels_
        
        for idx, etiqueta in zip(indices, etiquetas):
            df_detecciones.loc[idx, 'rol_equipo'] = f'Equipo_{etiqueta + 1}'

    return df_detecciones

def procesar_y_limpiar_dataset(video_original, csv_datos, video_salida, csv_salida_limpio):
    print("Iniciando algoritmo de Consolidación Adaptativa...")
    df = pd.read_csv(csv_datos)
    
    cap = cv2.VideoCapture(video_original)
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    roles_maestros = {}
    posiciones_promedio_id = {}

    print("Fase 1: Extrayendo histogramas y posiciones por ID...")

    for _, row in df.iterrows():
        idx = int(row['id'])
        caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(',')))

        if not es_dentro_terreno_juego(caja, w_video, h_video):
            continue

        if idx not in posiciones_promedio_id:
            posiciones_promedio_id[idx] = []
        posiciones_promedio_id[idx].append((caja[0] + caja[2]) / 2)

    # Reagrupación cromática global adaptativa con K-Means
    print("Fase 2: Consolidando roles mediante K-Means...")
    if 'rol_equipo' not in df.columns:
        df['rol_equipo'] = 'Jugador'
    if 'bbox' not in df.columns and 'coords_caja' in df.columns:
        df['bbox'] = df['coords_caja']

    df = clasificar_equipos_kmeans(df, cap)

    for idx in df['id'].unique():
        sub_df = df[df['id'] == idx]
        if not sub_df.empty:
            rol_ganador = sub_df['rol_equipo'].mode()[0]
            roles_maestros[idx] = rol_ganador

    PUNTOS_PENALTI_MANUALES = [
        {"x": int(w_video * 0.16), "y": int(h_video * 0.27)}, 
        {"x": int(w_video * 0.84), "y": int(h_video * 0.27)} 
    ]
    RADIO_EXCLUSION_PENALTI = 15  

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

        if balon_x != -1 and balon_y != -1:
            cv2.circle(frame, (balon_x, balon_y), 7, (0, 128, 255), -1)
            cv2.circle(frame, (balon_x, balon_y), 0, (0, 0, 0), 1)
            cv2.putText(frame, "BALON", (balon_x - 18, balon_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1, cv2.LINE_AA)

        for _, row in df_frame.iterrows():
            idx = int(row['id'])
            rol = roles_maestros.get(idx, "Desconocido")
            
            if rol == "Arbitro":
                continue

            caja = list(map(int, row['coords_caja'].replace('(', '').replace(')', '').split(','))) if 'coords_caja' in row else [row['x1'], row['y1'], row['x2'], row['y2']]
            
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