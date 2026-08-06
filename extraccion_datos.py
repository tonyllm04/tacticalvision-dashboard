import cv2
import pandas as pd
import numpy as np
import os
import csv
from ultralytics import YOLO

carpeta_camisetas = "dataset_camisetas_limpias"
if not os.path.exists(carpeta_camisetas):
    os.makedirs(carpeta_camisetas)

def extraer_torso_proporcional(frame, box):
    """
    Estrategia para planos lejanos: Extrae estrictamente el tercio superior 
    del cuerpo del jugador basándose en la caja de detección, evitando césped y piernas.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = box

    alto_caja = y2 - y1
    ancho_caja = x2 - x1

    # Si la caja de detección es ridículamente pequeña o deforme, la descartamos
    if alto_caja < 15 or ancho_caja < 5:
        return None

    # Tomamos verticalmente desde el 15% hasta el 45% de la altura del jugador (el torso/camiseta)
    y_min = max(0, int(y1 + alto_caja * 0.15))
    y_max = min(h_frame, int(y1 + alto_caja * 0.45))
    
    # Recortamos ligeramente los laterales (un 15% de cada lado) para evitar meter fondo/césped
    x_min = max(0, int(x1 + ancho_caja * 0.15))
    x_max = min(w_frame, int(x2 - ancho_caja * 0.15))

    if (y_max - y_min) > 2 and (x_max - x_min) > 2:
        return frame[y_min:y_max, x_min:x_max]
        
    return None

def generar_dataset_deteccion(video_path, csv_output, max_frames=3600):
    print(f"Cargando YOLOv8m para personas y balón...")
    model = YOLO("yolov8m.pt") 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("Error al abrir el vídeo.")
        return
    
    f = open(csv_output, mode='w', newline='')
    writer = csv.writer(f)
    writer.writerow(['frame', 'id', 'coords_caja', 'camiseta_path', 'balon_x', 'balon_y'])

    frame_count = 0
    FRAME_STRIDE = 3
    processed_frames = 0

    ultimo_balon_x, ultimo_balon_y = -1, -1
    frames_balon_desaparecido = 0
    MAX_FRAMES_MEMORIA = 5  # Persistencia en el aire

    # --------------------------------------------------------------------------
    # CONFIGURACIÓN DE OCLUSIÓN (COORDEANADAS DE LOS PUNTOS DE PENALTI)
    # --------------------------------------------------------------------------
    # Ajusta estos puntos según las coordenadas exactas en tus imágenes de 1280x720 o el tamaño nativo.
    PUNTO_PENALTI_IZQ = {"x": 230, "y": 415, "radio": 15}
    PUNTO_PENALTI_DER = {"x": 1770, "y": 415, "radio": 15}

    try:
        while cap.isOpened():
            if frame_count >= max_frames: break

            ret, frame = cap.read()
            if not ret: break

            frame_count += 1

            # Procesar solo 1 de cada 3 frames
            if frame_count % FRAME_STRIDE != 0:
                continue

            processed_frames += 1

            # Limitar número real de frames procesados
            if processed_frames >= max_frames:
                break

            # Enviamos el frame modificado (con los puntos de penalti "borrados") a YOLO
            results = model.track(source=frame, persist=True, classes=[0, 32], imgsz=640, conf=0.25, verbose=False)

            balon_detectado_este_frame = False
            bx, by = -1, -1

            boxes_objeto = results[0].boxes
            if boxes_objeto is not None and len(boxes_objeto) > 0:
                for box_det in boxes_objeto:
                    cls_id = int(box_det.cls[0])
                    if cls_id == 32:  # Posible balón
                        bx1, by1, bx2, by2 = box_det.xyxy.int().cpu().tolist()[0]
                        posible_bx = int((bx1 + bx2) / 2)
                        posible_by = int((by1 + by2) / 2)

                        dist_izq = ((posible_bx - PUNTO_PENALTI_IZQ["x"])**2 + (posible_by - PUNTO_PENALTI_IZQ["y"])**2)**0.5
                        dist_der = ((posible_bx - PUNTO_PENALTI_DER["x"])**2 + (posible_by - PUNTO_PENALTI_DER["y"])**2)**0.5

                        esta_en_penalti = (dist_izq < PUNTO_PENALTI_IZQ["radio"]) or (dist_der < PUNTO_PENALTI_DER["radio"])

                        if esta_en_penalti:
                            if ultimo_balon_x == -1 or frames_balon_desaparecido > 3:
                                continue
                        
                        bx = posible_bx
                        by = posible_by
                        balon_detectado_este_frame = True

                        break
            
            # --- Lógica de memoria/inercia normal ---
            if balon_detectado_este_frame:
                ultimo_balon_x, ultimo_balon_y = bx, by
                frames_balon_desaparecido = 0
            else:
                frames_balon_desaparecido += 1
                if frames_balon_desaparecido <= MAX_FRAMES_MEMORIA and ultimo_balon_x != -1:
                    bx, by = ultimo_balon_x, ultimo_balon_y
                else:
                    bx, by = -1, -1 
            
            # --- Guardar datos de jugadores (usamos el 'frame' original limpio de marcas negras) ---
            if boxes_objeto.id is not None:
                boxes_coords = boxes_objeto.xyxy.int().cpu().tolist()
                ids = boxes_objeto.id.int().cpu().tolist()
                clases = boxes_objeto.cls.int().cpu().tolist()

                for box, idx, cls_id in zip(boxes_coords, ids, clases):
                    if cls_id == 0:
                        coords_str = f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"
                        recorte_camiseta = extraer_torso_proporcional(frame, box) # Frame original limpio
                        nombre_archivo_camiseta = "None"

                        if recorte_camiseta is not None and recorte_camiseta.size > 0:
                            nombre_archivo_camiseta = f"{carpeta_camisetas}/f{frame_count}_id{idx}.jpg"
                            cv2.imwrite(nombre_archivo_camiseta, recorte_camiseta)

                        writer.writerow([frame_count, idx, coords_str, nombre_archivo_camiseta, bx, by])
                
    except KeyboardInterrupt:
        print("\nInterrupción manual.")
    finally:
        f.close()
        cap.release()

        print(f"\n💾 Dataset generado ocultando los puntos de penalti con éxito.")

if __name__ == "__main__":
    generar_dataset_deteccion('Partido1.mp4', 'posiciones_partido1_raw.csv', max_frames=3600)