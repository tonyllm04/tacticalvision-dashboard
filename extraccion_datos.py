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
    Extrae el tercio superior del cuerpo del jugador basándose en la caja de detección.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = box

    alto_caja = y2 - y1
    ancho_caja = x2 - x1

    if alto_caja < 15 or ancho_caja < 5:
        return None

    y_min = max(0, int(y1 + alto_caja * 0.15))
    y_max = min(h_frame, int(y1 + alto_caja * 0.45))
    
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
    
    MAX_FRAMES_MEMORIA = 15

    try:
        while cap.isOpened():
            if frame_count >= max_frames: break

            ret, frame = cap.read()
            if not ret: break

            frame_count += 1

            if frame_count % FRAME_STRIDE != 0:
                continue

            processed_frames += 1

            if processed_frames >= max_frames:
                break

            results = model.track(
                source=frame, 
                persist=True, 
                classes=[0, 32], 
                imgsz=1280, 
                conf=0.03,
                seed=42,
                verbose=False
            )

            balon_detectado_este_frame = False
            bx, by = -1, -1

            boxes_objeto = results[0].boxes

            if boxes_objeto is not None and len(boxes_objeto) > 0:
                for box_det in boxes_objeto:
                    cls_id = int(box_det.cls[0])

                    if cls_id == 32:  # Balón
                        bx1, by1, bx2, by2 = box_det.xyxy.int().cpu().tolist()[0]
                        bx = int((bx1 + bx2) / 2)
                        by = int((by1 + by2) / 2)
                        balon_detectado_este_frame = True
                        break
            
            if balon_detectado_este_frame:
                if ultimo_balon_x != -1:
                    vx = bx - ultimo_balon_x
                    vy = by - ultimo_balon_y
                else:
                    vx, vy = 0, 0
                ultimo_balon_x, ultimo_balon_y = bx, by
                frames_balon_desaparecido = 0
            else:
                frames_balon_desaparecido += 1
                if frames_balon_desaparecido <= MAX_FRAMES_MEMORIA and ultimo_balon_x != -1:
                    # Proyectamos la posición según la trayectoria previa
                    bx = int(ultimo_balon_x + (vx * frames_balon_desaparecido))
                    by = int(ultimo_balon_y + (vy * frames_balon_desaparecido))
            
            # Guardado en CSV
            if boxes_objeto.id is not None:
                boxes_coords = boxes_objeto.xyxy.int().cpu().tolist()
                ids = boxes_objeto.id.int().cpu().tolist()
                clases = boxes_objeto.cls.int().cpu().tolist()

                for box, idx, cls_id in zip(boxes_coords, ids, clases):
                    if cls_id == 0:  # Jugadores
                        coords_str = f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"
                        recorte_camiseta = extraer_torso_proporcional(frame, box)
                        nombre_archivo_camiseta = "None"

                        if recorte_camiseta is not None and recorte_camiseta.size > 0:
                            nombre_archivo_camiseta = f"{carpeta_camisetas}/f{frame_count}_id{idx}.jpg"
                            cv2.imwrite(nombre_archivo_camiseta, recorte_camiseta)

                        writer.writerow([
                            frame_count,
                            idx,
                            coords_str,
                            nombre_archivo_camiseta,
                            int(bx),
                            int(by)
                        ])
                
    except KeyboardInterrupt:
        print("\nInterrupción manual.")
    finally:
        f.close()
        cap.release()

        import gc
        del model
        gc.collect()

        print(f"\nDataset generado con éxito.")

if __name__ == "__main__":
    generar_dataset_deteccion('clipp1.mp4', 'posiciones_partido1_raw.csv', max_frames=3600)