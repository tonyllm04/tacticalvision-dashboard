import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

from extraccion_datos import generar_dataset_deteccion
from visualizar_seguimiento_equipos import procesar_y_limpiar_dataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario simple para rastrear estados de trabajos en memoria
jobs_status = {}

def tarea_procesamiento_segundo_plano(job_id: str, video_path: str):
    try:
        csv_raw = f"temp_raw_{job_id}.csv"
        csv_limpio = f"temp_limpio_{job_id}.csv"

        jobs_status[job_id]["status"] = "processing"
        jobs_status[job_id]["message"] = "Ejecutando YOLOv8 sobre 1800 frames..."

        # 1. Generar detecciones
        generar_dataset_deteccion(video_path, csv_raw, max_frames=1800)

        # 2. Limpiar y consolidar
        procesar_y_limpiar_dataset(video_path, csv_raw, None, csv_limpio)

        # 3. Cargar resultado final
        import pandas as pd
        df_resultado = pd.read_csv(csv_limpio)
        
        # Limpieza de archivos temporales de disco
        for path in [video_path, csv_raw, csv_limpio]:
            if os.path.exists(path):
                os.remove(path)

        jobs_status[job_id]["status"] = "completed"
        jobs_status[job_id]["data"] = df_resultado.to_dict(orient="records")

    except Exception as e:
        jobs_status[job_id]["status"] = "failed"
        jobs_status[job_id]["error"] = str(e)
        if os.path.exists(video_path):
            os.remove(video_path)

@app.post("/procesar")
async def iniciar_procesamiento(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    video_path = f"temp_video_{job_id}.mp4"

    # Escribir en disco por bloques (chunking) para no congelar el evento asíncrono
    with open(video_path, "wb") as buffer:
        while content := await video.read(1024 * 1024):  # Leer de 1 MB en 1 MB
            buffer.write(content)

    jobs_status[job_id] = {
        "status": "queued",
        "data": None,
        "error": None
    }

    # Encolar la tarea pesada
    background_tasks.add_task(tarea_procesamiento_segundo_plano, job_id, video_path)

    # Devolver respuesta inmediata
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def obtener_estado(job_id: str):
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    
    info = jobs_status[job_id]
    
    # Si ya se completó, liberamos la memoria tras la entrega
    if info["status"] in ["completed", "failed"]:
        respuesta = info.copy()
        # Opcional: borrar el job del diccionario tras entregarlo
        # del jobs_status[job_id] 
        return respuesta
        
    return {"status": info["status"]}