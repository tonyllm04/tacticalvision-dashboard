from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
import uuid
import pandas as pd
import asyncio
from fastapi.concurrency import run_in_threadpool

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

# Diccionario global para guardar el estado de las tareas
TAREAS = {}

def tarea_procesamiento(task_id: str, video_bytes: bytes, filename: str):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, filename)
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            csv_raw = os.path.join(tmpdir, 'raw.csv')
            csv_filtrado = os.path.join(tmpdir, 'filtrado.csv')
            video_ia = os.path.join(tmpdir, 'ia.mp4')

            generar_dataset_deteccion(video_path, csv_raw, max_frames=3600)
            procesar_y_limpiar_dataset(video_path, csv_raw, video_ia, csv_filtrado)

            df = pd.read_csv(csv_filtrado)
            TAREAS[task_id] = {"status": "completed", "data": df.to_dict(orient='records')}
    except Exception as e:
        TAREAS[task_id] = {"status": "error", "message": str(e)}

@app.post('/procesar')
async def iniciar_procesamiento(video: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    TAREAS[task_id] = {"status": "processing"}
    
    contenido = await video.read()
    
    # Ejecuta la tarea en un hilo secundario de CPU para no bloquear la API
    asyncio.create_task(
        run_in_threadpool(tarea_procesamiento, task_id, contenido, video.filename)
    )
    
    return {"task_id": task_id}

@app.get('/estado/{task_id}')
async def obtener_estado(task_id: str):
    if task_id not in TAREAS:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return TAREAS[task_id]