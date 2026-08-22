from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
import pandas as pd

from extraccion_datos import generar_dataset_deteccion
from visualizar_seguimiento_equipos import procesar_y_limpiar_dataset

app = FastAPI()

# Permitir conexiones desde Streamlit Cloud o cualquier navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/procesar')
async def procesar_video(video: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, video.filename)

            # Guardar el vídeo subido localmente
            with open(video_path, 'wb') as buffer:
                shutil.copyfileobj(video.file, buffer)

            csv_raw = os.path.join(tmpdir, 'raw.csv')
            csv_filtrado = os.path.join(tmpdir, 'filtrado.csv')
            video_ia = os.path.join(tmpdir, 'ia.mp4')

            # 1. Ejecutar YOLOv8 en tu PC (Máxima velocidad)
            generar_dataset_deteccion(
                video_path,
                csv_raw,
                max_frames=1800
            )

            # 2. Limpieza de datos
            procesar_y_limpiar_dataset(
                video_path,
                csv_raw,
                video_ia,
                csv_filtrado
            )

            df = pd.read_csv(csv_filtrado)
            return JSONResponse(df.to_dict(orient='records'))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))