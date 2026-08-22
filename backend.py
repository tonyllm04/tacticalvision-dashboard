from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
import pandas as pd

from extraccion_datos import generar_dataset_deteccion
from visualizar_seguimiento_equipos import procesar_y_limpiar_dataset

app = FastAPI()

@app.post('/procesar')
async def procesar_video(video: UploadFile = File(...)):

    with tempfile.TemporaryDirectory() as tmpdir:

        video_path = os.path.join(tmpdir, video.filename)

        with open(video_path, 'wb') as buffer:
            shutil.copyfileobj(video.file, buffer)

        csv_raw = os.path.join(tmpdir, 'raw.csv')
        csv_filtrado = os.path.join(tmpdir, 'filtrado.csv')
        video_ia = os.path.join(tmpdir, 'ia.mp4')

        generar_dataset_deteccion(
            video_path,
            csv_raw,
            max_frames=1800
        )

        procesar_y_limpiar_dataset(
            video_path,
            csv_raw,
            video_ia,
            csv_filtrado
        )

        df = pd.read_csv(csv_filtrado)

        return JSONResponse(df.to_dict(orient='records'))