import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial import ConvexHull
import time
import tempfile
import os
import gc
import requests
import io

from extraccion_datos import generar_dataset_deteccion
import extraccion_datos
from visualizar_seguimiento_equipos import procesar_y_limpiar_dataset

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TacticalVision - Analítica Amateur Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS personalizados para un acabado oscuro profesional
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: #0f172a !important;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        color: #10b981;
    }
    .tactical-card {
        background-color: #1e293b;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    .tactical-card-rival {
        background-color: #1e293b;
        border-left: 5px solid #f43f5e;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


def calcular_distribucion_tercios(df, home_team, away_team):
    df_jugadores = df[df['class'] == 'player'].copy()
    
    # Coordenadas X en rango métrico real 0.0 a 105.0 metros
    home_x = df_jugadores[df_jugadores['team'] == 'home']['x']
    away_x = df_jugadores[df_jugadores['team'] == 'away']['x']
    
    TERCIO_1 = 105.0 / 3.0  # 35.0 m
    TERCIO_2 = TERCIO_1 * 2  # 70.0 m

    def get_pcts_home(series):
        if len(series) == 0: return [0, 0, 0]
        defensivo = (series < TERCIO_1).mean() * 100
        medio = ((series >= TERCIO_1) & (series <= TERCIO_2)).mean() * 100
        ofensivo = (series > TERCIO_2).mean() * 100
        return [round(defensivo), round(medio), round(ofensivo)]

    def get_pcts_away(series):
        if len(series) == 0: return [0, 0, 0]
        defensivo = (series > TERCIO_2).mean() * 100
        medio = ((series >= TERCIO_1) & (series <= TERCIO_2)).mean() * 100
        ofensivo = (series < TERCIO_1).mean() * 100
        return [round(defensivo), round(medio), round(ofensivo)]

    home_pcts = get_pcts_home(home_x)
    away_pcts = get_pcts_away(away_x)

    return pd.DataFrame({
        'Tercio del Campo': ['Tercio Defensivo (Propio)', 'Tercio Medio (Creación)', 'Tercio Ofensivo (Rival)'],
        f'{home_team} (%)': home_pcts,
        f'{away_team} (%)': away_pcts
    })

# ------------------------------------------------------------------------------
# 2. MOTOR DE SIMULACIÓN Y PROCESAMIENTO TÁCTICO
# ------------------------------------------------------------------------------

def compute_distances_and_metrics(df, min_id_duration_frames=3):
    df = df.copy()

    # 1. INTERPOLACIÓN DE BALÓN
    frames_totales = sorted(df['frame'].unique())
    ball_df = df[df['class'] == 'ball'][['frame', 'x', 'y']].drop_duplicates('frame')
    
    if not ball_df.empty:
        full_ball = pd.DataFrame({'frame': frames_totales}).merge(ball_df, on='frame', how='left')
        full_ball['x'] = full_ball['x'].interpolate(method='linear', limit=30)
        full_ball['y'] = full_ball['y'].interpolate(method='linear', limit=30)
        ball_dict = full_ball.dropna(subset=['x', 'y']).set_index('frame')[['x', 'y']].to_dict('index')
    else:
        ball_dict = {}

    # 2. FILTRADO DE JUGADORES Y DISTANCIAS
    player_counts = df[df['class'] == 'player']['id'].value_counts()
    valid_ids = player_counts[player_counts >= min_id_duration_frames].index
    player_mask = (df['class'] == 'player') & (df['id'].isin(valid_ids))

    if player_mask.sum() < 100:
        player_mask = (df['class'] == 'player')

    players = df[player_mask].copy().sort_values(['id', 'frame'])

    players['dx'] = players.groupby('id')['x'].diff()
    players['dy'] = players.groupby('id')['y'].diff()
    players['distancia_px'] = np.sqrt(players['dx'] ** 2 + players['dy'] ** 2)

    UMBRAL_SALTO = 8.0
    UMBRAL_RUIDO = 0.05

    players.loc[players['distancia_px'] > UMBRAL_SALTO, 'distancia_px'] = 0.0
    players.loc[players['distancia_px'] < UMBRAL_RUIDO, 'distancia_px'] = 0.0
    players['distancia_px'] = players['distancia_px'].fillna(0.0)

    K_PIXELS_A_METROS = 0.25
    FRAME_STRIDE = 3

    #players['distancia_m'] = players['distancia_px'] * K_PIXELS_A_METROS * FRAME_STRIDE
    players['distancia_m'] = players['distancia_px']
    df['distancia_m'] = 0.0
    df.loc[players.index, 'distancia_m'] = players['distancia_m']

    team_distances = players.groupby('team')['distancia_m'].sum().to_dict()
    player_distances = (
        players.groupby(['id', 'team'])['distancia_m']
        .sum()
        .reset_index()
        .rename(columns={'distancia_m': 'dist_meters'})
    )

    # 3. CENTROIDES
    player_data = df[df['class'] == 'player']
    centroids = player_data.groupby(['frame', 'team'])[['x', 'y']].mean().reset_index()

    home_cent = centroids[centroids['team'] == 'home'].rename(columns={'x': 'x_home', 'y': 'y_home'})
    away_cent = centroids[centroids['team'] == 'away'].rename(columns={'x': 'x_away', 'y': 'y_away'})

    inter_df = pd.merge(home_cent, away_cent, on='frame')
    inter_df['inter_distance'] = np.sqrt(
        (inter_df['x_home'] - inter_df['x_away']) ** 2 +
        (inter_df['y_home'] - inter_df['y_away']) ** 2
    )

    # 4. POSESIÓN FRAME A FRAME
    UMBRAL_CONTACTO = 15.0
    UMBRAL_INERCIA = 25.0
    MAX_FRAMES_INERCIA = 25

    poseedor_actual_id = None
    poseedor_actual_team = None
    frames_inercia_restantes = 0

    distancias_debug = []
    ramas_debug = []

    for f in frames_totales:
        players_f = df[
            (df['frame'] == f) & 
            (df['team'].isin(['home', 'away'])) & 
            (df['class'] == 'player')
        ].copy()

        if players_f.empty:
            ramas_debug.append({
                'frame': f, 'jugador_id': None, 'equipo_cercano': None, 'distancia': None,
                'poseedor_actual': poseedor_actual_team, 'rama': 'DISPUTED', 'decision_final': 'disputed'
            })
            continue

        ball_pos = ball_dict.get(f)

        if ball_pos is None:
            if poseedor_actual_team is not None and frames_inercia_restantes > 0:
                rama = 'BALON_PERDIDO_INERCIA'
                decision_final = poseedor_actual_team
                frames_inercia_restantes -= 1
            else:
                rama = 'DISPUTED'
                decision_final = 'disputed'
                poseedor_actual_team = None
                poseedor_actual_id = None
                frames_inercia_restantes = 0

            ramas_debug.append({
                'frame': f, 'jugador_id': poseedor_actual_id, 'equipo_cercano': None, 'distancia': None,
                'poseedor_actual': poseedor_actual_team, 'rama': rama, 'decision_final': decision_final,
                'umbral_contacto': UMBRAL_CONTACTO, 'umbral_inercia': UMBRAL_INERCIA
            })
            continue

        bx, by = ball_pos['x'], ball_pos['y']
        players_f['distancia'] = np.sqrt((players_f['x'] - bx) ** 2 + (players_f['y'] - by) ** 2)
        closest = players_f.loc[players_f['distancia'].idxmin()]

        jugador_id = closest['id']
        equipo_cercano = closest['team']
        distancia_minima = float(closest['distancia'])

        distancias_debug.append({
            'frame': f, 'id': jugador_id, 'team': equipo_cercano, 'distancia': distancia_minima
        })

        if distancia_minima <= UMBRAL_CONTACTO:
            if poseedor_actual_team == equipo_cercano:
                rama = 'MANTENIMIENTO'
            elif poseedor_actual_team is not None:
                rama = 'CAMBIO_EQUIPO'
            else:
                rama = 'ADQUISICION'

            poseedor_actual_id = jugador_id
            poseedor_actual_team = equipo_cercano
            frames_inercia_restantes = MAX_FRAMES_INERCIA
            decision_final = equipo_cercano

        elif distancia_minima <= UMBRAL_INERCIA:
            if poseedor_actual_team is not None and frames_inercia_restantes > 0:
                rama = 'INERCIA'
                decision_final = poseedor_actual_team
                frames_inercia_restantes -= 1
            else:
                rama = 'ADQUISICION_INICIAL'
                poseedor_actual_id = jugador_id
                poseedor_actual_team = equipo_cercano
                frames_inercia_restantes = MAX_FRAMES_INERCIA
                decision_final = equipo_cercano

        else:
            if poseedor_actual_team is not None and frames_inercia_restantes > 0:
                rama = 'INERCIA_DECAIMIENTO'
                decision_final = poseedor_actual_team
                frames_inercia_restantes -= 1
            else:
                rama = 'DISPUTED'
                decision_final = 'disputed'
                poseedor_actual_team = None
                poseedor_actual_id = None
                frames_inercia_restantes = 0

        ramas_debug.append({
            'frame': f, 'jugador_id': jugador_id, 'equipo_cercano': equipo_cercano, 'distancia': distancia_minima,
            'poseedor_actual': poseedor_actual_team, 'rama': rama, 'decision_final': decision_final,
            'umbral_contacto': UMBRAL_CONTACTO, 'umbral_inercia': UMBRAL_INERCIA
        })

    # 5. AJUSTES RETROACTIVOS Y FINALES
    primer_poseedor = next((item['decision_final'] for item in ramas_debug if item['decision_final'] in ['home', 'away']), None)
    if primer_poseedor:
        for item in ramas_debug:
            if item['decision_final'] == 'disputed' and item['poseedor_actual'] is None:
                item['rama'] = 'INICIALIZACION_RETROACTIVA'
                item['decision_final'] = primer_poseedor
            else:
                break

    ultimo_poseedor = None
    for item in ramas_debug:
        if item['decision_final'] in ['home', 'away']:
            ultimo_poseedor = item['decision_final']
        elif item['decision_final'] == 'disputed' and ultimo_poseedor:
            item['rama'] = 'PROPAGACION_FINAL'
            item['decision_final'] = ultimo_poseedor

    indices_cambio = []
    for i in range(1, len(ramas_debug)):
        prev = ramas_debug[i-1]['decision_final']
        curr = ramas_debug[i]['decision_final']
        if prev in ['home', 'away'] and curr in ['home', 'away'] and prev != curr:
            indices_cambio.append(i)

    for idx in indices_cambio:
        equipo_nuevo = ramas_debug[idx]['decision_final']
        for retro in range(1, 25):
            idx_prev = idx - retro
            if idx_prev >= 0 and ramas_debug[idx_prev]['rama'] in ['INERCIA', 'BALON_PERDIDO_INERCIA', 'INERCIA_DECAIMIENTO']:
                ramas_debug[idx_prev]['decision_final'] = equipo_nuevo
                ramas_debug[idx_prev]['rama'] = 'TRANSICION_VUELO'
            else:
                break

    # 6. CÁLCULO FINAL DE MÉTRICAS Y PORCENTAJES
    decision_series = pd.Series([item['decision_final'] for item in ramas_debug])
    counts = decision_series.value_counts().to_dict()

    home_count = counts.get('home', 0)
    away_count = counts.get('away', 0)
    disputed_count = counts.get('disputed', 0)

    total_efectivo = home_count + away_count

    if total_efectivo > 0:
        poss_home = round(100 * home_count / total_efectivo)
        poss_away = round(100 * away_count / total_efectivo)
    else:
        poss_home = 0
        poss_away = 0

    # 7. DISTRIBUCIÓN TERRITORIAL
    home_name = st.session_state.get('home_team', 'Local')
    away_name = st.session_state.get('away_team', 'Visitante')
    
    zone_df = calcular_distribucion_tercios(df, home_name, away_name)

    return {
        'player_distances': player_distances,
        'team_distances': team_distances,
        'centroids': centroids,
        'inter_df': inter_df,
        'zone_df': zone_df,
        'poss_home': poss_home,
        'poss_away': poss_away,
        'possession_home_count': home_count,
        'possession_away_count': away_count,
        'possession_disputed_count': disputed_count,
        'possession_total_valid': total_efectivo,
        'decision_df': pd.DataFrame(ramas_debug),
        'distancias_debug': pd.DataFrame(distancias_debug),
        'ramas_df': pd.DataFrame(ramas_debug)
    }

# ------------------------------------------------------------------------------
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# ------------------------------------------------------------------------------
st.sidebar.title("Panel de Control")

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.df = None
    st.session_state.metrics = None

if not st.session_state.processed:
    st.title("TacticalVision")
    st.caption("Plataforma Táctica de Análisis Telemétrico para Fútbol Base y Amateur")

    col_main_left, col_main_right = st.columns([2, 1])

    with col_main_left:
        st.markdown("### Entrada de Vídeo del Partido")
        uploaded_file = st.file_uploader("Arrastra o selecciona el archivo de vídeo del partido (.mp4, .mov)", type=['mp4', 'mov'])
        st.info("Soporta grabaciones de cámaras tácticas elevadas, gran angular o clips descargados.")

        st.markdown("---")
        st.markdown("### Configuración de Equipos")
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            home_team_input = st.text_input("Equipo Local (Principal)", "C.F. Damm")
        with sub_col2:
            away_team_input = st.text_input("Equipo Rival (Visitante)", "U.E. Sant Andreu")

    with col_main_right:
        st.markdown("### Metodología de Análisis")
        st.info("El sistema procesa el vídeo en dos pasadas: extracción de posiciones mediante detección multiobjeto con calibración de Homografía 2D y consolidación de métricas de rendimiento físico-táctico.")

        st.markdown("###")
        run_button = st.button("Ejecutar Pipeline Táctico", use_container_width=True)

    if run_button:
        st.session_state.pop('pipeline_error', None)
        if uploaded_file is None:
            st.error("Por favor, selecciona un archivo de vídeo (.mp4 o .mov) para iniciar el análisis.")
        else:
            st.session_state.home_team = home_team_input
            st.session_state.away_team = away_team_input
            st.session_state.home_color = "#10b981"  # Verde Esmeralda
            st.session_state.away_color = "#f43f5e"  # Rosa Coral
            
            with st.status("Procesando vídeo en el servidor local de análisis...", expanded=True) as status:
                try:
                    st.write("1. Transfiriendo vídeo al motor de análisis YOLOv8...")

                    files = {
                        'video': (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or 'video/mp4'
                        )
                    }

                    BASE_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
                    INIT_URL = f"{BASE_URL}/procesar"
                    headers = {"ngrok-skip-browser-warning": "true"}

                    # Aumentamos el timeout a 1 hora para permitir que finalice la ejecución sincrónica de YOLOv8
                    response = requests.post(INIT_URL, files=files, headers=headers, timeout=3600)

                    if response.status_code != 200:
                        st.error(f"Error backend ({response.status_code}): {response.text}")
                        st.stop()

                    json_data = response.json()

                    # 1. Validación e Ingesta del JSON directo
                    if isinstance(json_data, dict):
                        if 'error' in json_data or 'detail' in json_data:
                            st.error(f"El backend reportó un error: {json_data.get('error') or json_data.get('detail')}")
                            st.stop()
                        
                        # Extraer la lista de datos si viene dentro de una clave wrapper
                        if 'data' in json_data:
                            raw_df = pd.DataFrame(json_data['data'])
                        elif 'result' in json_data:
                            raw_df = pd.DataFrame(json_data['result'])
                        elif 'detections' in json_data:
                            raw_df = pd.DataFrame(json_data['detections'])
                        else:
                            try:
                                raw_df = pd.DataFrame(json_data)
                            except ValueError:
                                raw_df = pd.DataFrame([json_data])
                    elif isinstance(json_data, list):
                        raw_df = pd.DataFrame(json_data)
                    else:
                        st.error("Formato JSON no reconocido devuelto por el servidor.")
                        st.stop()

                    if raw_df.empty:
                        st.error("El backend devolvió un DataFrame vacío.")
                        st.stop()

                    gc.collect()

                    # 2. RENOMBRADO FLEXIBLE Y NORMALIZACIÓN DE COLUMNAS
                    column_map = {
                        'id_jugador': 'id', 'player_id': 'id', 'track_id': 'id',
                        'rol_equipo': 'team', 'equipo': 'team', 'team_name': 'team',
                        'pos_x': 'x', 'x_pos': 'x', 'X': 'x', 'centroid_x': 'x',
                        'pos_y': 'y', 'y_pos': 'y', 'Y': 'y', 'centroid_y': 'y'
                    }
                    raw_df = raw_df.rename(columns=column_map)

                    # Si 'task_id' venía como una columna en el DataFrame, la eliminamos
                    if 'task_id' in raw_df.columns and len(raw_df.columns) > 1:
                        raw_df = raw_df.drop(columns=['task_id'], errors='ignore')

                    if 'x' not in raw_df.columns or 'y' not in raw_df.columns:
                        st.error(f"No se encontraron las coordenadas 'x' e 'y'. Columnas recibidas: {list(raw_df.columns)}")
                        st.stop()

                    if 'id' not in raw_df.columns:
                        raw_df['id'] = 0
                    if 'team' not in raw_df.columns:
                        raw_df['team'] = 'home'

                    raw_df['class'] = 'player'

                    # 3. Procesar filas de balón si existen
                    balon_x_col = next((c for c in ['balon_x', 'ball_x', 'b_x'] if c in raw_df.columns), None)
                    balon_y_col = next((c for c in ['balon_y', 'ball_y', 'b_y'] if c in raw_df.columns), None)

                    if balon_x_col and balon_y_col:
                        ball_rows = raw_df[
                            (raw_df[balon_x_col].notna()) &
                            (raw_df[balon_y_col].notna()) &
                            (raw_df[balon_x_col] != -1) &
                            (raw_df[balon_y_col] != -1)
                        ][['frame', balon_x_col, balon_y_col]].drop_duplicates()

                        if not ball_rows.empty:
                            ball_rows = ball_rows.rename(columns={balon_x_col: 'x', balon_y_col: 'y'})
                            ball_rows['id'] = 9999
                            ball_rows['team'] = 'ball'
                            ball_rows['class'] = 'ball'
                            ball_rows['bbox'] = 'ball'

                            raw_df = pd.concat([raw_df, ball_rows], ignore_index=True)

                    # 4. ESCALADO MÉTRICO INTELIGENTE (0..105m x 0..68m)
                    FIELD_LENGTH = 105.0
                    FIELD_WIDTH = 68.0

                    raw_df['x'] = pd.to_numeric(raw_df['x'], errors='coerce').fillna(0)
                    raw_df['y'] = pd.to_numeric(raw_df['y'], errors='coerce').fillna(0)

                    max_x = raw_df['x'].max()
                    max_y = raw_df['y'].max()

                    if max_x <= 1.0 and max_y <= 1.0 and max_x > 0:
                        raw_df['x'] = raw_df['x'] * FIELD_LENGTH
                        raw_df['y'] = raw_df['y'] * FIELD_WIDTH
                    elif max_x > FIELD_LENGTH or max_y > FIELD_WIDTH:
                        raw_df['x'] = (raw_df['x'] / max_x) * FIELD_LENGTH
                        raw_df['y'] = (raw_df['y'] / max_y) * FIELD_WIDTH
                    elif max_x < FIELD_LENGTH and max_x > 1.0:
                        raw_df['x'] = (raw_df['x'] / max_x) * FIELD_LENGTH
                        if max_y > 0:
                            raw_df['y'] = (raw_df['y'] / max_y) * FIELD_WIDTH

                    raw_df['x'] = raw_df['x'].clip(0, FIELD_LENGTH)
                    raw_df['y'] = raw_df['y'].clip(0, FIELD_WIDTH)

                    # 5. Normalizar nombres de equipos
                    raw_df['team'] = raw_df['team'].replace({
                        'Equipo_1': 'home',
                        'Equipo_2': 'away',
                        'equipo_1': 'home',
                        'equipo_2': 'away'
                    })

                    # 6. Calcular métricas finales
                    metrics = compute_distances_and_metrics(raw_df)

                    st.session_state.df = raw_df
                    st.session_state.metrics = metrics
                    st.session_state.processed = True

                    status.update(label="Análisis completado con éxito", state="complete")
                    st.rerun()

                except Exception as e:
                    status.update(label="Error en el pipeline", state="error")
                    import traceback
                    error_text = traceback.format_exc()
                    st.error("EXCEPCIÓN REAL DEL PIPELINE")
                    st.code(error_text)
                    raise e

else:
    # Cargar variables guardadas en sesión
    home_team = st.session_state.home_team
    away_team = st.session_state.away_team
    home_color = st.session_state.home_color
    away_color = st.session_state.away_color
    df = st.session_state.df
    metrics = st.session_state.metrics

    # Encabezado principal del Informe Telemétrico
    st.title("TacticalVision")
    st.caption(f"Informe Telemétrico Activo: **{home_team}** vs **{away_team}**")

    # MÉTRICAS CLAVE SUPERIORES
    dist_home = metrics['team_distances'].get('home', 0.0)
    dist_away = metrics['team_distances'].get('away', 0.0)
    avg_inter_dist = metrics['inter_df']['inter_distance'].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(f"Posesión {home_team}", f"{metrics['poss_home']}%")
    with m2:
        st.metric(f"Posesión {away_team}", f"{metrics['poss_away']}%")
    with m3:
        st.metric(f"Dist. Total {home_team}", f"{dist_home:.1f} m")
    with m4:
        st.metric(f"Dist. Total {away_team}", f"{dist_away:.1f} m")
    with m5:
        st.metric("Dist. Inter-Centroides", f"{avg_inter_dist:.1f} m")

    # PESTAÑAS DE NAVEGACIÓN
    tab_projection, tab_heatmap, tab_stats = st.tabs([
        "Proyección 2D & Centroides", 
        "Mapas de Calor (Equipos y Balón)", 
        "Métricas Físicas y Tácticas"
    ])

    # --------------------------------------------------------------------------
    # PESTAÑA 1: PROYECCIÓN 2D, CENTROIDES Y POLÍGONOS
    # --------------------------------------------------------------------------
    with tab_projection:
        st.subheader("Plano Métrico 2D Interactivo (Homografía Rectificada)")
        
        frames_disponibles = sorted(df['frame'].unique())

        selected_frame = st.select_slider(
            'Seleccionar instante (fotograma):',
            options=frames_disponibles,
            value=frames_disponibles[len(frames_disponibles)//2]
        )

        frame_df = df[df['frame'] == selected_frame]

        st.write('Fotograma mostrado:', selected_frame)
        st.write('Jugadores en frame:', len(frame_df[frame_df['class']=='player']))
        
        col_map, col_pedagogic = st.columns([2, 1])
        
        with col_map:
            fig, ax = plt.subplots(figsize=(10, 6.8))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1f7a1f')

            # Campo verde
            ax.add_patch(patches.Rectangle((0, 0), 105, 68, facecolor='#1f7a1f', edgecolor='white', lw=2))
            ax.plot([52.5, 52.5], [0, 68], color='white', lw=2)
            ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, color='white', lw=2))

            ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=2))
            ax.add_patch(patches.Rectangle((105-16.5, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=2))

            ax.add_patch(patches.Rectangle((0, 24.84), 5.5, 18.32, fill=False, edgecolor='white', lw=2))
            ax.add_patch(patches.Rectangle((105-5.5, 24.84), 5.5, 18.32, fill=False, edgecolor='white', lw=2))

            # Porterías
            ax.plot([-2, 0], [30.34, 30.34], color='white', lw=3)
            ax.plot([-2, 0], [37.66, 37.66], color='white', lw=3)
            ax.plot([-2, -2], [30.34, 37.66], color='white', lw=3)

            ax.plot([105, 107], [30.34, 30.34], color='white', lw=3)
            ax.plot([105, 107], [37.66, 37.66], color='white', lw=3)
            ax.plot([107, 107], [30.34, 37.66], color='white', lw=3)

            ax.set_xlim(-3, 108)
            ax.set_ylim(68, 0)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Jugadores Locales
            home_players = frame_df[frame_df['team'] == 'home']
            ax.scatter(home_players['x'], home_players['y'],
                        color=home_color, s=90,
                        edgecolor='white', linewidth=1.2,
                        label=home_team, zorder=5)
            for _, row in home_players.iterrows():
                ax.text(row['x'], row['y'], str(int(row['id'])), color='white', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
                
            # Jugadores Visitantes
            away_players = frame_df[frame_df['team'] == 'away']
            ax.scatter(away_players['x'], away_players['y'],
                        color=away_color, s=90,
                        edgecolor='white', linewidth=1.2,
                        label=away_team, zorder=5)
            for _, row in away_players.iterrows():
                ax.text(row['x'], row['y'], str(int(row['id'])), color='white', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
                
            # Balón
            ball_df = frame_df[frame_df['team'] == 'ball']
            if not ball_df.empty:
                ax.scatter(ball_df['x'], ball_df['y'], color='#fbbf24', s=90, edgecolor='black', linewidth=1.5, label="Balón", zorder=7)
            
            # CENTROIDES
            if not home_players.empty:
                c_home_x, c_home_y = home_players['x'].mean(), home_players['y'].mean()
                ax.scatter(c_home_x, c_home_y, color=home_color, s=300, marker='*', edgecolor='white', linewidth=1.5, label=f"Centroide {home_team}", zorder=8)
                
            if not away_players.empty:
                c_away_x, c_away_y = away_players['x'].mean(), away_players['y'].mean()
                ax.scatter(c_away_x, c_away_y, color=away_color, s=300, marker='*', edgecolor='white', linewidth=1.5, label=f"Centroide {away_team}", zorder=8)
                
            # Línea Inter-Centroides
            if not home_players.empty and not away_players.empty:
                ax.plot([c_home_x, c_away_x], [c_home_y, c_away_y], color='#94a3b8', linestyle=':', linewidth=2, zorder=4)
            
            # Polígono Convex Hull
            if len(home_players) > 3:
                try:
                    points = home_players[['x', 'y']].values
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], color=home_color, linestyle='--', alpha=0.5, zorder=4)
                except Exception:
                    pass
                    
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, facecolor='#1e293b')
            st.pyplot(fig, clear_figure=True)
            
        with col_pedagogic:
            st.subheader("Análisis Táctico de Centroides")
            
            inter_frame = metrics['inter_df'][metrics['inter_df']['frame'] == selected_frame]
            curr_inter_dist = inter_frame['inter_distance'].values[0] if not inter_frame.empty else 0.0

            st.markdown(f"""
            <div class="tactical-card">
                <h4>Centroide y Distancia Inter-Equipo</h4>
                <p>Las estrellas (★) representan el <b>Centro de Masas Posicional</b> de cada plantilla en este instante.</p>
                <ul>
                    <li><b>Distancia Inter-Centroides actual:</b> <code>{curr_inter_dist:.2f} m</code></li>
                    <li><b>Efecto Táctico:</b> Una distancia inter-equipo reducida (&lt; 20m) indica una fase de alta presión o bloqueo denso. Una distancia amplia (&gt; 30m) evidencia estiramiento entre líneas o transición limpia.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 2: MAPAS DE CALOR (EQUIPOS Y BALÓN)
    # --------------------------------------------------------------------------
    with tab_heatmap:
        st.subheader("Mapas de Densidad de Ocupación Terrenal (KDE)")
        
        heatmap_choice = st.radio(
            "Seleccionar elemento para visualizar densidad:", 
            (home_team, away_team, "Balón")
        )
        
        col_heat, col_heat_info = st.columns([2, 1])
        
        with col_heat:
            fig_heat, ax_heat = plt.subplots(figsize=(10, 6.8))
            fig_heat.patch.set_facecolor('#0f172a')
            ax_heat.set_facecolor('#1f7a1f')

            ax_heat.add_patch(patches.Rectangle((0, 0), 105, 68, facecolor='#1f7a1f', edgecolor='white', lw=2))
            ax_heat.plot([52.5, 52.5], [0, 68], color='white', lw=2)
            ax_heat.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, color='white', lw=2))

            ax_heat.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=2))
            ax_heat.add_patch(patches.Rectangle((105-16.5, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=2))

            ax_heat.set_xlim(0, 105)
            ax_heat.set_ylim(68, 0)
            ax_heat.set_aspect('equal')
            ax_heat.axis('off')
            
            if heatmap_choice == home_team:
                team_data = df[df['team'] == 'home']
                color_theme = "mako"
            elif heatmap_choice == away_team:
                team_data = df[df['team'] == 'away']
                color_theme = "rocket"
            else:
                ball_data = df[df['class'] == 'ball'].copy()
                ball_data['prev_x'] = ball_data['x'].shift(1)
                ball_data['prev_y'] = ball_data['y'].shift(1)
                ball_data['speed'] = np.sqrt((ball_data['x'] - ball_data['prev_x'])**2 + (ball_data['y'] - ball_data['prev_y'])**2)
                team_data = ball_data[ball_data['speed'] > 0.15]
                color_theme = "flare"
            
            if not team_data.empty and len(team_data) > 5:
                sns.kdeplot(
                    data=team_data,
                    x='x',
                    y='y',
                    fill=True,
                    cmap=color_theme,
                    alpha=0.55,
                    bw_adjust=0.5,
                    levels=40,
                    thresh=0.02,
                    ax=ax_heat,
                    zorder=3
                )
            else:
                st.warning("Datos insuficientes para generar el mapa de calor en esta categoría.")
                
            st.pyplot(fig_heat, clear_figure=True)
            
        with col_heat_info:
            st.subheader("Interpretación Telemétrica")
            
            st.markdown(f"""
            <div class="tactical-card">
                <h4>Análisis Posicional: {heatmap_choice}</h4>
                <p>Muestra los núcleos de presencia constante de los jugadores a lo largo del clip.</p>
                <ul>
                    <li><b>Saturación en Ocupación:</b> Las zonas más brillantes señalan las zonas donde más presencia tienen los equipos.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 3: MÉTRICAS FÍSICAS Y TÁCTICAS
    # --------------------------------------------------------------------------
    with tab_stats:
        st.subheader("Rendimiento Físico e Inter-Evolución Táctica")

        st.markdown("#### Evolución de la Distancia Inter-Centroides (m)")

        inter_df = metrics['inter_df']
        
        fig_lines, ax_lines = plt.subplots(figsize=(12, 4))
        fig_lines.patch.set_facecolor('#1e293b')
        ax_lines.set_facecolor('#0f172a')

        ax_lines.plot(inter_df['frame'], inter_df['inter_distance'], color='#10b981', linewidth=2.5, label="Distancia entre Centroides")
        ax_lines.axhline(inter_df['inter_distance'].mean(), color='#fbbf24', linestyle='--', label=f"Promedio: {inter_df['inter_distance'].mean():.1f}m")

        ax_lines.set_xlabel("Fotograma / Frame", color='#94a3b8')
        ax_lines.set_ylabel("Metros (m)", color='#94a3b8')
        ax_lines.tick_params(colors='#94a3b8')
        ax_lines.grid(True, linestyle=':', alpha=0.3, color='#475569')
        ax_lines.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')

        st.pyplot(fig_lines, use_container_width=True)

        st.markdown("---")
        
        st.subheader("Distribución Territorial del Control de Juego")
        zone_col1, zone_col2 = st.columns([1, 1.2])
        
        with zone_col1:
            zone_df = metrics.get('zone_df', pd.DataFrame()) 
            st.dataframe(zone_df, use_container_width=True, hide_index=True)
            
        with zone_col2:
            st.info("""
            **Guía de Interpretación Táctica para el Entrenador:**

            * **Distancia Inter-Centroides:** Mide la separación en metros entre el centro de gravedad del equipo local y el del visitante.
            * **Distribución Territorial por Tercios:** Identifica el porcentaje de presencia de cada plantilla en defensivo, medio y ofensivo.
            """)

    # Botón de reinicio
    st.markdown("---")
    col_reset_btn, _ = st.columns([1, 2])
    with col_reset_btn:
        if st.button("Cargar Nuevo Vídeo / Reiniciar", use_container_width=True):
            st.session_state.processed = False
            st.session_state.df = None
            st.session_state.metrics = None
            st.rerun()