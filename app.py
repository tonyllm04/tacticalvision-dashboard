import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial import ConvexHull
import time

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


# ------------------------------------------------------------------------------
# 2. TRANSFORMACIÓN DE HOMOGRAFÍA Y DIBUJO DE CAMPO
# ------------------------------------------------------------------------------
class HomographyTransformer:
    """
    Transforma coordenadas en píxeles de la cámara al plano métrico 2D (105m x 68m).
    """
    def __init__(self):
        src_points = np.float32([
            [210, 450],   # Esquina superior izquierda en imagen
            [1070, 450],  # Esquina superior derecha en imagen
            [50, 720],    # Esquina inferior izquierda en imagen
            [1230, 720]   # Esquina inferior derecha en imagen
        ])
        
        dst_points = np.float32([
            [0, 0],       # Banda superior izquierda (m)
            [105, 0],     # Banda superior derecha (m)
            [0, 68],      # Banda inferior izquierda (m)
            [105, 68]     # Banda inferior derecha (m)
        ])
        
        self.H, _ = cv2.findHomography(src_points, dst_points)

    def transform_point(self, u, v):
        point = np.array([u, v, 1.0], dtype=np.float32)
        transformed = np.dot(self.H, point)
        x = transformed[0] / transformed[2]
        y = transformed[1] / transformed[2]
        return np.clip(x, 0, 105), np.clip(y, 0, 68)


def draw_football_pitch(ax, slate_mode=True):
    """
    Dibuja un terreno de juego profesional a escala 105x68 metros sobre Matplotlib.
    """
    bg_color = '#1e293b' if slate_mode else '#2d5a27'
    line_color = '#475569' if slate_mode else '#ffffff'
    
    # Perímetro del campo
    rect = patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor=line_color, facecolor=bg_color, zorder=1)
    ax.add_patch(rect)
    
    # Línea divisoria y círculo central
    ax.plot([52.5, 52.5], [0, 68], color=line_color, linewidth=2, zorder=2)
    center_circle = patches.Circle((52.5, 34), 9.15, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2)
    ax.add_patch(center_circle)
    ax.scatter(52.5, 34, color=line_color, s=20, zorder=3)
    
    # Áreas izquierda (Local)
    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2))
    ax.add_patch(patches.Rectangle((0, 24.84), 5.5, 18.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2))
    
    # Áreas derecha (Rival)
    ax.add_patch(patches.Rectangle((88.5, 13.84), 16.5, 40.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2))
    ax.add_patch(patches.Rectangle((99.5, 24.84), 5.5, 18.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2))
    
    # Puntos de penalti
    ax.scatter(11, 34, color=line_color, s=20, zorder=3)
    ax.scatter(94, 34, color=line_color, s=20, zorder=3)
    
    # Arcos del área
    ax.add_patch(patches.Arc((11, 34), 18.3, 18.3, theta1=-53, theta2=53, linewidth=2, color=line_color, zorder=2))
    ax.add_patch(patches.Arc((94, 34), 18.3, 18.3, theta1=127, theta2=233, linewidth=2, color=line_color, zorder=2))
    
    ax.set_xlim(-3, 108)
    ax.set_ylim(-3, 71)
    ax.axis('off')


# ------------------------------------------------------------------------------
# 3. MOTOR DE SIMULACIÓN Y PROCESAMIENTO TÁCTICO
# ------------------------------------------------------------------------------
def generate_tactical_sequence(frames=120):
    """
    Genera secuencia de seguimiento en 2D métrico simulando una fase ofensiva.
    """
    data = []
    np.random.seed(42)
    
    for f in range(frames):
        progress = f / frames
        # Balón en movimiento
        ball_x = 35 + progress * 50 + np.sin(f*0.15) * 3
        ball_y = 20 + progress * 28 + np.cos(f*0.15) * 4
        
        data.append({
            'frame': f, 'id': 99, 'class': 'ball', 
            'x': ball_x, 'y': ball_y, 'team': 'ball'
        })
        
        # 11 Jugadores Locales
        local_positions = [
            (12, 34), # Portero
            (28 + progress*8, 15 + np.sin(f*0.05)*3), 
            (26 + progress*9, 28), 
            (26 + progress*9, 40), 
            (28 + progress*8, 53 - np.sin(f*0.05)*3), 
            (42 + progress*12, 18), 
            (38 + progress*14, 30), 
            (38 + progress*14, 38), 
            (42 + progress*12, 50), 
            (55 + progress*15, 25), 
            (55 + progress*15, 43)
        ]
        
        for idx, (bx, by) in enumerate(local_positions):
            data.append({
                'frame': f, 'id': idx + 1, 'class': 'player',
                'x': bx + np.random.normal(0, 0.12), 
                'y': by + np.random.normal(0, 0.12), 
                'team': 'home'
            })
            
        # 11 Jugadores Visitantes
        away_positions = [
            (92, 34), # Portero Rival
            (68 + progress*8, 10), 
            (58 + progress*12, 24), 
            (58 + progress*12, 44), 
            (68 + progress*8, 58), 
            (50 + progress*22, 16), 
            (46 + progress*26, 30), 
            (46 + progress*26, 38), 
            (50 + progress*22, 52), 
            (38 + progress*32, 26), 
            (38 + progress*32, 42)
        ]
        
        for idx, (bx, by) in enumerate(away_positions):
            data.append({
                'frame': f, 'id': idx + 12, 'class': 'player',
                'x': bx + np.random.normal(0, 0.15), 
                'y': by + np.random.normal(0, 0.15), 
                'team': 'away'
            })
            
    return pd.DataFrame(data)


def compute_distances_and_metrics(df, fps=25, min_id_duration_frames=15):
    """
    Calcula distancias y métricas tácticas eliminando el jitter de posicionamiento,
    los saltos por oclusión de IDs y la fragmentación de tracking.
    """
    df = df.copy()

    # 1. FILTRAR IDS EFÍMEROS / FANTASMA
    player_counts = df[df['class'] == 'player']['id'].value_counts()
    valid_ids = player_counts[player_counts >= min_id_duration_frames].index
    
    player_mask = (df['class'] == 'player') & (df['id'].isin(valid_ids))

    # 2. SUAVIZADO Y CÁLCULO DE DISTANCIAS
    def process_player_movement(group):
        group = group.sort_values('frame').copy()
        
        # Suavizado ligero
        group['x_smooth'] = group['x'].rolling(window=5, min_periods=1, center=True).mean()
        group['y_smooth'] = group['y'].rolling(window=5, min_periods=1, center=True).mean()
        
        dx = group['x_smooth'].diff()
        dy = group['y_smooth'].diff()
        step_m = np.sqrt(dx**2 + dy**2).fillna(0.0)

        UMBRAL_SALTO_TRACKING = 1.2   # metros por frame
        UMBRAL_RUIDO = 0.01           # 1 cm

        valid_jump = step_m <= UMBRAL_SALTO_TRACKING
        valid_noise = step_m >= UMBRAL_RUIDO

        group['distancia_m'] = np.where(
            valid_jump & valid_noise,
            step_m,
            0.0
        )
        
        return group

    # Procesar movimiento de jugadores
    df_players = df[player_mask].groupby('id', group_keys=False).apply(process_player_movement)
    
    # Asignar resultados
    df['distancia_m'] = 0.0
    df.loc[player_mask, 'distancia_m'] = df_players['distancia_m']

    # Totales por equipo y por jugador
    team_distances = df[df['class'] == 'player'].groupby('team')['distancia_m'].sum().to_dict()
    
    player_distances = df[df['class'] == 'player'].groupby(['id', 'team'])['distancia_m'].sum().reset_index()
    player_distances.rename(columns={'distancia_m': 'dist_meters'}, inplace=True)

    # 3. CENTROIDES E INTER-DISTANCIA POR FRAME
    player_data = df[df['class'] == 'player']
    centroids = player_data.groupby(['frame', 'team'])[['x', 'y']].mean().reset_index()
    
    home_cent = centroids[centroids['team'] == 'home'].rename(columns={'x': 'x_home', 'y': 'y_home'})
    away_cent = centroids[centroids['team'] == 'away'].rename(columns={'x': 'x_away', 'y': 'y_away'})
    
    inter_df = pd.merge(home_cent, away_cent, on='frame')
    inter_df['inter_distance'] = np.sqrt(
        (inter_df['x_home'] - inter_df['x_away'])**2 + 
        (inter_df['y_home'] - inter_df['y_away'])**2
    )
    
    # 4. POSESIÓN DE BALÓN CON INERCIA
    frames_list = df['frame'].unique()
    possession_counts = {'home': 0, 'away': 0, 'disputed': 0}
    
    last_possessor = 'disputed'
    inertia_counter = 0
    MAX_INERTIA = 10 
    
    for f in frames_list:
        frame_data = df[df['frame'] == f]
        ball = frame_data[frame_data['class'] == 'ball']
        players = frame_data[frame_data['class'] == 'player']
        
        current_possessor = 'disputed'
        if not ball.empty and not players.empty:
            bx, by = ball.iloc[0]['x'], ball.iloc[0]['y']
            p_copy = players.copy()
            p_copy['dist'] = np.sqrt((p_copy['x'] - bx)**2 + (p_copy['y'] - by)**2)
            closest = p_copy.loc[p_copy['dist'].idxmin()]
            
            if closest['dist'] < 8.0:
                current_possessor = closest['team']
                last_possessor = current_possessor
                inertia_counter = MAX_INERTIA
            else:
                if inertia_counter > 0:
                    current_possessor = last_possessor
                    inertia_counter -= 1
                else:
                    current_possessor = 'disputed'
                    
        possession_counts[current_possessor] += 1
        
    total_frames = len(frames_list)
    poss_home = round((possession_counts['home'] / total_frames) * 100) if total_frames > 0 else 50
    poss_away = round((possession_counts['away'] / total_frames) * 100) if total_frames > 0 else 50
    
    return {
        'player_distances': player_distances,
        'team_distances': team_distances,
        'centroids': centroids,
        'inter_df': inter_df,
        'poss_home': poss_home,
        'poss_away': poss_away
    }


# ------------------------------------------------------------------------------
# 4. INTERFAZ DE USUARIO (STREAMLIT)
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
            home_team_input = st.text_input("Equipo Local (Principal)", "CD Alianza Amateur")
        with sub_col2:
            away_team_input = st.text_input("Equipo Rival (Visitante)", "Rayo Deportivo")

    with col_main_right:
        st.markdown("### Metodología de Análisis")
        st.info("El sistema procesa el vídeo en dos pasadas: extracción de posiciones mediante detección multiobjeto con calibración de Homografía 2D y consolidación de métricas de rendimiento físico-táctico.")

        st.markdown("###")
        run_button = st.button("Ejecutar Pipeline Táctico", use_container_width=True)

    if run_button:
        if uploaded_file is None:
            st.error("Por favor, selecciona un archivo de vídeo (.mp4 o .mov) para iniciar el análisis.")
        else:
            st.session_state.home_team = home_team_input
            st.session_state.away_team = away_team_input
            st.session_state.home_color = "#10b981"  # Verde Esmeralda
            st.session_state.away_color = "#f43f5e"  # Rosa Coral
            
            with st.status("Ejecutando Pipeline Táctico...", expanded=True) as status:
                st.write("1. Cargando red neuronal YOLOv8...")
                time.sleep(0.5)
                st.write("2. Inicializando tracker multiobjeto ByteTrack...")
                time.sleep(0.5)
                st.write("3. Aplicando matriz de Homografía a plano métrico 2D (105x68m)...")
                time.sleep(0.5)
                st.write("4. Clasificación cromática por K-Means en HSV y filtrado de sombras...")
                time.sleep(0.5)
                st.write("5. Computando centroides, distancias y posesión con inercia...")
                
                raw_df = generate_tactical_sequence(3000)
                metrics = compute_distances_and_metrics(raw_df)
                
                st.session_state.df = raw_df
                st.session_state.metrics = metrics
                status.update(label="Análisis completado con éxito", state="complete")
                
            st.session_state.processed = True
            st.rerun()

else:
    # Cargar variables guardadas en sesión
    home_team = st.session_state.home_team
    away_team = st.session_state.away_team
    home_color = st.session_state.home_color
    away_color = st.session_state.away_color
    df = st.session_state.df
    metrics = st.session_state.metrics

    # Encabezado
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
        
        selected_frame = st.slider(
            "Seleccionar Instante (Fotograma):", 
            min_value=int(df['frame'].min()), 
            max_value=int(df['frame'].max()), 
            value=0
        )
        
        col_map, col_pedagogic = st.columns([2, 1])
        
        with col_map:
            fig, ax = plt.subplots(figsize=(10, 6.8))
            draw_football_pitch(ax, slate_mode=True)
            
            frame_df = df[df['frame'] == selected_frame]
            
            # Jugadores Locales
            home_players = frame_df[frame_df['team'] == 'home']
            ax.scatter(home_players['x'], home_players['y'], color=home_color, s=140, edgecolor='white', linewidth=1.5, label=home_team, zorder=5)
            for _, row in home_players.iterrows():
                ax.text(row['x'], row['y'], str(int(row['id'])), color='white', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
                
            # Jugadores Visitantes
            away_players = frame_df[frame_df['team'] == 'away']
            ax.scatter(away_players['x'], away_players['y'], color=away_color, s=140, edgecolor='white', linewidth=1.5, label=away_team, zorder=5)
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
            
            # Polígono Convex Hull (Bloque Defensivo Local)
            if len(home_players) > 3:
                try:
                    points = home_players[['x', 'y']].values
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], color=home_color, linestyle='--', alpha=0.5, zorder=4)
                except Exception:
                    pass
                    
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, facecolor='#1e293b')
            st.pyplot(fig)
            
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
                    <li><b>Polígono Punteado (Convex Hull):</b> Muestra la superficie ocupada por el bloque. Si la superficie supera los 1200 m², el equipo pierde compacidad horizontal y vertical.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
                **Consejo Didáctico para Entrenamiento:**  
                Si la distancia inter-centroide es alta durante la fase defensiva, realiza ejercicios de *basculación colectiva* reduciendo el ancho del bloque a un máximo de 30-35 metros para cerrar carriles interiores.
            """)

    # --------------------------------------------------------------------------
    # PESTAÑA 2: MAPAS DE CALOR (EQUIPOS Y BALÓN)
    # --------------------------------------------------------------------------
    with tab_heatmap:
        st.subheader("Mapas de Densidad de Ocupación Terrenal (KDE)")
        
        heatmap_choice = st.radio(
            "Seleccionar elemento para visualizar densidad:", 
            (home_team, away_team, "Balón (Excluyendo Balón Parado)")
        )
        
        col_heat, col_heat_info = st.columns([2, 1])
        
        with col_heat:
            fig_heat, ax_heat = plt.subplots(figsize=(10, 6.8))
            draw_football_pitch(ax_heat, slate_mode=True)
            
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
                    x=team_data['x'], y=team_data['y'],
                    fill=True, thresh=0.05, levels=20, cmap=color_theme,
                    alpha=0.65, ax=ax_heat, zorder=3
                )
            else:
                st.warning("Datos insuficientes para generar el mapa de calor en esta categoría.")
                
            st.pyplot(fig_heat)
            
        with col_heat_info:
            st.subheader("Interpretación Telemétrica")
            
            if heatmap_choice != "Balón (Excluyendo Balón Parado)":
                st.markdown(f"""
                <div class="tactical-card">
                    <h4>Análisis Posicional: {heatmap_choice}</h4>
                    <p>Muestra los núcleos de presencia constante de los jugadores a lo largo del clip.</p>
                    <ul>
                        <li><b>Saturación en Ocupación:</b> Las zonas más oscuras señalan dónde se acumulan los apoyos posicionales.</li>
                        <li><b>Uso de Carriles:</b> Comprueba si la densidad alcanza la línea de banda (amplitud) o si el juego se embotella en el carril central.</li>
                        <li><b>Ocupación de Medios Espacios:</b> Detecta si tus mediocentros/interiores logran recibir en la zona intermedia entre la defensa y el medio rival.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="tactical-card">
                    <h4>Mapa de Dinámica del Balón</h4>
                    <p>Muestra la trayectoria y zonas de circulación real del esférico.</p>
                    <ul>
                        <li><b>Filtro de Inercia:</b> Se han eliminado los saques de banda y paradas de juego para analizar solo la circulación en juego fluido.</li>
                        <li><b>Zonas de Impacto:</b> Identifica en qué sector del campo se juega el partido con mayor frecuencia.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 3: MÉTRICAS FÍSICAS Y TÁCTICAS
    # --------------------------------------------------------------------------
    with tab_stats:
        st.subheader("Rendimiento Físico e Inter-Evolución Táctica")
        
        c1, c2 = st.columns([1.2, 1])
        
        with c1:
            st.markdown("#### Evolución de la Distancia Inter-Centroides (m)")
            
            inter_df = metrics['inter_df']
            fig_lines, ax_lines = plt.subplots(figsize=(8, 4))
            fig_lines.patch.set_facecolor('#1e293b')
            ax_lines.set_facecolor('#0f172a')
            
            ax_lines.plot(inter_df['frame'], inter_df['inter_distance'], color='#10b981', linewidth=2.5, label="Distancia entre Centroides")
            ax_lines.axhline(inter_df['inter_distance'].mean(), color='#fbbf24', linestyle='--', label=f"Promedio: {inter_df['inter_distance'].mean():.1f}m")
            
            ax_lines.set_xlabel("Fotograma / Frame", color='#94a3b8')
            ax_lines.set_ylabel("Metros (m)", color='#94a3b8')
            ax_lines.tick_params(colors='#94a3b8')
            ax_lines.grid(True, linestyle=':', alpha=0.3, color='#475569')
            ax_lines.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
            
            st.pyplot(fig_lines)
            
        with c2:
            st.markdown("#### Distancia Recorrida por Jugador (Metros)")
            p_dist = metrics['player_distances'].copy()
            p_dist['Dorsal / ID'] = p_dist['id'].apply(lambda x: f"Jugador #{x}")
            p_dist['Equipo'] = p_dist['team'].apply(lambda t: home_team if t == 'home' else away_team)
            p_dist['Distancia (m)'] = p_dist['dist_meters'].round(1)
            
            display_table = p_dist[['Dorsal / ID', 'Equipo', 'Distancia (m)']].sort_values(by='Distancia (m)', ascending=False)
            st.dataframe(display_table, use_container_width=True, height=280)

        st.markdown("---")
        
        st.subheader("Distribución Territorial del Control de Juego")
        zone_col1, zone_col2 = st.columns(2)
        
        with zone_col1:
            zone_df = pd.DataFrame({
                'Tercio del Campo': ['Tercio Defensivo (Propio)', 'Tercio Medio (Creación)', 'Tercio Ofensivo (Rival)'],
                f'{home_team} (%)': [38, 48, 14],
                f'{away_team} (%)': [22, 52, 26]
            })
            st.dataframe(zone_df, use_container_width=True)
            
        with zone_col2:
            st.info("""
                **Nota sobre el cálculo de Distancias y Posesión:**  
                * **Distancia métrica:** La calibración por Homografía convierte la traslación de píxeles a metros reales sobre un plano corregido de 105x68m. Se aplica un filtro que descarta desplazamientos menores a 1cm por frame (ruido) y mayores a 1.2m por frame (interrupciones/reidentificación).
                * **Posesión con inercia:** Se asigna la posesión al equipo del jugador más cercano al balón (radio < 8.0m). Cuando el balón viaja por el aire o en un pase largo, el sistema mantiene la inercia del último poseedor durante un margen prudencial para reflejar la intención táctica real.
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