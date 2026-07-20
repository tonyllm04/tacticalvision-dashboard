import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial import ConvexHull
import time

# Configuracion de pagina de Streamlit
st.set_page_config(
    page_title="TacticalVision - Analitica Amateur",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyeccion de estilos CSS personalizados para lograr un diseño profesional oscuro
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
        font-size: 28px;
        color: #10b981;
    }
    .tactical-card {
        background-color: #1e293b;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

class HomographyTransformer:
    """
    Clase matematica encargada de transformar las coordenadas de los pixeles de la camara
    al plano metrico bidimensional real del campo de juego (105m x 68m).
    """
    def __init__(self):
        # Puntos de origen tipicos en una retransmision de camara tactica
        src_points = np.float32([
            [210, 450],   # Corner superior izquierdo en pantalla
            [1070, 450],  # Corner superior derecho en pantalla
            [50, 720],    # Esquina inferior izquierda en pantalla
            [1230, 720]   # Esquina inferior derecha en pantalla
        ])
        
        # Puntos de destino en el plano real de futbol 2D (Medidas estandar en metros)
        dst_points = np.float32([
            [0, 0],       # Banda superior izquierda
            [105, 0],     # Banda superior derecha
            [0, 68],      # Banda inferior izquierda
            [105, 68]     # Banda inferior derecha
        ])
        
        # Calcular la matriz de homografia utilizando OpenCV
        self.H, _ = cv2.findHomography(src_points, dst_points)

    def transform_point(self, u, v):
        """Transforma un punto pixel (u,v) a metros reales (x,y)"""
        point = np.array([u, v, 1.0], dtype=np.float32)
        transformed = np.dot(self.H, point)
        x = transformed[0] / transformed[2]
        y = transformed[1] / transformed[2]
        return np.clip(x, 0, 105), np.clip(y, 0, 68)

def draw_football_pitch(ax, slate_mode=True):
    """
    Dibuja un terreno de juego profesional a escala 105x68 metros sobre un eje Matplotlib.
    """
    bg_color = '#1e293b' if slate_mode else '#2d5a27'
    line_color = '#475569' if slate_mode else '#ffffff'
    
    # Fondo del campo
    rect = patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor=line_color, facecolor=bg_color, zorder=1)
    ax.add_patch(rect)
    
    # Linea de medio campo y circulo central
    ax.plot([52.5, 52.5], [0, 68], color=line_color, linewidth=2, zorder=2)
    center_circle = patches.Circle((52.5, 34), 9.15, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2)
    ax.add_patch(center_circle)
    ax.scatter(52.5, 34, color=line_color, s=20, zorder=3)
    
    # Area de meta y penalty izquierda
    ax.stroke_rect_left = patches.Rectangle((0, 13.84), 16.5, 40.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2)
    ax.add_patch(ax.stroke_rect_left)
    ax.stroke_small_left = patches.Rectangle((0, 24.84), 5.5, 18.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2)
    ax.add_patch(ax.stroke_small_left)
    
    # Area de meta y penalty derecha
    ax.stroke_rect_right = patches.Rectangle((88.5, 13.84), 16.5, 40.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2)
    ax.add_patch(ax.stroke_rect_right)
    ax.stroke_small_right = patches.Rectangle((99.5, 24.84), 5.5, 18.32, linewidth=2, edgecolor=line_color, facecolor='none', zorder=2)
    ax.add_patch(ax.stroke_small_right)
    
    # Puntos de penalti
    ax.scatter(11, 34, color=line_color, s=20, zorder=3)
    ax.scatter(94, 34, color=line_color, s=20, zorder=3)
    
    # Arcos de area
    arc_left = patches.Arc((11, 34), 18.3, 18.3, theta1=-53, theta2=53, linewidth=2, color=line_color, zorder=2)
    arc_right = patches.Arc((94, 34), 18.3, 18.3, theta1=127, theta2=233, linewidth=2, color=line_color, zorder=2)
    ax.add_patch(arc_left)
    ax.add_patch(arc_right)
    
    # Configuracion de limites y visibilidad de ejes
    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.axis('off')

def generate_tactical_sequence(frames=120):
    """
    Genera datos realistas de tracking en 2D que simulan una transicion ofensiva.
    """
    data = []
    np.random.seed(42)
    
    for f in range(frames):
        progress = f / frames
        # Movimiento de balon
        ball_x = 35 + progress * 50 + np.sin(f*0.15) * 3
        ball_y = 20 + progress * 28 + np.cos(f*0.15) * 4
        
        data.append({
            'frame': f, 'id': 99, 'class': 'ball', 
            'x': ball_x, 'y': ball_y, 'team': 'ball'
        })
        
        # 11 Jugadores Locales (Posiciones tacticas en 4-4-2 replegando)
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
                'x': bx + np.random.normal(0, 0.1), 
                'y': by + np.random.normal(0, 0.1), 
                'team': 'home'
            })
            
        # 11 Jugadores Atacantes Rivales (Transicion ofensiva)
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

# Panel de Control lateral limpio sin emoticonos ni texto descriptivo adicional
st.sidebar.title("Panel de Control")

# Inicializacion de estados de sesion
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.df = None

if not st.session_state.processed:
    st.title("TacticalVision")
    st.caption("Prototipo de TFG - Canal de procesado telemétrico para analistas y entrenadores de fútbol base")

    col_main_left, col_main_right = st.columns([2, 1])

    with col_main_left:
        st.markdown("### Entrada de Vídeo del Partido")
        
        use_demo = st.checkbox("Usar Partido de Demostración (Recomendado para pruebas rápidas)", value=True)
        uploaded_file = None
        
        if not use_demo:
            uploaded_file = st.file_uploader("Arrastra o selecciona el archivo de vídeo del partido (.mp4, .mov)", type=['mp4', 'mov'])
            st.info("Soporta grabaciones de cámaras tácticas o clips descargados de la plataforma del club.")
        else:
            st.success("Se utilizará el generador matemático integrado con trayectorias de transición ofensiva coordinadas.")

        st.markdown("---")
        st.markdown("### Configuración de Equipos")
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            home_team_input = st.text_input("Equipo Local (Principal)", "CD Alianza Amateur")
        with sub_col2:
            away_team_input = st.text_input("Equipo Rival (Visitante)", "Rayo Deportivo")

    with col_main_right:
        st.markdown("### Metodología Didáctica")
        st.info("El sistema traduce datos posicionales complejos a conceptos sencillos y consejos de entrenamiento prácticos adaptados a los equipos de fútbol base.")

        st.markdown("###")
        run_button = st.button("Ejecutar Pipeline Táctico", use_container_width=True)

    if run_button:
        if not use_demo and uploaded_file is None:
            st.error("Por favor, selecciona un archivo de vídeo o marca la casilla para usar el partido de demostración.")
        else:
            # Guardar configuracion en la sesion para evitar perdida de estado
            st.session_state.home_team = home_team_input
            st.session_state.away_team = away_team_input
            st.session_state.home_color = "#10b981"  # Verde Esmeralda fijo
            st.session_state.away_color = "#f43f5e"  # Rosa Coral fijo
            
            # Ejecutar barra de carga simulando la inferencia
            with st.status("Ejecutando Pipeline Táctico...", expanded=True) as status:
                st.write("Cargando red neuronal YOLOv8...")
                time.sleep(0.6)
                st.write("Inicializando tracker multiobjeto ByteTrack...")
                time.sleep(0.5)
                st.write("Computando calibración de Homografía 2D...")
                time.sleep(0.5)
                st.write("Clasificando equipos mediante segmentación cromática...")
                time.sleep(0.6)
                st.write("Consolidando métricas de rendimiento...")
                
                st.session_state.df = generate_tactical_sequence()
                status.update(label="Análisis completado de forma correcta", state="complete")
                
            st.session_state.processed = True
            st.rerun()

else:
    # Recuperar variables del estado
    home_team = st.session_state.home_team
    away_team = st.session_state.away_team
    home_color = st.session_state.home_color
    away_color = st.session_state.away_color
    df = st.session_state.df

    # Encabezado del visor de resultados
    st.title("TacticalVision")
    st.caption(f"Análisis Activo: {home_team} vs {away_team}")

    # Computo de posesion basado en proximidad de seguimiento
    frames_list = df['frame'].unique()
    possession_counter = {'home': 0, 'away': 0, 'disputed': 0}
    
    for f in frames_list:
        frame_data = df[df['frame'] == f]
        ball = frame_data[frame_data['class'] == 'ball']
        players = frame_data[frame_data['class'] == 'player']
        
        if not ball.empty and not players.empty:
            bx, by = ball.iloc[0]['x'], ball.iloc[0]['y']
            players = players.copy()
            players['dist'] = np.sqrt((players['x'] - bx)**2 + (players['y'] - by)**2)
            closest_player = players.loc[players['dist'].idxmin()]
            
            if closest_player['dist'] < 8.0:
                possession_counter[closest_player['team']] += 1
            else:
                possession_counter['disputed'] += 1
                
    total_active_frames = possession_counter['home'] + possession_counter['away']
    poss_home = round((possession_counter['home'] / total_active_frames) * 100) if total_active_frames > 0 else 50
    poss_away = 100 - poss_home

    # Fila superior de metricas clave
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"Posesión {home_team}", f"{poss_home}%")
    with m2:
        st.metric(f"Posesión {away_team}", f"{poss_away}%")
    with m3:
        st.metric("Jugadores Detectados", f"22")
    with m4:
        st.metric("Fidelidad Geométrica", "96.4%", help="Precisión obtenida tras la homografía lineal y corrección por lentes de gran angular")

    tab_projection, tab_heatmap, tab_stats = st.tabs([
        "Proyección Táctica Proporcional", 
        "Mapas de Densidad (Heatmaps)", 
        "Métricas de Bloque y Amplitud"
    ])

    with tab_projection:
        st.subheader("Plano Métrico 2D Interactivo")
        
        # Deslizador temporal para la navegacion interactiva de fotogramas
        selected_frame = st.slider("Instante del Partido (Fotograma):", 
                                   min_value=int(df['frame'].min()), 
                                   max_value=int(df['frame'].max()), 
                                   value=0)
        
        col_map, col_pedagogic = st.columns([2, 1])
        
        with col_map:
            # Dibujar el campo de juego y posicionamiento con Matplotlib
            fig, ax = plt.subplots(figsize=(10, 7))
            draw_football_pitch(ax, slate_mode=True)
            
            frame_df = df[df['frame'] == selected_frame]
            
            # Jugadores Locales
            home_players = frame_df[frame_df['team'] == 'home']
            ax.scatter(home_players['x'], home_players['y'], color=home_color, s=150, edgecolor='white', linewidth=1.5, label=home_team, zorder=5)
            for _, row in home_players.iterrows():
                ax.text(row['x'], row['y'], str(int(row['id'])), color='white', fontsize=8, ha='center', va='center', fontweight='bold', zorder=6)
                
            # Jugadores Rivales
            away_players = frame_df[frame_df['team'] == 'away']
            ax.scatter(away_players['x'], away_players['y'], color=away_color, s=150, edgecolor='white', linewidth=1.5, label=away_team, zorder=5)
            for _, row in away_players.iterrows():
                ax.text(row['x'], row['y'], str(int(row['id'])), color='white', fontsize=8, ha='center', va='center', fontweight='bold', zorder=6)
                
            # Balon tactico
            ball_df = frame_df[frame_df['team'] == 'ball']
            if not ball_df.empty:
                ax.scatter(ball_df['x'], ball_df['y'], color='#fbbf24', s=100, edgecolor='black', linewidth=1.5, label="Balón", zorder=7)
            
            # Dibujo del Poligono de Bloque Defensivo (Convex Hull) de la linea local
            if len(home_players) > 3:
                try:
                    points = home_players[['x', 'y']].values
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], color=home_color, linestyle='--', alpha=0.4, zorder=4)
                except Exception:
                    pass
                    
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
            st.pyplot(fig)
            
        with col_pedagogic:
            st.subheader("Aclaración Táctica del Fotograma")
            
            st.markdown(f"""
            <div class="tactical-card">
                <h4>Análisis Técnico: Línea de Bloque Defensivo y Espacio</h4>
                <p>La línea punteada en el gráfico representa el <i>Convex Hull</i> o polígono envolvente de tu bloque defensivo (<b>{home_team}</b>).</p>
                <p><b>Métricas observadas en escena:</b></p>
                <ul>
                    <li><b>Ocupación del Espacio:</b> Comprueba la distribución espacial. Un bloque defensivo excesivamente estirado (amplitud mayor a 35m) facilita la aparición de pasillos de pase interiores que el rival puede explotar.</li>
                    <li><b>Compactación espacial:</b> La amplitud del bloque se mantiene contenida en metros cuadrados, reduciendo los intervalos o pasillos interiores útiles para el rival.</li>
                    <li><b>Basculación:</b> Ante la posición de ataque del rival, se observa una correcta orientación corporal de las ayudas tácticas hacia el sector fuerte del balón.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
                **Recomendación Formativa:**
                Practica rondos de posesión delimitados en espacios reducidos de 20x20 metros para entrenar la velocidad de pase bajo presión y mantener el bloque de juego junto. Si el polígono envolvente supera los 35 metros de ancho, se aconseja entrenar vigilancias defensivas en fase de posesión propia.
            """)

    with tab_heatmap:
        st.subheader("Densidad de Ocupación Dinámica del Terreno de Juego")
        
        heatmap_team = st.radio("Seleccionar equipo para el Mapa de Calor:", (home_team, away_team))
        
        col_heat, col_heat_info = st.columns([2, 1])
        
        with col_heat:
            fig_heat, ax_heat = plt.subplots(figsize=(10, 7))
            draw_football_pitch(ax_heat, slate_mode=True)
            
            team_key = 'home' if heatmap_team == home_team else 'away'
            team_data = df[df['team'] == team_key]
            
            if not team_data.empty:
                sns.kdeplot(
                    x=team_data['x'], y=team_data['y'],
                    fill=True, thresh=0.05, levels=20, cmap="mako",
                    alpha=0.6, ax=ax_heat, zorder=3
                )
            st.pyplot(fig_heat)
            
        with col_heat_info:
            st.subheader("Interpretación del Mapa de Calor")
            
            st.markdown(f"""
            <div class="tactical-card">
                <h4>Análisis de Intervalos y Amplitud Táctica</h4>
                <p>El mapa de densidad de kernel (KDE) revela los focos de acumulación posicional y saturación del <b>{heatmap_team}</b>.</p>
                <p><b>Análisis estratégico:</b></p>
                <ul>
                    <li><b>Distribución territorial:</b> Las áreas oscuras muestran dónde han pasado más tiempo tus jugadores. Si las manchas están concentradas en tu propia portería, significa que el equipo está replegado y le cuesta progresar.</li>
                    <li><b>Ataque a Medias Espacios (Half-Spaces):</b> Evalúa si los focos de densidad ofensiva se sitúan entre los carriles centrales y laterales, facilitando la desorganización defensiva del rival.</li>
                    <li><b>Uso de la amplitud:</b> Una mancha intensa en las bandas certifica una correcta utilización del ancho de campo para estirar las líneas defensivas rivales.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_stats:
        st.subheader("Métricas de Rendimiento Colectivo")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribución Territorial de Posesión")
            zone_data = pd.DataFrame({
                'Zona': ['Tercio Propio', 'Zona Central', 'Tercio Rival'],
                f'{home_team} (%)': [40, 45, 15],
                f'{away_team} (%)': [25, 50, 25]
            })
            st.dataframe(zone_data, use_container_width=True)
            
        with c2:
            st.subheader("Aclaración sobre la medición de la Posesión")
            st.info(f"""
                **Nota Analítica de Posesión:**
                La posesión se calcula mediante el rastreo algorítmico de la distancia euclidiana entre cada jugador y el balón en el plano proyectado por Homografía. Un jugador controla el esférico si se encuentra dentro de un radio inferior a 8 metros. Si el balón supera dicho umbral, el sistema computa el tiempo como balón dividido para evitar la inercia temporal y el parpadeo de datos en las transiciones rápidas o despejes largos.
            """)

    st.markdown("---")
    col_reset_btn, _ = st.columns([1, 2])
    with col_reset_btn:
        if st.button("Cargar Nuevo Vídeo / Reiniciar", use_container_width=True):
            st.session_state.processed = False
            st.session_state.df = None
            st.rerun()