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
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from extraccion_datos import generar_dataset_deteccion
import extraccion_datos
from visualizar_seguimiento_equipos import procesar_y_limpiar_dataset

# Limpieza de variables residuales si no se ha iniciado un análisis explícito
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.df = None
    st.session_state.metrics = None
    st.session_state.pop('task_id', None)

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TacticalVision - Analítica Amateur Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def fig_to_image_buffer(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    return img_buf

def generar_figura_heatmap(df, tipo_equipo, home_team, away_team):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1f7a1f')

    ax.add_patch(patches.Rectangle((0, 0), 105, 68, facecolor='#1f7a1f', edgecolor='white', lw=1.5))
    ax.plot([52.5, 52.5], [0, 68], color='white', lw=1.5)
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, color='white', lw=1.5))
    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=1.5))
    ax.add_patch(patches.Rectangle((105-16.5, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=1.5))

    ax.set_xlim(0, 105)
    ax.set_ylim(68, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    if tipo_equipo == 'home':
        team_data = df[df['team'] == 'home']
        color_theme = "mako"
        titulo = f"Ocupación: {home_team}"
    elif tipo_equipo == 'away':
        team_data = df[df['team'] == 'away']
        color_theme = "rocket"
        titulo = f"Ocupación: {away_team}"
    else:
        ball_data = df[df['class'] == 'ball'].copy()
        if not ball_data.empty:
            ball_data['prev_x'] = ball_data['x'].shift(1)
            ball_data['prev_y'] = ball_data['y'].shift(1)
            ball_data['speed'] = np.sqrt((ball_data['x'] - ball_data['prev_x'])**2 + (ball_data['y'] - ball_data['prev_y'])**2)
            team_data = ball_data[ball_data['speed'] > 0.15]
        else:
            team_data = pd.DataFrame()
        color_theme = "flare"
        titulo = "Zonas de Transición del Balón"

    if not team_data.empty and len(team_data) > 5:
        sns.kdeplot(
            data=team_data, x='x', y='y', fill=True, cmap=color_theme,
            alpha=0.6, bw_adjust=0.5, levels=30, thresh=0.02, ax=ax, zorder=3
        )
    ax.set_title(titulo, color='white', fontsize=10, pad=6)
    plt.tight_layout()
    return fig

def generar_figura_2d_snapshot(df, frame_id, home_team, away_team, home_color, away_color):
    fig, ax = plt.subplots(figsize=(10, 6.8))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1f7a1f')

    ax.add_patch(patches.Rectangle((0, 0), 105, 68, facecolor='#1f7a1f', edgecolor='white', lw=2))
    ax.plot([52.5, 52.5], [0, 68], color='white', lw=2)
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, color='white', lw=2))
    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=2))
    ax.add_patch(patches.Rectangle((105-16.5, 13.84), 16.5, 40.32, fill=False, edgecolor='white', lw=2))

    ax.set_xlim(-3, 108)
    ax.set_ylim(68, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    frame_df = df[df['frame'] == frame_id]

    home_players = frame_df[frame_df['team'] == 'home']
    away_players = frame_df[frame_df['team'] == 'away']
    ball_df = frame_df[frame_df['team'] == 'ball']

    ax.scatter(home_players['x'], home_players['y'], color=home_color, s=90, edgecolor='white', lw=1.2, label=home_team, zorder=5)
    ax.scatter(away_players['x'], away_players['y'], color=away_color, s=90, edgecolor='white', lw=1.2, label=away_team, zorder=5)

    if not ball_df.empty:
        ax.scatter(ball_df['x'], ball_df['y'], color='#fbbf24', s=90, edgecolor='black', lw=1.5, label="Balón", zorder=7)

    if not home_players.empty and not away_players.empty:
        c_home_x, c_home_y = home_players['x'].mean(), home_players['y'].mean()
        c_away_x, c_away_y = away_players['x'].mean(), away_players['y'].mean()
        ax.scatter(c_home_x, c_home_y, color=home_color, s=250, marker='*', edgecolor='white', lw=1.5, zorder=8)
        ax.scatter(c_away_x, c_away_y, color=away_color, s=250, marker='*', edgecolor='white', lw=1.5, zorder=8)
        ax.plot([c_home_x, c_away_x], [c_home_y, c_away_y], color='#94a3b8', linestyle=':', lw=2, zorder=4)

        if len(home_players) > 3:
            try:
                pts = home_players[['x', 'y']].values
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], color=home_color, linestyle='--', alpha=0.5, zorder=4)
            except Exception:
                pass

    ax.set_title(f"Instantánea en Fotograma: {frame_id}", color='white', fontsize=11, pad=8)
    plt.tight_layout()
    return fig

def generar_pdf_informe(home_team, away_team, metrics, fig_inter=None, fig_heat_home=None, fig_heat_away=None, fig_heat_ball=None, fig_2d_snapshot=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=25, leftMargin=25, topMargin=25, bottomMargin=25)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#10b981"), spaceAfter=2)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor("#64748b"), spaceAfter=10)
    section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#0f172a"), spaceBefore=12, spaceAfter=6)
    desc_style = ParagraphStyle('TacticalDesc', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor("#334155"), leading=11, spaceBefore=4, spaceAfter=8)
    physical_style = ParagraphStyle('PhysicalBig', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor("#0f172a"), leading=14, spaceBefore=10, spaceAfter=10)

    story = []

    story.append(Paragraph("TacticalVision — Informe Telemétrico Completo", title_style))
    story.append(Paragraph(f"<b>Partido:</b> {home_team} vs {away_team}", subtitle_style))

    poss_home = metrics.get('poss_home', 0)
    poss_away = metrics.get('poss_away', 0)
    dist_home = metrics['team_distances'].get('home', 0.0)
    dist_away = metrics['team_distances'].get('away', 0.0)
    avg_inter = metrics['inter_df']['inter_distance'].mean() if not metrics['inter_df'].empty else 0.0

    kpi_data = [
        [f"Posesión {home_team}", f"Posesión {away_team}", "Dist. Inter-Centroides"],
        [f"{poss_home}%", f"{poss_away}%", f"{avg_inter:.1f} m"]
    ]
    kpi_table = Table(kpi_data, colWidths=[180, 180, 180])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TEXTCOLOR', (0, 0), (0, 1), colors.HexColor("#10b981")),
        ('TEXTCOLOR', (1, 0), (1, 1), colors.HexColor("#f43f5e")),
        ('TEXTCOLOR', (2, 0), (2, 1), colors.HexColor("#0f172a")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 14),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6))

    if fig_inter is not None:
        story.append(Paragraph("Evolución Táctica y Distancia Inter-Centroides", section_style))
        story.append(Image(fig_to_image_buffer(fig_inter), width=540, height=140))
        story.append(Paragraph(
            "<b>Guía de Interpretación Táctica:</b> La Distancia Inter-Centroides mide la separación en metros entre el punto medio del equipo local y el visitante. "
            "Una distancia reducida (<b>&lt; 20m</b>) indica alta presión o bloque denso. Una distancia amplia (<b>&gt; 30m</b>) evidencia estiramiento entre líneas.",
            desc_style
        ))
        story.append(Spacer(1, 6))

    if fig_2d_snapshot is not None:
        story.append(Paragraph("Plano Métrico 2D (Instantánea de Fotograma Seleccionado)", section_style))
        story.append(Image(fig_to_image_buffer(fig_2d_snapshot), width=540, height=210))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("Mapas de Densidad de Ocupación Terrenal (KDE)", section_style))
    
    heat_home_img = Image(fig_to_image_buffer(fig_heat_home), width=260, height=170) if fig_heat_home else None
    heat_away_img = Image(fig_to_image_buffer(fig_heat_away), width=260, height=170) if fig_heat_away else None
    heat_ball_img = Image(fig_to_image_buffer(fig_heat_ball), width=260, height=170) if fig_heat_ball else None

    if heat_home_img and heat_away_img:
        heat_table_1 = Table([[heat_home_img, heat_away_img]], colWidths=[270, 270])
        heat_table_1.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(heat_table_1)

    if heat_ball_img:
        story.append(Spacer(1, 4))
        heat_table_2 = Table([[heat_ball_img]], colWidths=[540])
        heat_table_2.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(heat_table_2)

    story.append(Spacer(1, 8))
    story.append(Paragraph("Distribución Territorial y Rendimiento Físico", section_style))
    zone_df = metrics.get('zone_df', pd.DataFrame())

    if not zone_df.empty:
        tabla_data = [["Tercio del Campo", f"{home_team} (%)", f"{away_team} (%)"]]
        for _, row in zone_df.iterrows():
            tabla_data.append([
                str(row['Tercio del Campo']),
                f"{row[f'{home_team} (%)']}%",
                f"{row[f'{away_team} (%)']}%"
            ])

        tabla_tercios = Table(tabla_data, colWidths=[240, 150, 150])
        tabla_tercios.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(tabla_tercios)

    story.append(Paragraph(
        f"<b>Rendimiento Físico Acumulado:</b><br/>"
        f"• Distancia Total Recorrida <b>{home_team}</b>: <font color='#10b981'><b>{dist_home:.1f} m</b></font><br/>"
        f"• Distancia Total Recorrida <b>{away_team}</b>: <font color='#f43f5e'><b>{dist_away:.1f} m</b></font>",
        physical_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

def calcular_distribucion_tercios(df, home_team, away_team):
    df_jugadores = df[df['class'] == 'player'].copy()
    
    home_x = df_jugadores[df_jugadores['team'] == 'home']['x']
    away_x = df_jugadores[df_jugadores['team'] == 'away']['x']
    
    TERCIO_1 = 105.0 / 3.0
    TERCIO_2 = TERCIO_1 * 2

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

    return pd.DataFrame({
        'Tercio del Campo': ['Tercio Defensivo (Propio)', 'Tercio Medio (Creación)', 'Tercio Ofensivo (Rival)'],
        f'{home_team} (%)': get_pcts_home(home_x),
        f'{away_team} (%)': get_pcts_away(away_x)
    })

# ------------------------------------------------------------------------------
# 2. MOTOR DE SIMULACIÓN Y PROCESAMIENTO TÁCTICO
# ------------------------------------------------------------------------------

def compute_distances_and_metrics(df, min_id_duration_frames=3):
    df = df.copy()

    # 1. INTERPOLACIÓN Y SUAVIZADO DEL BALÓN
    frames_totales = sorted(df['frame'].unique())
    ball_df = df[df['class'] == 'ball'][['frame', 'x', 'y']].drop_duplicates('frame')
    
    if not ball_df.empty:
        full_ball = pd.DataFrame({'frame': frames_totales}).merge(ball_df, on='frame', how='left')
        full_ball['x'] = full_ball['x'].interpolate(method='linear', limit=30)
        full_ball['y'] = full_ball['y'].interpolate(method='linear', limit=30)
        
        full_ball['x'] = full_ball['x'].rolling(window=5, min_periods=1, center=True).mean()
        full_ball['y'] = full_ball['y'].rolling(window=5, min_periods=1, center=True).mean()
        
        ball_dict = full_ball.dropna(subset=['x', 'y']).set_index('frame')[['x', 'y']].to_dict('index')
    else:
        ball_dict = {}

    # 2. FILTRADO DE JUGADORES Y DISTANCIAS REALES (Sin multiplicadores sintéticos)
    player_counts = df[df['class'] == 'player']['id'].value_counts()
    valid_ids = player_counts[player_counts >= min_id_duration_frames].index
    player_mask = (df['class'] == 'player') & (df['id'].isin(valid_ids))

    if player_mask.sum() < 50:
        player_mask = (df['class'] == 'player')

    players = df[player_mask].copy().sort_values(['id', 'frame'])

    # Diferencia de posición en metros entre frames consecutivos por jugador
    players['dx'] = players.groupby('id')['x'].diff()
    players['dy'] = players.groupby('id')['y'].diff()
    players['distancia_m_raw'] = np.sqrt(players['dx'] ** 2 + players['dy'] ** 2)

    # UMBRALES MÉTRICOS (Metros reales por fotograma)
    UMBRAL_SALTO = 1.2   # Descarta saltos > 1.2m por frame (errores de Tracking ID / cortes)
    UMBRAL_RUIDO = 0.04  # Descarta temblores < 4 cm por frame (ruido de detección)

    players['distancia_m'] = players['distancia_m_raw']
    players.loc[players['distancia_m'] > UMBRAL_SALTO, 'distancia_m'] = 0.0
    players.loc[players['distancia_m'] < UMBRAL_RUIDO, 'distancia_m'] = 0.0
    players['distancia_m'] = players['distancia_m'].fillna(0.0)

    df['distancia_m'] = 0.0
    df.loc[players.index, 'distancia_m'] = players['distancia_m']

    # Suma directa sin factores de multiplicación
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
    UMBRAL_CONTACTO = 2.5
    UMBRAL_INERCIA = 5.0
    MAX_FRAMES_INERCIA = 30

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

if not st.session_state.processed:
    st.title("TacticalVision")
    st.caption("Plataforma Táctica de Análisis Telemétrico para Fútbol Base y Amateur")

    col_main_left, col_main_right = st.columns([2, 1])

    with col_main_left:
        st.markdown("### Entrada de Vídeo del Partido")
        uploaded_file = st.file_uploader("Arrastra o selecciona el archivo de vídeo del partido (.mp4, .mov)", type=['mp4', 'mov'])

        st.markdown("---")
        st.markdown("### Configuración de Equipos")
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            home_team_input = st.text_input("Equipo 1", "C.F. Damm")
            home_color_input = st.color_picker("Color Camiseta Equipo 1", "#10b981")
        with sub_col2:
            away_team_input = st.text_input("Equipo 2", "U.E. Sant Andreu")
            away_color_input = st.color_picker("Color Camiseta Equipo 2", "#f43f5e")

    with col_main_right:
        st.markdown("### Metodología de Análisis")
        st.info("El sistema procesa el vídeo mediante detección multiobjeto, Homografía 2D y K-Means adaptativo para clasificación cromática.")
        run_button = st.button("Ejecutar Pipeline Táctico", use_container_width=True)

    if run_button:
        st.session_state.pop('pipeline_error', None)
        if uploaded_file is None:
            st.error("Por favor, selecciona un archivo de vídeo (.mp4 o .mov) para iniciar el análisis.")
        else:
            st.session_state.home_team = home_team_input
            st.session_state.away_team = away_team_input
            st.session_state.home_color = home_color_input
            st.session_state.away_color = away_color_input
            
            with st.status("Procesando vídeo en el servidor local de análisis...", expanded=True) as status:
                try:
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

                    response = requests.post(INIT_URL, files=files, headers=headers, timeout=60)

                    if response.status_code != 200:
                        st.error(f"Error backend ({response.status_code}): {response.text}")
                        st.stop()

                    json_data = response.json()
                    task_id = json_data.get('task_id')

                    if not task_id:
                        st.error("El backend no devolvió un task_id válido.")
                        st.stop()

                    STATUS_URL = f"{BASE_URL}/estado/{task_id}"
                    completado = False
                    progress_text = st.empty()
                    
                    for intento in range(900):
                        time.sleep(2)
                        try:
                            res_status = requests.get(STATUS_URL, headers=headers, timeout=10)
                            if res_status.status_code == 200:
                                status_data = res_status.json()
                                estado = status_data.get('status')
                                
                                if estado == 'completed':
                                    json_data = status_data.get('data', [])
                                    completado = True
                                    break
                                elif estado == 'error':
                                    st.error(f"La tarea falló en el servidor: {status_data.get('message')}")
                                    st.stop()
                                else:
                                    minutos = (intento * 2) // 60
                                    segundos = (intento * 2) % 60
                                    progress_text.text(f"Procesando frame por frame con YOLOv8... Tiempo transcurrido: {minutos:02d}:{segundos:02d}")
                        except requests.exceptions.RequestException:
                            pass
                    
                    progress_text.empty()

                    if not completado:
                        st.error("El tiempo de espera para completar la tarea ha expirado.")
                        st.stop()

                    raw_df = pd.DataFrame(json_data)
                    if raw_df.empty:
                        st.error("El backend devolvió un DataFrame vacío.")
                        st.stop()

                    gc.collect()

                    column_map = {
                        'id_jugador': 'id', 'player_id': 'id', 'track_id': 'id',
                        'rol_equipo': 'team', 'equipo': 'team', 'team_name': 'team',
                        'pos_x': 'x', 'x_pos': 'x', 'X': 'x', 'centroid_x': 'x',
                        'pos_y': 'y', 'y_pos': 'y', 'Y': 'y', 'centroid_y': 'y'
                    }
                    raw_df = raw_df.rename(columns=column_map)

                    if 'id' not in raw_df.columns: raw_df['id'] = 0
                    if 'team' not in raw_df.columns: raw_df['team'] = 'home'

                    raw_df['class'] = 'player'

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

                    # ESCALADO ROBUSTO (Evita la acumulación en la parte superior del clip 2)
                    FIELD_LENGTH = 105.0
                    FIELD_WIDTH = 68.0

                    raw_df['x'] = pd.to_numeric(raw_df['x'], errors='coerce').fillna(0)
                    raw_df['y'] = pd.to_numeric(raw_df['y'], errors='coerce').fillna(0)

                    valid_coords = raw_df[(raw_df['x'] > 0) & (raw_df['y'] > 0)]

                    if not valid_coords.empty and len(valid_coords) > 20:
                        # Usar percentiles en lugar de min/max absolutos para omitir 'outliers'
                        min_x, max_x = valid_coords['x'].quantile(0.02), valid_coords['x'].quantile(0.98)
                        min_y, max_y = valid_coords['y'].quantile(0.02), valid_coords['y'].quantile(0.98)

                        # Evitar división por cero si el rango es insignificante
                        range_x = (max_x - min_x) if (max_x - min_x) > 10.0 else 1.0
                        range_y = (max_y - min_y) if (max_y - min_y) > 10.0 else 1.0

                        raw_df['x'] = ((raw_df['x'] - min_x) / range_x) * FIELD_LENGTH
                        raw_df['y'] = ((raw_df['y'] - min_y) / range_y) * FIELD_WIDTH
                    else:
                        raw_df['x'] = FIELD_LENGTH / 2.0
                        raw_df['y'] = FIELD_WIDTH / 2.0

                    # Asegurar límites físicos de la cancha
                    raw_df['x'] = raw_df['x'].clip(0, FIELD_LENGTH)
                    raw_df['y'] = raw_df['y'].clip(0, FIELD_WIDTH)

                    raw_df['team'] = raw_df['team'].replace({
                        'Equipo_1': 'home', 'Equipo_2': 'away',
                        'equipo_1': 'home', 'equipo_2': 'away'
                    })

                    metrics = compute_distances_and_metrics(raw_df)

                    st.session_state.df = raw_df
                    st.session_state.metrics = metrics
                    st.session_state.processed = True

                    status.update(label="Análisis completado con éxito", state="complete")
                    st.rerun()

                except Exception as e:
                    status.update(label="Error en el pipeline", state="error")
                    import traceback
                    st.error("EXCEPCIÓN REAL DEL PIPELINE")
                    st.code(traceback.format_exc())
                    raise e

else:
    home_team = st.session_state.home_team
    away_team = st.session_state.away_team
    home_color = st.session_state.home_color
    away_color = st.session_state.away_color
    df = st.session_state.df
    metrics = st.session_state.metrics

    col_title, col_invert = st.columns([3, 1])
    with col_title:
        st.title("TacticalVision")
        st.caption(f"Informe Telemétrico Activo: **{home_team}** vs **{away_team}**")
    with col_invert:
        st.write("##")
        if st.button("🔄 Invertir Equipos", use_container_width=True):
            st.session_state.home_team, st.session_state.away_team = away_team, home_team
            st.session_state.home_color, st.session_state.away_color = away_color, home_color
            st.session_state.metrics['zone_df'] = calcular_distribucion_tercios(
                df, st.session_state.home_team, st.session_state.away_team
            )
            st.rerun()

    dist_home = metrics['team_distances'].get('home', 0.0)
    dist_away = metrics['team_distances'].get('away', 0.0)
    avg_inter_dist = metrics['inter_df']['inter_distance'].mean() if not metrics['inter_df'].empty else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric(f"Posesión {home_team}", f"{metrics['poss_home']}%")
    with m2: st.metric(f"Posesión {away_team}", f"{metrics['poss_away']}%")
    with m3: st.metric(f"Dist. Total {home_team}", f"{dist_home:.1f} m")
    with m4: st.metric(f"Dist. Total {away_team}", f"{dist_away:.1f} m")
    with m5: st.metric("Dist. Inter-Centroides", f"{avg_inter_dist:.1f} m")

    tab_projection, tab_heatmap, tab_stats = st.tabs([
        "Proyección 2D & Centroides", 
        "Mapas de Calor (Equipos y Balón)", 
        "Métricas Físicas y Tácticas"
    ])

    with tab_projection:
        st.subheader("Plano Métrico 2D Interactivo")
        frames_disponibles = sorted(df['frame'].unique())

        selected_frame = st.select_slider(
            'Seleccionar instante (fotograma):',
            options=frames_disponibles,
            value=frames_disponibles[len(frames_disponibles)//2]
        )

        frame_df = df[df['frame'] == selected_frame]
        col_map, col_pedagogic = st.columns([2, 1])
        
        with col_map:
            fig = generar_figura_2d_snapshot(df, selected_frame, home_team, away_team, home_color, away_color)
            st.session_state.fig_2d_snapshot = fig
            st.pyplot(fig, clear_figure=True)
            
        with col_pedagogic:
            st.subheader("Análisis Táctico de Centroides")
            inter_frame = metrics['inter_df'][metrics['inter_df']['frame'] == selected_frame]
            curr_inter_dist = inter_frame['inter_distance'].values[0] if not inter_frame.empty else 0.0

            st.markdown(f"""
            <div class="tactical-card">
                <h4>Centroide y Distancia Inter-Equipo</h4>
                <ul>
                    <li><b>Distancia Inter-Centroides actual:</b> <code>{curr_inter_dist:.2f} m</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_heatmap:
        st.subheader("Mapas de Densidad de Ocupación Terrenal (KDE)")
        heatmap_choice = st.radio("Seleccionar elemento para visualizar densidad:", (home_team, away_team, "Balón"))
        
        col_heat, col_heat_info = st.columns([2, 1])
        
        with col_heat:
            tipo_mapa = 'home' if heatmap_choice == home_team else ('away' if heatmap_choice == away_team else 'ball')
            fig_heat = generar_figura_heatmap(df, tipo_mapa, home_team, away_team)
            st.pyplot(fig_heat, clear_figure=True)
            
        with col_heat_info:
            st.markdown(f"""
            <div class="tactical-card">
                <h4>Análisis Posicional: {heatmap_choice}</h4>
                <p>Las zonas más densas muestran las áreas de mayor control y permanencia sobre el terreno.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_stats:
        st.subheader("Rendimiento Físico e Inter-Evolución Táctica")
        inter_df = metrics['inter_df']
        
        fig_lines, ax_lines = plt.subplots(figsize=(12, 4))
        fig_lines.patch.set_facecolor('#1e293b')
        ax_lines.set_facecolor('#0f172a')

        if not inter_df.empty:
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
        zone_df = metrics.get('zone_df', pd.DataFrame()) 
        st.dataframe(zone_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    fig_heat_home = generar_figura_heatmap(df, 'home', home_team, away_team)
    fig_heat_away = generar_figura_heatmap(df, 'away', home_team, away_team)
    fig_heat_ball = generar_figura_heatmap(df, 'ball', home_team, away_team)
    fig_2d_snap = generar_figura_2d_snapshot(df, selected_frame, home_team, away_team, home_color, away_color)
    
    pdf_data = generar_pdf_informe(
        home_team=home_team,
        away_team=away_team,
        metrics=metrics,
        fig_inter=fig_lines,
        fig_heat_home=fig_heat_home,
        fig_heat_away=fig_heat_away,
        fig_heat_ball=fig_heat_ball,
        fig_2d_snapshot=fig_2d_snap
    )

    plt.close(fig_heat_home)
    plt.close(fig_heat_away)
    plt.close(fig_heat_ball)
    plt.close(fig_2d_snap)

    col_pdf_btn, col_reset_btn = st.columns(2)
    with col_pdf_btn:
        st.download_button(
            label="Descargar Informe PDF Completo",
            data=pdf_data,
            file_name=f"Informe_Táctico_{home_team}_vs_{away_team}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col_reset_btn:
        if st.button("Cargar Nuevo Vídeo / Reiniciar", use_container_width=True):
            st.session_state.processed = False
            st.session_state.df = None
            st.session_state.metrics = None
            st.rerun()