# AUTORES:
# José Aguilar Milla
# Javier Herrero Pérez


import dash
from dash import dcc, html, callback, Input, Output, dash_table
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import mannwhitneyu
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# CARGA Y PREPARACIÓN DE DATOS (como teníamos en el código de la otras tareas, limpieza, filtrado etc)
df = pd.read_csv('data/Arritmias.csv')

# Convertir comas a puntos en columnas numéricas
cols = df.columns[1:-4]
for i in range(len(cols)):
    df[cols[i]] = df[cols[i]].str.replace(",", ".").astype(float)

# Obtener todas las columnas de marcadores (excepto ID y AV)
MARCADORES = df.columns[1:-1]

# Separar por grupo AV
df0 = df[df['AV'] == 0]
df1 = df[df['AV'] == 1]

# Lista de IDs de pacientes
PACIENTES = df['PACIENTES'].tolist()

# Colores
COLOR_AV0 = '#4C72B0'
COLOR_AV1 = '#DD8452'
COLOR_PACIENTE = '#2CA02C'  # verde para destacar al paciente
PALETTE = {0: COLOR_AV0, 1: COLOR_AV1}
LABEL_AV0 = 'AV = 0 (sin arritmia)'
LABEL_AV1 = 'AV = 1 (con arritmia)'

# ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA (para predicciones individuales)

# Usamos los marcadores cardíacos (sin EDAD, SEXO, ID, AV) para el modelo
# Esto es coherente con lo que hace el notebook
MARCADORES_MODELO = [m for m in MARCADORES if m.lower() not in ['edad', 'sexo']]

X_modelo = df[MARCADORES_MODELO].values
y_modelo = df['AV'].values

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_modelo)

# SMOTE para balancear clases (como en el notebook)
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_modelo)

# Entrenamiento del modelo
modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
modelo_lr.fit(X_resampled, y_resampled)


def predecir_probabilidad(paciente_id):
    """Predice la probabilidad de arritmia para un paciente concreto"""
    fila = df[df['PACIENTES'] == paciente_id]
    if fila.empty:
        return None
    X_paciente = fila[MARCADORES_MODELO].values
    X_paciente_scaled = scaler.transform(X_paciente)
    prob = modelo_lr.predict_proba(X_paciente_scaled)[0, 1]
    return prob


# FUNCIONES AUXILIARES PARA EL ANÁLISIS

def calcular_distancia_mahalanobis(col_a, col_b):
    """Calcula la distancia de Mahalanobis entre dos grupos para dos variables"""
    X0 = df0[[col_a, col_b]].dropna().values
    X1 = df1[[col_a, col_b]].dropna().values
    
    if len(X0) < 2 or len(X1) < 2:
        return np.nan
    
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    n0, n1 = len(X0), len(X1)
    
    cov_pooled = ((n0-1)*np.cov(X0, rowvar=False) +
                  (n1-1)*np.cov(X1, rowvar=False)) / (n0+n1-2)
    
    try:
        return mahalanobis(mu0, mu1, np.linalg.inv(cov_pooled))
    except np.linalg.LinAlgError:
        return np.nan


def crear_puntos_elipse(center, width, height, angle, num_points=100):
    """Crea puntos para dibujar una elipse en plotly"""
    t = np.linspace(0, 2*np.pi, num_points)
    
    x_local = (width/2) * np.cos(t)
    y_local = (height/2) * np.sin(t)
    
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = cos_a * x_local - sin_a * y_local
    y_rot = sin_a * x_local + cos_a * y_local
    
    x = x_rot + center[0]
    y = y_rot + center[1]
    
    return x, y


def crear_elipses_confianza(col_a, col_b):
    """Crea las elipses de confianza para ambos grupos"""
    elipses = {}
    
    for grp, color, label in [(0, COLOR_AV0, LABEL_AV0), (1, COLOR_AV1, LABEL_AV1)]:
        sub = df[df["AV"] == grp]
        vals = sub[[col_a, col_b]].dropna().values
        
        if len(vals) < 2:
            continue
        
        mu = vals.mean(axis=0)
        cov = np.cov(vals, rowvar=False)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            
            k = 1.96
            width = 2 * k * np.sqrt(eigvals[0])
            height = 2 * k * np.sqrt(eigvals[1])
            
            elipses[grp] = {
                'center': mu,
                'width': width,
                'height': height,
                'angle': angle,
                'color': color
            }
        except:
            pass
    
    return elipses


def construir_scatter(eje_x, eje_y, escala_x, escala_y, paciente_id=None):
    """Construye el scatter plot. Si paciente_id se da, lo destaca."""
    fig = go.Figure()

    dist_mahal = calcular_distancia_mahalanobis(eje_x, eje_y)
    elipses = crear_elipses_confianza(eje_x, eje_y)

    # Elipses de confianza
    for grp, elipse_data in elipses.items():
        x_elipse, y_elipse = crear_puntos_elipse(
            center=elipse_data['center'],
            width=elipse_data['width'],
            height=elipse_data['height'],
            angle=elipse_data['angle']
        )

        label = LABEL_AV0 if grp == 0 else LABEL_AV1
        fig.add_trace(go.Scatter(
            x=x_elipse,
            y=y_elipse,
            mode='lines',
            name=f"{label} (95% conf)",
            line=dict(color=elipse_data['color'], width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=[elipse_data['center'][0]],
            y=[elipse_data['center'][1]],
            mode='markers+text',
            name=f"{label} (centroide)",
            marker=dict(size=20, color=elipse_data['color'], symbol='x'),
            hovertemplate=f"<b>Centroide {label}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
            showlegend=False
        ))

    # Si hay paciente seleccionado, los puntos del resto se hacen más tenues
    opacity_otros = 0.25 if paciente_id is not None else 0.6

    fig.add_trace(go.Scatter(
        x=df0[eje_x].dropna(),
        y=df0[eje_y].dropna(),
        mode='markers',
        name=LABEL_AV0,
        marker=dict(
            size=8,
            color=COLOR_AV0,
            opacity=opacity_otros,
            line=dict(width=0.5, color='white')
        ),
        text=df0['PACIENTES'],
        hovertemplate=f"<b>Paciente</b>: %{{text}}<br>{eje_x}: %{{x:.2f}}<br>{eje_y}: %{{y:.2f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df1[eje_x].dropna(),
        y=df1[eje_y].dropna(),
        mode='markers',
        name=LABEL_AV1,
        marker=dict(
            size=8,
            color=COLOR_AV1,
            opacity=opacity_otros,
            line=dict(width=0.5, color='white')
        ),
        text=df1['PACIENTES'],
        hovertemplate=f"<b>Paciente</b>: %{{text}}<br>{eje_x}: %{{x:.2f}}<br>{eje_y}: %{{y:.2f}}<extra></extra>"
    ))

    # Punto destacado del paciente
    if paciente_id is not None:
        fila = df[df['PACIENTES'] == paciente_id]
        if not fila.empty:
            grp_paciente = int(fila['AV'].values[0])
            label_grp = LABEL_AV0 if grp_paciente == 0 else LABEL_AV1
            fig.add_trace(go.Scatter(
                x=fila[eje_x],
                y=fila[eje_y],
                mode='markers',
                name=f"Paciente {paciente_id}",
                marker=dict(
                    size=22,
                    color=COLOR_PACIENTE,
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                hovertemplate=(
                    f"<b>Paciente {paciente_id}</b> ({label_grp})<br>"
                    f"{eje_x}: %{{x:.2f}}<br>"
                    f"{eje_y}: %{{y:.2f}}<extra></extra>"
                )
            ))

    title_text = f"<b>{eje_x} vs {eje_y}</b>"
    if not np.isnan(dist_mahal):
        title_text += f"<br>Distancia Mahalanobis: {dist_mahal:.3f}"

    fig.update_layout(
        title=title_text,
        xaxis_title=eje_x,
        yaxis_title=eje_y,
        xaxis_type=escala_x,
        yaxis_type=escala_y,
        height=1000,
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=60, b=60, t=100, r=20),
        legend=dict(x=0.02, y=0.98, font=dict(size=20))
    )

    return fig


def construir_radar(paciente_id=None):
    """Radar chart con perfil normalizado por grupo. Si paciente_id se da, lo superpone."""
    marcadores_sin_demo = [m for m in MARCADORES if m.lower() not in ['edad', 'sexo']]

    df_norm = df[marcadores_sin_demo].copy()
    for col in marcadores_sin_demo:
        mn, mx = df[col].min(), df[col].max()
        df_norm[col] = (df[col] - mn) / (mx - mn) * 100

    mean0 = df_norm[df['AV'] == 0].mean()
    mean1 = df_norm[df['AV'] == 1].mean()

    labels = list(marcadores_sin_demo)

    fig = go.Figure()

    for mean, color, label in [(mean0, COLOR_AV0, LABEL_AV0), (mean1, COLOR_AV1, LABEL_AV1)]:
        vals = mean.tolist() + [mean.iloc[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels + [labels[0]],
            fill='toself',
            name=label,
            line=dict(color=color),
            marker=dict(size=5),
            fillcolor=color,
            opacity=0.3
        ))

    # Superponer paciente
    if paciente_id is not None:
        fila_norm = df_norm[df['PACIENTES'] == paciente_id]
        if not fila_norm.empty:
            vals_p = fila_norm.iloc[0].tolist() + [fila_norm.iloc[0, 0]]
            fig.add_trace(go.Scatterpolar(
                r=vals_p,
                theta=labels + [labels[0]],
                fill='toself',
                name=f"Paciente {paciente_id}",
                line=dict(color=COLOR_PACIENTE, width=3),
                marker=dict(size=8, color=COLOR_PACIENTE),
                fillcolor=COLOR_PACIENTE,
                opacity=0.5
            ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title='Perfil medio normalizado por grupo AV (Radar Chart)',
        height=1000,
        showlegend=True,
        template='plotly_white',
        legend=dict(x=0.02, y=0.98, font=dict(size=20))
    )

    return fig


# CREAR APP

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True  # lo ponemos para evitar errores al cargar los callbacks de pestañas que no están en el layout inicial

# LAYOUTS POR PESTAÑA

# --- Layout pestaña GRUPAL (la original) ---
layout_grupal = html.Div([
    html.H2("Análisis de Grupos y Relaciones",
            style={'textAlign': 'center', 'marginBottom': 30}),

    html.Div([
        html.Div([
            dcc.Graph(id='graph-scatter-grupo')
        ], style={
            'width': '70%',
            'display': 'inline-block',
            'verticalAlign': 'middle',
            'padding': '20px',
            'boxSizing': 'border-box'
        }),

        html.Div([
            html.Div([
                html.H3("Controles", style={'marginTop': 0}),

                html.Div([
                    html.Label("Eje X:", style={'fontWeight': 'bold', 'marginTop': 20}),
                    dcc.Dropdown(
                        id='dropdown-eje-x',
                        options=[{'label': m, 'value': m} for m in MARCADORES],
                        value=MARCADORES[0],
                        clearable=False
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Eje Y:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='dropdown-eje-y',
                        options=[{'label': m, 'value': m} for m in MARCADORES],
                        value=MARCADORES[1],
                        clearable=False
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Escala X:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='radio-escala-x',
                        options=[
                            {'label': ' Lineal', 'value': 'linear'},
                            {'label': ' Logarítmica', 'value': 'log'}
                        ],
                        value='linear',
                        labelStyle={'display': 'block', 'marginBottom': 10}
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Escala Y:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='radio-escala-y',
                        options=[
                            {'label': ' Lineal', 'value': 'linear'},
                            {'label': ' Logarítmica', 'value': 'log'}
                        ],
                        value='linear',
                        labelStyle={'display': 'block', 'marginBottom': 10}
                    )
                ], style={'marginBottom': 20}),

            ], style={
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '20px',
                'borderRadius': '10px',
                'borderLeft': '5px solid #4C72B0'
            })
        ], style={
            'width': '28%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'boxSizing': 'border-box'
        })
    ], style={
        'display': 'flex',
        'width': '100%'
    }),

    html.Div([
        dcc.Graph(id='graph-radar')
    ], style={'padding': '20px'})
], style={'padding': '20px'})


# --- Layout pestaña INDIVIDUAL (nueva) ---
layout_individual = html.Div([
    html.H2("Análisis Individual del Paciente",
            style={'textAlign': 'center', 'marginBottom': 30}),

    html.Div([
        # Columna izquierda: scatter
        html.Div([
            dcc.Graph(id='graph-scatter-individual')
        ], style={
            'width': '70%',
            'display': 'inline-block',
            'verticalAlign': 'middle',
            'padding': '20px',
            'boxSizing': 'border-box'
        }),

        # Columna derecha: controles + predicción + tabla
        html.Div([
            # Panel de controles
            html.Div([
                html.H3("Controles", style={'marginTop': 0}),

                html.Div([
                    html.Label("Paciente:", style={'fontWeight': 'bold', 'marginTop': 20}),
                    dcc.Dropdown(
                        id='dropdown-paciente',
                        options=[{'label': p, 'value': p} for p in PACIENTES],
                        value=PACIENTES[0],
                        clearable=False
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Eje X:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='dropdown-eje-x-ind',
                        options=[{'label': m, 'value': m} for m in MARCADORES],
                        value='LVEF' if 'LVEF' in MARCADORES else MARCADORES[0],
                        clearable=False
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Eje Y:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='dropdown-eje-y-ind',
                        options=[{'label': m, 'value': m} for m in MARCADORES],
                        value='LV MASS (g)' if 'LV MASS (g)' in MARCADORES else MARCADORES[1],
                        clearable=False
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Escala X:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='radio-escala-x-ind',
                        options=[
                            {'label': ' Lineal', 'value': 'linear'},
                            {'label': ' Logarítmica', 'value': 'log'}
                        ],
                        value='linear',
                        labelStyle={'display': 'block', 'marginBottom': 10}
                    )
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Escala Y:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='radio-escala-y-ind',
                        options=[
                            {'label': ' Lineal', 'value': 'linear'},
                            {'label': ' Logarítmica', 'value': 'log'}
                        ],
                        value='linear',
                        labelStyle={'display': 'block', 'marginBottom': 10}
                    )
                ], style={'marginBottom': 20}),

            ], style={
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '20px',
                'borderRadius': '10px',
                'borderLeft': f'5px solid {COLOR_PACIENTE}',
                'marginBottom': 20
            }),

            # Panel de predicción del modelo
            html.Div(id='panel-prediccion', style={
                'padding': '20px',
                'borderRadius': '10px',
                'marginBottom': 20
            })

        ], style={
            'width': '28%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'boxSizing': 'border-box'
        })
    ], style={
        'display': 'flex',
        'width': '100%'
    }),

    # Radar
    html.Div([
        dcc.Graph(id='graph-radar-individual')
    ], style={'padding': '20px'}),

    # Tabla con los datos del paciente
    html.Div([
        html.H3("Datos del paciente vs. medias por grupo",
                style={'textAlign': 'center', 'marginTop': 30, 'marginBottom': 20}),
        html.Div(id='tabla-paciente')
    ], style={'padding': '20px'})

], style={'padding': '20px'})


# LAYOUT PRINCIPAL CON PESTAÑAS


app.layout = html.Div([
    html.H1("Dashboard: Marcadores Pro-Arrítmicos",
            style={'textAlign': 'center'}),

    dcc.Tabs(id='tabs', value='tab-grupal', children=[
        dcc.Tab(label='Análisis Grupal', value='tab-grupal',
                style={'fontWeight': 'bold'},
                selected_style={'fontWeight': 'bold', 'borderTop': f'3px solid {COLOR_AV0}'}),
        dcc.Tab(label='Análisis Individual', value='tab-individual',
                style={'fontWeight': 'bold'},
                selected_style={'fontWeight': 'bold', 'borderTop': f'3px solid {COLOR_PACIENTE}'}),
    ]),

    html.Div(id='contenido-tab')
])


# CALLBACKS

@callback(
    Output('contenido-tab', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-grupal':
        return layout_grupal
    elif tab == 'tab-individual':
        return layout_individual


# --- Callbacks de la pestaña GRUPAL (los originales) ---

@callback(
    Output('graph-scatter-grupo', 'figure'),
    [Input('dropdown-eje-x', 'value'),
     Input('dropdown-eje-y', 'value'),
     Input('radio-escala-x', 'value'),
     Input('radio-escala-y', 'value')]
)
def update_scatter_grupo(eje_x, eje_y, escala_x, escala_y):
    """Actualiza el scatter plot cuando cambian los ejes o escalas"""
    return construir_scatter(eje_x, eje_y, escala_x, escala_y, paciente_id=None)


@callback(
    Output('graph-radar', 'figure'),
    [Input('dropdown-eje-x', 'value')]
)
def update_radar(dummy):
    """Radar chart con perfil normalizado por grupo"""
    return construir_radar(paciente_id=None)


# --- Callbacks de la pestaña INDIVIDUAL (nuevos) ---

@callback(
    Output('graph-scatter-individual', 'figure'),
    [Input('dropdown-eje-x-ind', 'value'),
     Input('dropdown-eje-y-ind', 'value'),
     Input('radio-escala-x-ind', 'value'),
     Input('radio-escala-y-ind', 'value'),
     Input('dropdown-paciente', 'value')],
    prevent_initial_call=True
)
def update_scatter_individual(eje_x, eje_y, escala_x, escala_y, paciente_id):
    return construir_scatter(eje_x, eje_y, escala_x, escala_y, paciente_id=paciente_id)


@callback(
    Output('graph-radar-individual', 'figure'),
    [Input('dropdown-paciente', 'value')],
    prevent_initial_call=True
)
def update_radar_individual(paciente_id):
    return construir_radar(paciente_id=paciente_id)


@callback(
    Output('panel-prediccion', 'children'),
    [Input('dropdown-paciente', 'value')],
    prevent_initial_call=True
)
def update_prediccion(paciente_id):
    """Muestra la probabilidad de arritmia estimada por el modelo"""
    if paciente_id is None:
        return html.Div()

    prob = predecir_probabilidad(paciente_id)
    fila = df[df['PACIENTES'] == paciente_id]
    if fila.empty or prob is None:
        return html.Div("Paciente no encontrado")

    av_real = int(fila['AV'].values[0])
    label_real = LABEL_AV0 if av_real == 0 else LABEL_AV1

    # Color del riesgo según probabilidad
    if prob < 0.33:
        color_riesgo = COLOR_AV0
        nivel = "BAJO"
    elif prob < 0.66:
        color_riesgo = '#E5A23B'
        nivel = "MEDIO"
    else:
        color_riesgo = COLOR_AV1
        nivel = "ALTO"

    return html.Div([
        html.H3("Predicción del modelo", style={'marginTop': 0}),
        html.P("Regresión Logística con SMOTE",
               style={'fontStyle': 'italic', 'fontSize': 12, 'color': '#666', 'marginTop': 0}),

        html.Div([
            html.Div([
                html.Span("Prob. de arritmia:", style={'fontWeight': 'bold'}),
                html.Br(),
                html.Span(f"{prob*100:.1f}%",
                          style={'fontSize': 32, 'fontWeight': 'bold', 'color': color_riesgo})
            ], style={'marginBottom': 15}),

            # Barra visual de probabilidad
            html.Div([
                html.Div(style={
                    'width': f'{prob*100}%',
                    'height': '20px',
                    'backgroundColor': color_riesgo,
                    'borderRadius': '5px',
                    'transition': 'width 0.5s'
                })
            ], style={
                'width': '100%',
                'height': '20px',
                'backgroundColor': '#e0e0e0',
                'borderRadius': '5px',
                'marginBottom': 15
            }),

            html.Div([
                html.Span("Nivel de riesgo: ", style={'fontWeight': 'bold'}),
                html.Span(nivel, style={'color': color_riesgo, 'fontWeight': 'bold'})
            ], style={'marginBottom': 10}),

            html.Hr(),

            html.Div([
                html.Span("Estado real: ", style={'fontWeight': 'bold'}),
                html.Span(label_real,
                          style={'color': COLOR_AV0 if av_real == 0 else COLOR_AV1})
            ])
        ])
    ], style={
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '20px',
        'borderRadius': '10px',
        'borderLeft': f'5px solid {color_riesgo}'
    })


@callback(
    Output('tabla-paciente', 'children'),
    [Input('dropdown-paciente', 'value')],
    prevent_initial_call=True
)
def update_tabla(paciente_id):
    """Muestra una tabla con los datos del paciente comparados con las medias por grupo"""
    if paciente_id is None:
        return html.Div()

    fila = df[df['PACIENTES'] == paciente_id]
    if fila.empty:
        return html.Div("Paciente no encontrado")

    av_real = int(fila['AV'].values[0])

    # Construimos los datos de la tabla (excluyendo SEXO y EDAD)
    rows = []
    for marcador in MARCADORES:
        # Saltar SEXO y EDAD
        if marcador.lower() in ['sexo', 'edad']:
            continue
            
        valor_paciente = fila[marcador].values[0]
        media_av0 = df0[marcador].mean()
        media_av1 = df1[marcador].mean()

        dist_0 = abs(valor_paciente - media_av0)
        dist_1 = abs(valor_paciente - media_av1)
        mas_cerca = 'AV = 0' if dist_0 < dist_1 else 'AV = 1'

        rows.append({
            'Marcador': marcador,
            'Valor paciente': f"{valor_paciente:.2f}",
            'Media AV=0': f"{media_av0:.2f}",
            'Media AV=1': f"{media_av1:.2f}",
            'Más cerca de': mas_cerca
        })
        if marcador.lower() in ['sexo', 'edad']:
            continue
    # Estilos condicionales: resaltar la columna del grupo real del paciente
    grupo_real_col = 'Media AV=0' if av_real == 0 else 'Media AV=1'

    return dash_table.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in
                 ['Marcador', 'Valor paciente', 'Media AV=0', 'Media AV=1', 'Más cerca de']],
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'sans-serif'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': '1px solid #ccc'
        },
        style_data_conditional=[
            # Resaltar columna "Valor paciente"
            {
                'if': {'column_id': 'Valor paciente'},
                'backgroundColor': 'rgba(44, 160, 44, 0.15)',
                'fontWeight': 'bold'
            },
            # Resaltar columna del grupo real
            {
                'if': {'column_id': grupo_real_col},
                'backgroundColor': f'rgba({76 if av_real == 0 else 221}, '
                                   f'{114 if av_real == 0 else 132}, '
                                   f'{176 if av_real == 0 else 82}, 0.15)'
            },
            # Resaltar fila si "Más cerca de" coincide con AV=1 (alerta)
            {
                'if': {
                    'filter_query': '{Más cerca de} = "AV = 1"',
                    'column_id': 'Más cerca de'
                },
                'color': COLOR_AV1,
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{Más cerca de} = "AV = 0"',
                    'column_id': 'Más cerca de'
                },
                'color': COLOR_AV0,
                'fontWeight': 'bold'
            }
        ],
        style_table={'overflowX': 'auto'}
    )


if __name__ == '__main__':
    app.run(debug=True)
