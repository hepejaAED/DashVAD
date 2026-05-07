import dash
from dash import dcc, html, callback, Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import mannwhitneyu
from itertools import combinations

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

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

# Colores
COLOR_AV0 = '#4C72B0'
COLOR_AV1 = '#DD8452'
PALETTE = {0: COLOR_AV0, 1: COLOR_AV1}
LABEL_AV0 = 'AV = 0 (sin arritmia)'
LABEL_AV1 = 'AV = 1 (con arritmia)'

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

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

# ============================================================================
# CREAR APP
# ============================================================================

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    html.H1("Dashboard: Marcadores Pro-Arrítmicos", 
            style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    html.Div([
        html.H2("Análisis de Grupos y Relaciones", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='graph-scatter-grupo')
            ], style={
                'width': '70%',
                'display': 'inline-block',
                'verticalAlign': 'top',
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
                            value=MARCADORES[1] if len(MARCADORES) > 1 else MARCADORES[0],
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
                    'borderRadius': '5px',
                    'borderLeft': '3px solid #4C72B0'
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
        })
    ], style={'padding': '20px'})
])

# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output('graph-scatter-grupo', 'figure'),
    [Input('dropdown-eje-x', 'value'),
     Input('dropdown-eje-y', 'value'),
     Input('radio-escala-x', 'value'),
     Input('radio-escala-y', 'value')]
)
def update_scatter_grupo(eje_x, eje_y, escala_x, escala_y):
    """Actualiza el scatter plot cuando cambian los ejes o escalas"""
    
    fig = go.Figure()
    
    dist_mahal = calcular_distancia_mahalanobis(eje_x, eje_y)
    elipses = crear_elipses_confianza(eje_x, eje_y)
    
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
            marker=dict(size=12, color=elipse_data['color'], symbol='x', line=dict(width=3)),
            text=[''],
            hovertemplate=f"<b>Centroide {label}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=df0[eje_x].dropna(),
        y=df0[eje_y].dropna(),
        mode='markers',
        name=LABEL_AV0,
        marker=dict(
            size=8,
            color=COLOR_AV0,
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        text=df0.index,
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
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        text=df1.index,
        hovertemplate=f"<b>Paciente</b>: %{{text}}<br>{eje_x}: %{{x:.2f}}<br>{eje_y}: %{{y:.2f}}<extra></extra>"
    ))
    
    title_text = f"<b>{eje_x} vs {eje_y}</b>"
    if not np.isnan(dist_mahal):
        title_text += f"<br><sub>Distancia Mahalanobis: {dist_mahal:.3f}</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=eje_x,
        yaxis_title=eje_y,
        xaxis_type=escala_x,
        yaxis_type=escala_y,
        height=600,
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=60, b=60, t=100, r=20),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)