import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Datos ────────────────────────────────────────────────────

df = pd.read_csv('data/Arritmias.csv')

# Limpieza
cols = df.columns[1:-1]
for c in cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors='coerce')


# Definir marcadores
MARCADORES     = [c for c in df.columns if c not in ['PACIENTES', 'AV']]
MARCADORES_PCA = [c for c in MARCADORES if c != 'SEXO']
VARS_HISTO     = [c for c in df.columns if c not in ['PACIENTES', 'AV', 'SEXO']]

# ── PCA ──────────────────────────────────────────────────────

X_scaled = StandardScaler().fit_transform(df[MARCADORES_PCA])
X_pca    = PCA(n_components=2).fit_transform(X_scaled)

df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PACIENTES': df['PACIENTES'],
    'AV': df['AV']
})


C0 = '#4C72B0'
C1= '#DD8452'
CP= '#2CA02C'

# ── Figuras ───────────────────────────────────────────────────

def construir_pca(paciente_id=None):
    fig = go.Figure()

    for av, color, label in [(0, C0, 'AV = 0'), (1, C1, 'AV = 1')]:
        d = df_pca[df_pca['AV'] == av]
        fig.add_trace(go.Scatter(
            x=d['PC1'], y=d['PC2'], mode='markers', name=label,
            marker=dict(size=10, color=color, opacity=0.8),
            text=d['PACIENTES'],
        ))

    if paciente_id:
        fila = df_pca[df_pca['PACIENTES'] == paciente_id]
        if not fila.empty:
            fig.add_trace(go.Scatter(
                x=fila['PC1'], y=fila['PC2'], mode='markers',
                name=f'Paciente {paciente_id}',
                marker=dict(size=20, color=CP, symbol='circle-open', line=dict(width=3)),
            ))

    fig.update_layout(
        title={'text': '<b>PCA de marcadores cardíacos</b>', 'x': 0.5},
        xaxis_title='Componente Principal 1', yaxis_title='Componente Principal 2',
        template='plotly_white', height=800,
        legend=dict(x=0.02, y=0.98, font=dict(size=20)),

    )
    return fig


def construir_histograma(variable, paciente_id=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[variable], nbinsx=20,
        marker=dict(color='gray', line=dict(color='white', width=0.5)),
        name='Distribución',
        opacity=0.8
    ))

    if paciente_id:
        fila = df[df['PACIENTES'] == paciente_id]
        if not fila.empty:
            valor = fila[variable].values[0]
            color = C0 if int(fila['AV'].values[0]) == 0 else C1
            fig.add_vline(x=valor, line_width=4, line_dash='dash', line_color=color)
            fig.add_trace(go.Scatter(
                x=[valor], y=[0], mode='markers',
                marker=dict(size=16, color=color, symbol='diamond-open', line=dict(width=3, color='black')),
                name=f'Paciente {paciente_id}',
            ))

    fig.update_layout(
        title={'text': f'<b>Distribución: {variable}</b>', 'x': 0.5},
        template='plotly_white', height=400, bargap=0.05,
        xaxis_title=variable, yaxis_title='Frecuencia',
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(font=dict(size=20)),
    )
    return fig


def construir_radar(paciente_id=None):
    marker_cols = [c for c in MARCADORES if c.lower() not in ['edad', 'sexo']]

    df_norm = df[marker_cols].copy()
    for col in marker_cols:
        mn, mx = df[col].min(), df[col].max()
        df_norm[col] = (df[col] - mn) / (mx - mn) * 100

    fig = go.Figure()
    theta = marker_cols + [marker_cols[0]]

    for av, color, fill, label in [
        (0, C0, 'rgba(76,114,176,0.20)',  'AV = 0'),
        (1, C1, 'rgba(221,132,82,0.20)',  'AV = 1'),
    ]:
        vals = df_norm[df['AV'] == av].mean().tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=theta, fill='toself', name=label,
            line=dict(color=color, width=2), fillcolor=fill
        ))

    if paciente_id:
        idx = df[df['PACIENTES'] == paciente_id].index
        if len(idx) > 0:
            vals = df_norm.loc[idx[0], marker_cols].tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=theta, fill='toself',
                name=f'Paciente {paciente_id}',
                line=dict(color=CP, width=3), fillcolor='green', opacity=0.3
            ))

    fig.update_layout(
        title={'text': '<b>Perfil normalizado de marcadores</b>', 'x': 0.5},
        template='plotly_white', height=500,
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(x=0.02, y=1.1, font=dict(size=20))
    )
    return fig



# ── Layout ────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([
    html.H1("Dashboard Arritmias", style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        # Izquierda – PCA
        html.Div([
            dcc.Graph(id='graph-pca', clear_on_unhover=False)
        ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Derecha – Histograma + Radar
        html.Div([
            html.Div([
                html.Label("Variables", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='dropdown-variable',
                    options=[{'label': v, 'value': v} for v in VARS_HISTO],
                    value='LVEF', clearable=False
                )
            ], style={'marginBottom': '10px'}),
            dcc.Graph(id='histograma-variable'),
            dcc.Graph(id='radar-plot')
        ], style={'width': '44%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
    ])
])

# ── Callbacks ─────────────────────────────────────────────────

# Cuando se haga click en un punto de la gráfica de PCA dash genere un diccionario que se llame clickdata
def get_paciente(clickData):
    return clickData['points'][0]['text'] if clickData else None # Ponemos [0] porque es paciente, en un instante inicial está en None

@app.callback(Output('graph-pca', 'figure'), Input('graph-pca', 'clickData')) # Cuando el usuario haga click en la gráfica se actualiza la figura
def update_pca(clickData):
    return construir_pca(get_paciente(clickData))


@app.callback(
    Output('histograma-variable', 'figure'),
    [Input('graph-pca', 'clickData'), Input('dropdown-variable', 'value')]
)
def update_histograma(clickData, variable):
    return construir_histograma(variable, get_paciente(clickData))

@app.callback(Output('radar-plot', 'figure'), Input('graph-pca', 'clickData'))
def update_radar(clickData):
    return construir_radar(get_paciente(clickData))



if __name__ == '__main__':
    app.run(debug=True)