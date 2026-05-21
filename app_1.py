# ============================================================
# DASHBOARD PCA - ARRITMIAS
# ============================================================

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np

import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# CARGA DE DATOS Y TRATAMIENTO DE ESTOS

df = pd.read_csv('data/Arritmias.csv')

# COLUMNAS NUMÉRICAS
cols = df.columns[1:-1]

for c in cols:
    df[c] = df[c].astype(str).str.replace(",", ".")
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

PACIENTES = df['PACIENTES'].tolist()

MARCADORES = [
    c for c in df.columns
    if c not in ['PACIENTES', 'AV'] # AV es la variable objetivo, no un marcador
]

# Excluimos edad y sexo para PCA
MARCADORES_PCA = [
    c for c in MARCADORES
    if c not in ['SEXO']
]
VARIABLES_HISTOGRAMA = [
    c for c in df.columns
    if c not in ['PACIENTES', 'AV',"SEXO"]
]


# PCA
X = df[MARCADORES_PCA]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PACIENTES': df['PACIENTES'],
    'AV': df['AV']
})

# ============================================================
# COLORES
# ============================================================

COLOR_AV0 = '#4C72B0'
COLOR_AV1 = '#DD8452'
COLOR_PACIENTE = '#2CA02C'

LABEL_AV0 = 'AV = 0'
LABEL_AV1 = 'AV = 1'



def construir_histograma(variable, paciente_id=None):

    fig = go.Figure()

    # --------------------------------------------------------
    # Histograma general
    # --------------------------------------------------------

    fig.add_trace(go.Histogram(

        x=df[variable],

        nbinsx=20,

        marker=dict(
            color='rgba(120,120,120,0.6)',
            line=dict(
                color='white',
                width=1
            )
        ),

        name='Distribución'
    ))

    # --------------------------------------------------------
    # Paciente seleccionado
    # --------------------------------------------------------

    if paciente_id is not None:

        fila = df[df['PACIENTES'] == paciente_id]

        if not fila.empty:

            valor = fila[variable].values[0]

            av_real = int(fila['AV'].values[0])

            color_sel = COLOR_AV0 if av_real == 0 else COLOR_AV1

            # Línea vertical
            fig.add_vline(
                x=valor,

                line_width=4,

                line_dash='dash',

                line_color=color_sel
            )

            # Marcador
            fig.add_trace(go.Scatter(

                x=[valor],
                y=[0],

                mode='markers',

                marker=dict(
                    size=16,
                    color=color_sel,
                    symbol='diamond-open',
                    line=dict(
                        width=3,
                        color='black'
                    )
                ),

                name=f'Paciente {paciente_id}',

                hovertemplate=
                f"<b>Paciente {paciente_id}</b><br>" +
                f"{variable}: {valor:.2f}<extra></extra>"
            ))

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        title={
            'text': f'<b>Distribución: {variable}</b>',
            'x': 0.5
        },

        template='plotly_white',

        height=400,

        bargap=0.05,

        xaxis=dict(
            title=variable
        ),

        yaxis=dict(
            title='Frecuencia'
        ),

        margin=dict(
            l=40,
            r=20,
            t=70,
            b=40
        )
    )

    return fig

def construir_radar(paciente_id=None):

    # --------------------------------------------------------
    # Marcadores
    # --------------------------------------------------------

    marker_cols = [
        c for c in MARCADORES
        if c.lower() not in ['edad', 'sexo']
    ]

    # --------------------------------------------------------
    # Normalización global Min-Max
    # --------------------------------------------------------

    df_norm = df[marker_cols].copy()

    for col in marker_cols:

        mn = df[col].min()
        mx = df[col].max()

        df_norm[col] = (
            (df[col] - mn) / (mx - mn)
        ) * 100

    # --------------------------------------------------------
    # Medias grupos
    # --------------------------------------------------------

    mean0 = df_norm[df['AV'] == 0].mean()
    mean1 = df_norm[df['AV'] == 1].mean()

    labels = marker_cols

    fig = go.Figure()

    # --------------------------------------------------------
    # Grupo AV=0
    # --------------------------------------------------------

    vals0 = mean0.tolist()
    vals0 += [vals0[0]]

    fig.add_trace(go.Scatterpolar(

        r=vals0,

        theta=labels + [labels[0]],

        fill='toself',

        name=LABEL_AV0,

        line=dict(
            color=COLOR_AV0,
            width=2
        ),

        fillcolor='rgba(76,114,176,0.20)'
    ))

    # --------------------------------------------------------
    # Grupo AV=1
    # --------------------------------------------------------

    vals1 = mean1.tolist()
    vals1 += [vals1[0]]

    fig.add_trace(go.Scatterpolar(

        r=vals1,

        theta=labels + [labels[0]],

        fill='toself',

        name=LABEL_AV1,

        line=dict(
            color=COLOR_AV1,
            width=2
        ),

        fillcolor='rgba(221,132,82,0.20)'
    ))

    # --------------------------------------------------------
    # Paciente seleccionado
    # --------------------------------------------------------

    if paciente_id is not None:

        idx = df[df['PACIENTES'] == paciente_id].index

        if len(idx) > 0:

            vals_p = df_norm.loc[idx[0], marker_cols].tolist()
            vals_p += [vals_p[0]]

            fig.add_trace(go.Scatterpolar(

                r=vals_p,

                theta=labels + [labels[0]],

                fill='toself',

                name=f'Paciente {paciente_id}',

                line=dict(
                    color=COLOR_PACIENTE,
                    width=3
                ),

                fillcolor='rgba(44,160,44,0.15)'
            ))

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        title={
            'text': '<b>Perfil normalizado de marcadores</b>',
            'x': 0.5
        },


        template='plotly_white',

        height=500,

        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        ),

        legend=dict(
            x=0.02,
            y=1.1
        )
    )

    return fig

# ============================================================
# APP
# ============================================================

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)

# ============================================================
# FIGURA PCA
# ============================================================

def construir_pca(paciente_id):

    fig = go.Figure()

    # --------------------------------------------------------
    # PUNTOS AV = 0
    # --------------------------------------------------------

    df0 = df_pca[df_pca['AV'] == 0]

    fig.add_trace(go.Scatter(
        x=df0['PC1'],
        y=df0['PC2'],
        mode='markers',
        name=LABEL_AV0,

        marker=dict(
            size=10,
            color=COLOR_AV0,
            opacity=0.8,
            line=dict(
                width=0.5,
                color='white'
            )
        ),

        text=df0['PACIENTES'],

        hovertemplate=
        "<b>%{text}</b><br>" +
        "PC1: %{x:.2f}<br>" +
        "PC2: %{y:.2f}<extra></extra>"
    ))

    # --------------------------------------------------------
    # PUNTOS AV = 1
    # --------------------------------------------------------

    df1 = df_pca[df_pca['AV'] == 1]

    fig.add_trace(go.Scatter(
        x=df1['PC1'],
        y=df1['PC2'],
        mode='markers',
        name=LABEL_AV1,

        marker=dict(
            size=10,
            color=COLOR_AV1,
            opacity=0.8,
            line=dict(
                width=0.5,
                color='white'
            )
        ),

        text=df1['PACIENTES'],

        hovertemplate=
        "<b>%{text}</b><br>" +
        "PC1: %{x:.2f}<br>" +
        "PC2: %{y:.2f}<extra></extra>"
    ))

    # --------------------------------------------------------
    # PACIENTE DESTACADO
    # --------------------------------------------------------

    fila = df_pca[df_pca['PACIENTES'] == paciente_id]

    if not fila.empty:

        av_real = int(fila['AV'].values[0])

        label_real = LABEL_AV0 if av_real == 0 else LABEL_AV1

        fig.add_trace(go.Scatter(
            x=fila['PC1'],
            y=fila['PC2'],

            mode='markers',

            name=f'Paciente {paciente_id}',

            marker=dict(
                size=20,
                color=COLOR_PACIENTE,
                symbol='circle-open',
                line=dict(
                    width=3,
                )
            ),

            hovertemplate=
            f"<b>Paciente {paciente_id}</b><br>" +
            f"{label_real}<br>" +
            "PC1: %{x:.2f}<br>" +
            "PC2: %{y:.2f}<extra></extra>"
        ))

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        title={
            'text': '<b>PCA de marcadores cardíacos</b>',
            'x': 0.5
        },

        xaxis=dict(
            title='Componente Principal 1'
        ),

        yaxis=dict(
            title='Componente Principal 2'
        ),

        template='plotly_white',

        hovermode='closest',

        height=800,

        legend=dict(
            x=0.02,
            y=0.98,
            font=dict(size=16)
        ),

        margin=dict(
            l=50,
            r=20,
            t=80,
            b=50
        )
    )

    return fig


# ============================================================
# TABLA PACIENTE
# ============================================================

def crear_tabla_paciente(paciente_id):

    fila = df[df['PACIENTES'] == paciente_id]

    if fila.empty:
        return html.Div("Paciente no encontrado")

    av_real = int(fila['AV'].values[0])

    color_real = COLOR_AV0 if av_real == 0 else COLOR_AV1

    rows = []

    for marcador in MARCADORES_PCA:

        valor = fila[marcador].values[0]

        media0 = df[df['AV'] == 0][marcador].mean()
        media1 = df[df['AV'] == 1][marcador].mean()

        rows.append(html.Tr([
            html.Td(marcador),
            html.Td(f"{valor:.2f}"),
            html.Td(f"{media0:.2f}"),
            html.Td(f"{media1:.2f}")
        ]))

    return html.Div([

        html.H4(
            f"Paciente seleccionado: {paciente_id}",
            style={'color': color_real}
        ),

        html.Table([

            html.Thead(
                html.Tr([
                    html.Th("Marcador"),
                    html.Th("Paciente"),
                    html.Th("Media AV=0"),
                    html.Th("Media AV=1")
                ])
            ),

            html.Tbody(rows)

        ],

        style={
            'width': '100%',
            'borderCollapse': 'collapse'
        })

    ])


# ============================================================
# LAYOUT
# ============================================================

app.layout = html.Div([

    html.H1(
        "Dashboard Arritmias",
        style={
            'textAlign': 'center',
            'marginBottom': '30px'
        }
    ),

    html.Div([

        # =====================================================
        # IZQUIERDA - PCA
        # =====================================================

        html.Div([

            dcc.Graph(
                id='graph-pca',
                clear_on_unhover=False
            )

        ],

        style={
            'width': '58%',
            'display': 'inline-block',
            'verticalAlign': 'top'
        }),

        # =====================================================
        # DERECHA
        # =====================================================

        html.Div([

            html.Div([

                html.Label(
                    "Variables",
                    style={
                        'fontWeight': 'bold',
                        'marginBottom': '10px'
                    }
                ),

                dcc.Dropdown(

                    id='dropdown-variable',

                    options=[
                        {
                            'label': v,
                            'value': v
                        }
                        for v in VARIABLES_HISTOGRAMA
                    ],

                    value='EDAD',

                    clearable=False
                )

            ],

            style={
                'marginBottom': '10px'
            }),

            dcc.Graph(
                id='histograma-variable'
            ),

            dcc.Graph(
                id='radar-plot'
            )

        ],

        style={
            'width': '40%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'paddingLeft': '20px'
        })

    ])

])


# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    Output('graph-pca', 'figure'),
    Input('graph-pca', 'clickData')
)
def update_graph(clickData):

    paciente_id = None

    # --------------------------------------------------------
    # Si el usuario hace click en un punto
    # --------------------------------------------------------

    if clickData is not None:

        paciente_id = clickData['points'][0]['text']

    return construir_pca(paciente_id)


@app.callback(
    Output('tabla-paciente', 'children'),
    Input('dropdown-paciente', 'value')
)
def update_tabla(paciente_id):

    return crear_tabla_paciente(paciente_id)

@app.callback(
    Output('histograma-variable', 'figure'),

    [
        Input('graph-pca', 'clickData'),
        Input('dropdown-variable', 'value')
    ]
)
def update_histograma(clickData, variable):

    paciente_id = None

    if clickData is not None:

        paciente_id = clickData['points'][0]['text']

    return construir_histograma(variable, paciente_id)


@app.callback(
    Output('radar-plot', 'figure'),
    Input('graph-pca', 'clickData')
)
def update_radar(clickData):

    paciente_id = None

    if clickData is not None:

        paciente_id = clickData['points'][0]['text']

    return construir_radar(paciente_id)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)