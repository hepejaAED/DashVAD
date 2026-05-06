# Dashboard Interactivo de Marcadores Pro-Arrítmicos
## Guía de División de Trabajo

---

## 📋 DESCRIPCIÓN DEL PROYECTO

Crear un dashboard interactivo con **Dash en Python** para visualizar datos de pacientes con marcadores pro-arrítmicos.

**Dataset:** `Arritmias.csv`

**Objetivo:** Dashboard de página única con dos secciones principales.

---

## 👥 DIVISIÓN DE TRABAJO

### **PERSONA 1: Página Orientada al Paciente**

#### Responsabilidades:
1. **Selector de Paciente**
   - Dropdown para elegir paciente por ID
   - Actualizar automáticamente todos los gráficos

2. **Tabla de Datos**
   - Mostrar todos los marcadores del paciente seleccionado:
     - LV Mass(g)
     - BZ+Core (g)
     - BZ (g, %)
     - Core (g, %)
     - Channel_Mass (g)
     - LVEF
     - Edad, Sexo
     - AV (0 o 1)

3. **Gráficos Comparativos** (mínimo 2):
   - Radar chart: Marcadores del paciente vs. promedio grupo
   - Gráfico de barras: Cada marcador del paciente vs. media (con rango)
   - Indicador visual de riesgo (AV: sí/no)

#### Archivos a crear:
- `persona1_paciente.py` (contiene la función que retorna el layout)

#### Estructura esperada:
```python
def crear_layout_paciente(df):
    # Retorna html.Div con toda la sección
    return html.Div([...])
```

---

### **PERSONA 2: Análisis de Grupos y Relaciones**

#### Responsabilidades:
1. **Distribución General de Marcadores**
   - Histogramas o boxplots para cada marcador principal
   - Desplegable para seleccionar qué marcador visualizar

2. **Matriz de Correlaciones**
   - Heatmap con correlaciones entre todos los marcadores
   - Código de colores (rojo/azul)

3. **Comparativa por Grupo (AV: 0 vs 1)**
   - Violinplot o boxplot comparativo para cada marcador
   - Mostrar diferencias entre pacientes sin arritmia y con arritmia

4. **Análisis Demográfico** (opcional pero recomendado)
   - Distribución por Edad
   - Distribución por Sexo
   - Cualquier patrón relevante

#### Archivos a crear:
- `persona2_grupos.py` (contiene la función que retorna el layout)

#### Estructura esperada:
```python
def crear_layout_grupos(df):
    # Retorna html.Div con toda la sección
    return html.Div([...])
```

---

## 🔧 ESTRUCTURA TÉCNICA GENERAL

### Archivos principales:
```
proyecto/
├── Arritmias.csv
├── app.py (archivo principal - COMPARTIDO)
├── persona1_paciente.py
├── persona2_grupos.py
└── README.md (este archivo)
```

### `app.py` - Estructura base (TODOS USAN ESTO)
```python
import dash
from dash import dcc, html, callback, Input, Output
import pandas as pd
from persona1_paciente import crear_layout_paciente
from persona2_grupos import crear_layout_grupos

# Cargar datos
df = pd.read_csv('Arritmias.csv')

# Crear app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout principal con dos secciones
app.layout = html.Div([
    html.H1("Dashboard: Marcadores Pro-Arrítmicos", style={'textAlign': 'center'}),
    
    html.Div([
        html.H2("Página del Paciente"),
        crear_layout_paciente(df)
    ], style={'padding': '20px', 'borderBottom': '2px solid #ccc'}),
    
    html.Div([
        html.H2("Análisis de Grupos"),
        crear_layout_grupos(df)
    ], style={'padding': '20px'})
])

# CALLBACKS PERSONA 1 (aquí va el código de Persona 1)

# CALLBACKS PERSONA 2 (aquí va el código de Persona 2)

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📊 COLUMNAS DEL DATASET

| Columna | Tipo | Descripción |
|---------|------|-------------|
| LV Mass(g) | numérico | Masa ventrículo izquierdo (gramos) |
| BZ+Core (g) | numérico | Zona infartada + borde (gramos) |
| BZ (g) | numérico | Zona de borde (gramos) |
| BZ (%) | numérico | Zona de borde (porcentaje) |
| Core (g) | numérico | Zona infartada (gramos) |
| Core (%) | numérico | Zona infartada (porcentaje) |
| Channel_Mass (g) | numérico | Masa de canales (gramos) |
| LVEF | numérico | Fracción eyección ventrículo |
| Edad | numérico | Edad del paciente |
| Sexo | categórico | M/F o 0/1 |
| AV | binario | 0=sin arritmia, 1=con arritmia |

---

## 💡 INSTRUCCIONES PARA CADA PERSONA

### Persona 1 - Copia esto en tu Claude:
```
Eres experto en Dash/Python. 

TAREA: Crear la sección "Página del Paciente" de un dashboard.

REQUISITOS:
1. Archivo: persona1_paciente.py
2. Función: crear_layout_paciente(df) que retorna html.Div
3. Contener:
   - Dropdown para seleccionar paciente
   - Tabla mostrando marcadores del paciente
   - 2+ gráficos comparativos (paciente vs grupo)
   - Indicador visual de riesgo (AV)
4. Usar callbacks para actualizaciones interactivas
5. Incluir IDs HTML descriptivos (ej: 'dropdown-paciente', 'tabla-paciente')

DATASET: Arritmias.csv con columnas: LV Mass(g), BZ+Core (g), BZ (g, %), 
Core (g, %), Channel_Mass (g), LVEF, Edad, Sexo, AV

NOTA: El df se pasa como parámetro a la función.
```

### Persona 2 - Copia esto en tu Claude:
```
Eres experto en Dash/Python.

TAREA: Crear la sección "Análisis de Grupos" de un dashboard.

REQUISITOS:
1. Archivo: persona2_grupos.py
2. Función: crear_layout_grupos(df) que retorna html.Div
3. Contener:
   - Visualización de distribución (histogramas/boxplots seleccionables)
   - Matriz de correlaciones (heatmap)
   - Comparativa AV=0 vs AV=1 (violinplot/boxplot)
   - Análisis por Edad/Sexo (opcional)
4. Usar callbacks para interactividad
5. Incluir IDs HTML descriptivos (ej: 'dropdown-marcador', 'heatmap-correlacion')

DATASET: Arritmias.csv con columnas: LV Mass(g), BZ+Core (g), BZ (g, %), 
Core (g, %), Channel_Mass (g), LVEF, Edad, Sexo, AV

NOTA: El df se pasa como parámetro a la función.
```

---

## ESPECIFICACIONES DE DISEÑO

- **Tema:** CSS externo (codepen default)
- **Colores recomendados:** 
  - Rojo para AV=1 (riesgo)
  - Verde para AV=0 (sin riesgo)
- **Responsive:** Usar layouts de Dash responsivos

---

## CHECKLIST DE ENTREGA

- [ ] Persona 1: `persona1_paciente.py` funcional
- [ ] Persona 2: `persona2_grupos.py` funcional
- [ ] `app.py` integra ambas secciones
- [ ] Todos los callbacks funcionan correctamente
- [ ] Dashboard corre sin errores: `python app.py`
- [ ] Gráficos son interactivos y actualizan en tiempo real

---

## 🚀 CÓMO EJECUTAR

```bash
python app.py
```
Luego abrir: `http://127.0.0.1:8050/`

---

## 📝 NOTAS IMPORTANTES

1. **No duplicar código:** Cada persona trabaja en su archivo
2. **IDs únicos:** Todos los elementos deben tener IDs descriptivos y únicos
3. **Pasado de df:** El dataframe se pasa como parámetro, NO se carga dentro del archivo
4. **Callbacks:** Cada persona define sus propios callbacks en `app.py`
5. **Fusión final:** Al terminar, combinar callbacks en `app.py`

---

**Fecha:** [COMPLETAR]  
**Persona 1:** [NOMBRE]  
**Persona 2:** [NOMBRE]