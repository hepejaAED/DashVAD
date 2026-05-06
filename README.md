# Dashboard Interactivo de Marcadores Pro-Arrítmicos
## Guía de División de Trabajo

---

## DESCRIPCIÓN DEL PROYECTO

Crear un dashboard interactivo con **Dash en Python** para visualizar datos de pacientes con marcadores pro-arrítmicos.

**Dataset:** `Arritmias.csv`

**Objetivo:** Dashboard de página única con dos secciones principales.

---

## DIVISIÓN DE TRABAJO

### **1. Página Orientada al Paciente**

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



---

### **2. Análisis de Grupos y Relaciones**

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



---

## ESTRUCTURA GENERAL

### Archivos principales:
```
proyecto/
├── Arritmias.csv
├── app.py (archivo principal - COMPARTIDO)
├── persona1_paciente.py
├── persona2_grupos.py
└── README.md 
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

