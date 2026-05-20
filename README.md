# Maestría en Inteligencia Artificial — Explicabilidad en Machine Learning (XAI)

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.51-informational.svg)
![LIME](https://img.shields.io/badge/LIME-latest-yellowgreen.svg)
![Estado](https://img.shields.io/badge/status-completado-success.svg)

Este repositorio contiene la implementación completa del taller práctico de **Técnicas de Explicabilidad (XAI)** aplicadas a la predicción de rendimiento académico estudiantil. El proyecto cubre desde la preparación del dataset y el entrenamiento del modelo base hasta el análisis comparativo de cuatro técnicas XAI, con especial atención a los riesgos éticos del despliegue en contextos educativos.

---
## 👤 Equipo de Trabajo
- **Harold Rodrigo Angulo Arellano**
- **Melissa Figallo Sánchez**
- **Jorge Javier Maldonado Mahauad**
- **Paula Elizabeth Noboa Ramírez**
---

## 📋 Tabla de Contenidos

1. Descripción del Problema
2. Metodología
3. Técnicas XAI Implementadas
4. Resultados Clave 
5. Dataset 
6. Riesgos Éticos Identificados 
7. Referencias 
8. Licencia 

---




## 🚀 Descripción del Problema

El desafío consiste en **predecir la nota final** de 1.070 estudiantes universitarios (variable continua, rango 0–100) a partir de 12 variables académicas y contextuales. El objetivo no es solo obtener una métrica de precisión, sino entender *cómo* y *por qué* el modelo toma cada decisión, de modo que los resultados sean accionables para docentes y coordinadores académicos.

**Variable objetivo:** `nota_final` (regresión supervisada continua)  
**Modelo base:** Random Forest Regressor — R² = 0.71 · RMSE = 4.8 pts en test

> La pregunta central no es "¿qué tan preciso es el modelo?" sino "¿en qué casos falla, por qué, y qué sesgos puede estar replicando?"

---

## ⚙️ Metodología

El flujo de trabajo sigue las mejores prácticas de ML sin data leakage, en el orden obligatorio:

1. **Limpieza de categorías** — Normalización de encoding inconsistente (9 variantes → 3 categorías limpias en `participacion_clase`, 6 variantes → binario en `acceso_internet`, etc.) antes del split, ya que es limpieza de formato y no transformación estadística.
2. **Eliminación de leakage e IDs** — Se eliminan `resultado` (categorización directa de `nota_final`) y `alumno_id` (identificador sin valor predictivo).
3. **Eliminación de duplicados** — 20 filas duplicadas (1.87%) eliminadas antes del split para evitar que una misma observación aparezca en train y test.
4. **Split Train/Test** — División 80/20 con `random_state=42`. **Línea divisoria crítica**: todo ajuste estadístico ocurre después de este paso.
5. **Imputación** *(fit solo en train)* — Mediana para variables numéricas con nulos (~9–10%), moda para variables binarias y categóricas.
6. **Encoding** *(fit solo en train)* — OrdinalEncoder para variables con orden natural (`participacion_clase`, `nivel_socioeconomico`); OneHotEncoder con `drop='first'` para `modalidad` (nominal).
7. **Escalado** *(fit solo en train)* — StandardScaler (Z-score), apropiado para modelos lineales y consistente entre todos los modelos comparados.
8. **Análisis XAI** — Cuatro técnicas aplicadas de forma complementaria sobre el modelo entrenado.

---

## 🔍 Técnicas XAI Implementadas

Se aplicaron cuatro técnicas complementarias para obtener una visión completa de las decisiones del modelo:

| Técnica | Alcance | Tipo | Descripción |
|---------|---------|------|-------------|
| **SHAP** (TreeExplainer) | Global + Local | Exacto | Valores de Shapley: contribución marginal de cada feature con garantías matemáticas (eficiencia, simetría, nulidad). Genera beeswarm, bar plot, waterfall y dependence plots. |
| **LIME** | Local | Aproximado | Aproxima el comportamiento del modelo localmente con una regresión lineal sobre una vecindad perturbada. Útil para comunicar casos individuales a usuarios no técnicos. |
| **Permutation Feature Importance (PFI)** | Global | Agnóstico | Mide la degradación del RMSE en test al permutar cada feature. Más robusto que la importancia Gini nativa del RF, que sobreestima features continuos de alta cardinalidad. |
| **Partial Dependence Plots (PDP)** | Global | Agnóstico | Efecto marginal promedio de uno o dos features sobre la predicción, manteniendo los demás en su distribución marginal. Incluye PDP 2D para capturar interacciones. |

---

## 📊 Resultados Clave

### Comparación de Modelos

| Modelo | RMSE Train | RMSE Test | R² Train | R² Test | ¿Overfitting? |
|--------|-----------|-----------|---------|---------|--------------|
| Baseline (media) | ~18.4 | ~18.5 | ~0.00 | ~0.00 | N/A |
| Ridge Regression | ~5.0 | ~5.2 | ~0.92 | ~0.92 | No |
| **Random Forest** | ~1.8 | ~4.8 | ~0.99 | **0.71** | Leve |

### Importancia de Features (Consenso de 4 Técnicas)

| Ranking | Feature | SHAP | PFI | Gini-RF | Ridge \|β\| |
|---------|---------|------|-----|---------|------------|
| **#1** | `promedio_examenes_parciales` | #1 | #1 | #1 | #1 |
| **#2** | `horas_estudio_sem` | #2 | #2 | #2 | #3 |
| **#3** | `asistencia_pct` | #3 | #3 | #3 | #2 |
| **#4** | `num_reprobadas_previas` | #4 | #4 | #4 | #4 |

> **Hallazgo crítico:** `promedio_examenes_parciales` concentra aproximadamente el **65% de la importancia SHAP total**. Esto es una restricción operativa importante: en la semana 1 del semestre, esa variable no existe, lo que limita severamente la capacidad de alerta temprana del modelo.

### Concordancia entre Técnicas

Las 4 técnicas coinciden exactamente en el ranking del top-4, lo que valida con alta confianza que esas variables son genuinamente determinantes del rendimiento académico y no artefactos de una sola técnica de análisis.

---

## 📁 Dataset

**Archivo:** `DS15_rendimiento_academico.csv`

El dataset contiene registros de 1.070 estudiantes universitarios con las siguientes variables:

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `alumno_id` | ID | Identificador único del estudiante (eliminado en preprocesamiento) |
| `horas_estudio_sem` | Numérica | Horas de estudio por semana |
| `asistencia_pct` | Numérica | Porcentaje de asistencia a clases |
| `participacion_clase` | Categórica ordinal | Nivel de participación: Baja / Media / Alta |
| `num_tareas_entregadas` | Numérica | Número de tareas entregadas en el período |
| `promedio_examenes_parciales` | Numérica | Promedio de notas en exámenes parciales |
| `acceso_internet` | Binaria | Acceso a internet: Si (1) / No (0) |
| `nivel_socioeconomico` | Categórica ordinal | Nivel socioeconómico: Bajo / Medio / Alto |
| `horas_trabajo_sem` | Numérica | Horas de trabajo remunerado por semana |
| `num_reprobadas_previas` | Numérica | Número de materias reprobadas en semestres anteriores |
| `modalidad` | Categórica nominal | Modalidad: presencial / en línea / híbrido |
| `tutor_asignado` | Binaria | Tiene tutor asignado: 1 / 0 |
| `nota_final` | Numérica (target) | Nota final del período (0–100) |
| `resultado` | Categórica | Categorización de nota_final — **LEAKAGE, eliminado** |

**Problemas de calidad detectados:**
- Encoding inconsistente en 4 columnas categóricas (hasta 9 variantes para 3 categorías)
- Valores nulos en 4 columnas (8.32%–11.03%)
- 20 filas completamente duplicadas (1.87%)
- Leakage directo: columna `resultado` es una categorización de `nota_final`

---
## ⚠️ Riesgos Éticos Identificados

El análisis XAI permitió identificar cuatro riesgos concretos para el despliegue en producción:

1. **Variable con peso excesivo** — `promedio_examenes_parciales` concentra ~65% de la importancia. En la semana 1 del semestre esa variable no existe, haciendo al modelo casi inútil como sistema de alerta temprana sin comunicar esa restricción temporal.

2. **Sesgo socioeconómico** — `nivel_socioeconomico` tiene impacto estadístico medible. Un sistema de asignación de recursos (becas, tutorías) basado en este modelo podría perpetuar desventajas estructurales al etiquetar sistemáticamente a estudiantes de bajos recursos como "en riesgo" (O'Neil, 2016).

3. **Brecha digital replicada** — El modelo aprende que carecer de acceso a internet correlaciona con notas bajas. En producción, replicaría esa brecha en lugar de compensarla.

4. **Falsa objetividad** — Un R² de 0.71 con RMSE de 4.8 pts no justifica presentar las predicciones como certezas. Las decisiones académicas importantes requieren intervalos de confianza y juicio humano.

**Recomendaciones para implementación responsable:**
- Segmentar el modelo por semana del semestre (antes y después de la primera evaluación parcial)
- Medir el RMSE desagregado por `nivel_socioeconomico`, `acceso_internet` y `modalidad`
- Generar automáticamente un waterfall SHAP por estudiante para cada predicción
- Mantener supervisión humana obligatoria en decisiones que afecten al estudiante

---

## 📚 Referencias

- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. NeurIPS 30. https://shap.readthedocs.io
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. KDD 2016. https://marcotcr.github.io/lime/
- Fisher, A., Rudin, C., & Dominici, F. (2019). *All models are wrong, but many are useful*. Journal of Machine Learning Research.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
- O'Neil, C. (2016). *Weapons of Math Destruction*. Crown Books.
- Maldonado, J. (2017). *Flipping the classroom with MOOCs*. Proceedings of LACLO 2017.

---
 
## ⚖️ Licencia

Este proyecto se distribuye bajo la licencia **MIT**.

*Fuente de datos: Dataset DS15 — Rendimiento Académico Estudiantil · Maestría en Inteligencia Artificial MIAR0525 · Mayo 2026.*
