# PSet #3 – Experimentación, Validación y Selección del Modelo con MLflow

Proyecto de experimentación reproducible de Machine Learning para el **nowcasting de ventas mensuales sectoriales (46 ramas CIIU/CNT)** a partir del corte temprano del Formulario 104 del SRI (`temprano_15d`, día 15 posterior al cierre fiscal), integrado con el índice real de precios por rama en **Snowflake** y trazado de extremo a extremo en **MLflow**.

---

## Resumen de Resultados

- **Fuentes de datos (Snowflake):** `PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS` (12,635 filas; 6,287 en `temprano_15d`) cruzada al 100% con `PSET2_DB.ANALYTICS.PRECIOS_RAMA_CNT46` (`data/precios.xlsx`, 46 ramas × 141 meses).
- **Partición temporal sin fuga:**
  - *Warm-up* causal (`lag_12`): `2015-01` a `2015-12` (552 filas)
  - Entrenamiento + CV (`TimeSeriesSplit(n_splits=5)` mensual): `2016-01` a `2024-12` (4,907 filas)
  - Prueba *hold-out* final (intocable hasta `05_seleccion.ipynb`): `2025-01` a `2026-06` (828 filas)
- **Modelo ganador (`@champion`):** `tuning_lasso` (`Lasso`, `alpha=3.6361`), versión **1** en el Model Registry `nowcasting-f104-modelo` (`run_id`: `b4bd242a500b42eea087a6cc9918097a`):
  - **MAE CV (5 pliegues):** `13.9173 ± 1.1121 M USD` reales
  - **MAE Hold-Out (828 filas):** `14.7905 M USD` reales (`RMSE = 27.1961 M`, `R² = 0.9995`)
  - **WAPE Hold-Out:** `3.267%` (error agregado mensual a nivel IMAEC Flash de las 46 ramas: `0.57%`)
  - **Costo asimétrico de negocio Hold-Out:** `18.6305 M USD` (latencia de inferencia: `3.784 ms` por lote de 828 filas)
- **Segundo mejor (`@challenger`):** `tuning_ridge` (`Ridge`, `alpha=0.4238`), versión **2** (`run_id`: `2ae1ddac80d24c5d92acc83b3f4f6b01`), con `MAE CV = 16.5067 ± 1.8505 M USD` y `MAE Hold-Out = 17.0578 M USD`.

---

## Estructura del Repositorio

```text
pset_3/
├── docker-compose.yml              # Servidor de tracking y Model Registry de MLflow (puerto 5000)
├── .env.example                    # Plantilla segura de credenciales Snowflake y MLflow
├── .gitignore
├── requirements.txt                # Dependencias reproducibles del entorno Python
├── README.md
├── data/
│   └── precios.xlsx                # Índice real de precios por rama CNT46 (46 ramas × 141 meses)
├── src/
│   ├── __init__.py
│   ├── datos.py                    # Carga desde Snowflake, caché determinista y partición temporal
│   └── pipeline.py                 # Features sin fuga, ColumnTransformer, HeuristicaBCE y métricas
├── notebooks/
│   ├── 01_eda.ipynb                # Exploración, atípicos (IQR/Z-score), Spearman vs. Pearson y hallazgos
│   ├── 02_features.ipynb           # Ingeniería de características sin fuga y codificación categórica
│   ├── 03_baseline_candidatos.ipynb# 2 líneas base, 5 familias de candidatos y evaluación LLM
│   ├── 04_tuning.ipynb             # Ajuste Optuna (3 modelos × 30 configuraciones anidadas = 90 corridas)
│   └── 05_seleccion.ipynb          # Hold-out único, Model Registry (@champion/@challenger) y exportación
└── docs/
    ├── corridas.csv                # Bitácora completa exportada desde MLflow (105 corridas)
    └── PSet3_memo_Ortega_Pasmay.pdf# Memorando técnico ejecutivo (6 páginas)
```

---

## 1. Configuración de Credenciales (Snowflake y MLflow)

Copie `.env.example` a `.env` en la raíz del proyecto y complete sus credenciales de Snowflake (`.env` está excluido del control de versiones):

```bash
cp .env.example .env            # Linux / macOS
```

```powershell
Copy-Item .env.example .env     # Windows PowerShell
```

Variables requeridas en `.env`:

- `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE` (`PSET2_DB`), `SNOWFLAKE_SCHEMA` (`ANALYTICS`)
- `MLFLOW_TRACKING_URI=http://localhost:5000`
- `MLFLOW_EXPERIMENT_NAME=nowcasting-f104-sri`

> **Nota de integridad:** El proyecto realiza únicamente consultas `SELECT` sobre `PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS` y `PSET2_DB.ANALYTICS.PRECIOS_RAMA_CNT46` (sin modificar tablas previas del PSet #2).

---

## 2. Entorno Python y Servidor MLflow en Docker

### 2.1 Instalar dependencias en entorno virtual (Python 3.12)

```bash
python -m venv .venv
# Activar entorno (.venv/Scripts/activate en Windows o source .venv/bin/activate en Linux/macOS)
pip install -r requirements.txt
python -m ipykernel install --user --name pset3-venv --display-name "Python 3 (.venv)"
```

### 2.2 Levantar el servidor de MLflow

```bash
docker compose up -d
```

Verifique que la interfaz y el endpoint de salud respondan en <http://localhost:5000> (`http://localhost:5000/health` → `OK`). El backend SQLite y los artefactos quedan persistidos en `./mlflow_data/`.

### 2.3 Verificar módulos base antes de los notebooks

```bash
python src/pipeline.py
```

Al ejecutarse por primera vez, `src/datos.py` consulta Snowflake, construye el snapshot local determinista `data/obt_precios_snapshot.parquet` (MD5 `e4cd092d7514`) e imprime las dimensiones preparadas (`(4907, 76)` en entrenamiento y `(828, 76)` en prueba).

---

## 3. Orden de Ejecución de los Notebooks

Ejecute cada notebook en orden estricto seleccionando el kernel `Python 3 (.venv)` y ejecutando **Restart Kernel & Run All** (o por CLI mediante `jupyter nbconvert`):

1. `notebooks/01_eda.ipynb` — Análisis exploratorio de datos, detección de atípicos, correlaciones y 4 hallazgos clave.
2. `notebooks/02_features.ipynb` — Validación de features derivadas al día 15 y comparación `OneHotEncoder` vs. `OrdinalEncoder`.
3. `notebooks/03_baseline_candidatos.ipynb` — Registro en MLflow de las 2 líneas base, 8 candidatos de 5 familias y evaluación formal de prompting LLM.
4. `notebooks/04_tuning.ipynb` — Búsqueda de hiperparámetros con Optuna (30 configuraciones anidadas por cada uno de los 3 mejores candidatos).
5. `notebooks/05_seleccion.ipynb` — Evaluación final única en *hold-out*, explicabilidad por permutación, registro de `@champion` y `@challenger`, verificación exacta desde el Model Registry y exportación de `docs/corridas.csv`.

---

## 4. Consumo del Modelo Ganador desde el Model Registry

Con el servidor de MLflow activo (`MLFLOW_TRACKING_URI=http://localhost:5000`), el pipeline completo (preprocesamiento + modelo entrenado) se recupera directamente por su alias:

```python
import os
import mlflow.sklearn
from src.pipeline import cargar_y_dividir, calcular_metricas

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
modelo_champion = mlflow.sklearn.load_model("models:/nowcasting-f104-modelo@champion")

_, X_prueba, _, y_prueba = cargar_y_dividir()
metricas = calcular_metricas(y_prueba, modelo_champion.predict(X_prueba))
print(metricas)  # mae: 14.7905 M USD reales, wape_pct: 3.267%
```
