"""Pipeline modular de features y preprocesamiento sin fuga para PSet #3.

Sigue la estructura estándar del curso (pipeline_proyecto.py):
    from src.pipeline import cargar_y_dividir, construir_preparacion, calcular_metricas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from src.datos import (
        dividir_train_test,
        obtener_metadatos_snapshot,
        preparar_universo_nowcasting,
    )
except ImportError:
    from datos import (
        dividir_train_test,
        obtener_metadatos_snapshot,
        preparar_universo_nowcasting,
    )

OBJETIVO = "ventas_reales_definitivo_ipc_real"

NUMERICAS = [
    "ventas_temprano_ipc_real",
    "NUM_DECLARANTES",
    "ventas_temprano_por_declarante_real",
    "INDICE_PRECIO_REAL",
    "inflacion_sectorial_12m_lag1",
    "RATIO_REPRESENTATIVIDAD_HIST_12M",
    "proyeccion_cobertura_12m",
    "ratio_temprano_vs_lag12",
    "ratio_temprano_vs_lag1",
    "ventas_temprano_real_lag1_musd",
    "ventas_temprano_real_lag12_musd",
    "ventas_definitivo_real_lag2_musd",
    "ventas_definitivo_real_lag12_musd",
    "impulso_declarantes",
]

CATEGORICAS = [
    "CODIGO_INDUSTRIA",
    "SECTOR_MACRO",
    "SECTOR_ECONOMICO",
]

YA_LISTAS = [
    "mes_sin",
    "mes_cos",
    "es_diciembre",
    "TRIMESTRE",
    "FLAG_VENTAS_CERO_CON_DECLARANTES",
]

COLUMNAS_MODELO = NUMERICAS + CATEGORICAS + YA_LISTAS


def agregar_features(datos: pd.DataFrame) -> pd.DataFrame:
    """Construye las features derivadas disponibles en el corte temprano (día 15 del mes t).

    Todas las transformaciones usan exclusivamente información observada hasta el día 15
    del mes t (ventas tempranas, índice real de precios por rama y rezagos históricos).
    """
    df = datos.copy()

    # 1. Precios reales: deflactar ventas tempranas con el índice real CNT46 (en M USD)
    deflactor = df["INDICE_PRECIO_REAL"] / 100.0
    df["ventas_temprano_ipc_real"] = (df["VENTAS_NOMINALES_USD"] / deflactor) / 1e6
    df["ventas_temprano_por_declarante_real"] = (
        df["VENTAS_POR_DECLARANTE_USD"] / deflactor
    ) / 1e3
    df["inflacion_sectorial_12m_lag1"] = np.where(
        df["indice_precio_real_lag13"] > 0,
        (df["indice_precio_real_lag1"] / df["indice_precio_real_lag13"] - 1.0) * 100.0,
        np.nan,
    )

    # 2. Escala y cobertura histórica (acotada de forma segura en [0.05, 1.0])
    cobertura_segura = df["RATIO_REPRESENTATIVIDAD_HIST_12M"].clip(lower=0.05, upper=1.0)
    df["proyeccion_cobertura_12m"] = np.where(
        cobertura_segura.notna(),
        df["ventas_temprano_ipc_real"] / cobertura_segura,
        np.nan,
    )
    df["ratio_temprano_vs_lag12"] = np.where(
        df["ventas_temprano_real_lag12_musd"] > 0,
        (df["ventas_temprano_ipc_real"] / df["ventas_temprano_real_lag12_musd"]).clip(0.0, 10.0),
        np.nan,
    )
    df["ratio_temprano_vs_lag1"] = np.where(
        df["ventas_temprano_real_lag1_musd"] > 0,
        (df["ventas_temprano_ipc_real"] / df["ventas_temprano_real_lag1_musd"]).clip(0.0, 10.0),
        np.nan,
    )

    # 3. Estacionalidad cíclica
    df["mes_sin"] = np.sin(2.0 * np.pi * df["MES"] / 12.0)
    df["mes_cos"] = np.cos(2.0 * np.pi * df["MES"] / 12.0)
    df["es_diciembre"] = (df["MES"] == 12).astype(int)

    # 4. Dinámica de contribuyentes declarantes frente a su media móvil histórica [t-12, t-1]
    df["impulso_declarantes"] = np.where(
        df["media_movil_declarantes_12m_lag1"] > 0,
        (df["NUM_DECLARANTES"] / df["media_movil_declarantes_12m_lag1"]).clip(0.0, 5.0),
        np.nan,
    )

    df["TRIMESTRE"] = df["TRIMESTRE"].astype(int)
    df["FLAG_VENTAS_CERO_CON_DECLARANTES"] = df["FLAG_VENTAS_CERO_CON_DECLARANTES"].astype(int)
    return df


def cargar_y_dividir(
    forzar_snowflake: bool = False,
    columnas_extra: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga el universo sin fuga, construye features y aplica la división temporal única."""
    universo = preparar_universo_nowcasting(forzar_snowflake=forzar_snowflake)
    datos = agregar_features(universo)
    train_df, test_df = dividir_train_test(datos)

    cols = COLUMNAS_MODELO + [c for c in columnas_extra if c not in COLUMNAS_MODELO]
    X_ent = train_df[cols].copy()
    y_ent = train_df[OBJETIVO].astype(float).copy()
    X_prueba = test_df[cols].copy()
    y_prueba = test_df[OBJETIVO].astype(float).copy()
    return X_ent, X_prueba, y_ent, y_prueba


def construir_preparacion(codificacion: str = "onehot") -> ColumnTransformer:
    """Construye el ColumnTransformer sin fuga (imputación + escalado + codificación).

    Args:
        codificacion: 'onehot' (OneHotEncoder, por defecto para comparación justa) o
                      'ordinal' (OrdinalEncoder, alternativa compacta para árboles).
    """
    if codificacion == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif codificacion == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        raise ValueError(f"Codificación no soportada: {codificacion}")

    return ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                NUMERICAS,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("codificar", encoder),
                    ]
                ),
                CATEGORICAS,
            ),
            ("ya_listas", "passthrough", YA_LISTAS),
        ],
        remainder="drop",
    )


def calcular_wape_pct(y_real: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    """Métrica macroeconómica: WAPE (%) ponderado por volumen real de ventas."""
    real = np.asarray(y_real, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    denom = np.sum(np.abs(real))
    if denom == 0:
        return float("nan")
    return float(100.0 * np.sum(np.abs(real - pred)) / denom)


def calcular_costo_negocio_musd(y_real: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    """Métrica de negocio asimétrica (M USD reales por rama-mes).

    Penaliza 1.5x la sobreestimación (y_pred > y_real: falso optimismo que retrasa
    alertas de desaceleración en el IMAEC Flash) y 1.0x la subestimación (sesgo conservador).
    """
    real = np.asarray(y_real, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    diff = pred - real
    costo = np.where(diff > 0, 1.5 * diff, 1.0 * (-diff))
    return float(np.mean(costo))


def calcular_metricas(y_real: np.ndarray | pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Devuelve el diccionario estándar de métricas técnicas y de negocio."""
    real = np.asarray(y_real, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "mae": float(mean_absolute_error(real, pred)),
        "rmse": float(root_mean_squared_error(real, pred)),
        "r2": float(r2_score(real, pred)),
        "wape_pct": calcular_wape_pct(real, pred),
        "costo_negocio_musd": calcular_costo_negocio_musd(real, pred),
    }


class HeuristicaBCE(BaseEstimator, RegressorMixin):
    """Línea base de negocio (PSet #1): extrapolación directa por cobertura histórica 12m.

    Predice `proyeccion_cobertura_12m` = ventas_temprano_ipc_real / cobertura_hist_12m,
    usando la mediana observada en entrenamiento como respaldo si falta la ventana de 12m.
    """

    def __init__(self, columna: str = "proyeccion_cobertura_12m") -> None:
        self.columna = columna

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "HeuristicaBCE":
        if y is not None:
            self.fallback_mediana_ = float(np.nanmedian(np.asarray(y, dtype=float)))
        else:
            self.fallback_mediana_ = float(np.nanmedian(X[self.columna].to_numpy(dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        vals = X[self.columna].to_numpy(dtype=float)
        pred = np.where(np.isnan(vals), self.fallback_mediana_, vals)
        return np.clip(pred, 0.0, None)


if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    meta = obtener_metadatos_snapshot()
    prep = construir_preparacion()
    X_ent_prep = prep.fit_transform(X_ent)
    X_prueba_prep = prep.transform(X_prueba)
    print("Metadatos snapshot:", meta)
    print("Entrenamiento:", X_ent.shape, "-> preparado:", X_ent_prep.shape)
    print("Prueba (hold-out):", X_prueba.shape, "-> preparado:", X_prueba_prep.shape)
    print("Target train mean/median (M USD reales):", round(float(y_ent.mean()), 2), "/", round(float(y_ent.median()), 2))
    print("Columnas preparadas:", len(prep.get_feature_names_out()))
