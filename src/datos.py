"""Carga reproducible desde Snowflake (OBT + precios reales) y partición temporal sin fuga.

Fuentes en Snowflake (solo lectura, sin modificar tablas del PSet #2):
    - PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS
    - PSET2_DB.ANALYTICS.PRECIOS_RAMA_CNT46
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
RUTA_ENV = ROOT_DIR / ".env"
RUTA_SNAPSHOT = ROOT_DIR / "data" / "obt_precios_snapshot.parquet"

TABLA_ORIGEN = "PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS + PSET2_DB.ANALYTICS.PRECIOS_RAMA_CNT46"
FECHA_CORTE_SNAPSHOT = "2026-06-01"

FECHA_TRAIN_INICIO = pd.Timestamp("2016-01-01")
FECHA_TRAIN_FIN = pd.Timestamp("2024-12-01")
FECHA_TEST_INICIO = pd.Timestamp("2025-01-01")
FECHA_TEST_FIN = pd.Timestamp("2026-06-01")

# Columnas con información posterior al día 15 del mes t (prohibidas como predictores)
COLUMNAS_FUGA = [
    "RATIO_REPRESENTATIVIDAD_CORTE",
    "RATIO_CORTE_VS_HIST_12M",
    "VENTAS_NOMINALES_DEFINITIVO_MES",
    "VENTAS_REALES_DEFINITIVO_MES",
    "FLAG_TEMPRANO_MAYOR_DEFINITIVO",
]


def _cargar_entorno() -> None:
    if RUTA_ENV.exists():
        load_dotenv(RUTA_ENV, override=False)


def _conectar_snowflake() -> snowflake.connector.SnowflakeConnection:
    _cargar_entorno()
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "PSET2_DB"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS"),
    )


def cargar_obt_completa(forzar_snowflake: bool = False) -> pd.DataFrame:
    """Carga la OBT cruzada con el índice real de precios por rama.

    Usa caché determinista local en Parquet si existe (salvo forzar_snowflake=True)
    para asegurar reproducibilidad exacta y cómputo ágil en notebooks.
    """
    if RUTA_SNAPSHOT.exists() and not forzar_snowflake:
        df = pd.read_parquet(RUTA_SNAPSHOT)
        df["FECHA_MES"] = pd.to_datetime(df["FECHA_MES"])
        return df

    query = """
    SELECT
        o.*,
        LPAD(TRIM(o.CODIGO_INDUSTRIA), 3, '0') AS CODIGO_INDUSTRIA_PAD,
        p.INDICE_PRECIO_REAL
    FROM PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS o
    LEFT JOIN PSET2_DB.ANALYTICS.PRECIOS_RAMA_CNT46 p
        ON LPAD(TRIM(o.CODIGO_INDUSTRIA), 3, '0') = p.CODIGO_INDUSTRIA
       AND o.TIEMPO_KEY = p.TIEMPO_KEY
    ORDER BY LPAD(TRIM(o.CODIGO_INDUSTRIA), 3, '0'), o.TIEMPO_KEY, o.CORTE_KEY
    """
    conn = _conectar_snowflake()
    try:
        df = conn.cursor().execute(query).fetch_pandas_all()
    finally:
        conn.close()

    df["CODIGO_INDUSTRIA"] = df["CODIGO_INDUSTRIA_PAD"].astype(str)
    df = df.drop(columns=["CODIGO_INDUSTRIA_PAD"])
    df["FECHA_MES"] = pd.to_datetime(df["FECHA_MES"])
    df = df.sort_values(["CODIGO_INDUSTRIA", "TIEMPO_KEY", "CORTE_KEY"]).reset_index(drop=True)

    RUTA_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RUTA_SNAPSHOT, index=False, compression="snappy")
    return df


def obtener_huella_datos(ruta: Path = RUTA_SNAPSHOT) -> str:
    """Devuelve los primeros 12 caracteres del hash MD5 del snapshot local."""
    if not ruta.exists():
        cargar_obt_completa(forzar_snowflake=True)
    return hashlib.md5(ruta.read_bytes()).hexdigest()[:12]


def obtener_git_commit() -> str:
    """Devuelve el hash corto del commit actual de Git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip() or "sin-commit"
    except Exception:  # noqa: BLE001
        return "sin-commit"


def obtener_metadatos_snapshot() -> dict[str, str]:
    """Metadatos de trazabilidad exigidos en cada ejecución de MLflow."""
    return {
        "tabla_origen": TABLA_ORIGEN,
        "fecha_corte_snapshot": FECHA_CORTE_SNAPSHOT,
        "huella_datos": obtener_huella_datos(),
        "git_commit": obtener_git_commit(),
    }


def _rezago_por_indice_mes(
    df_base: pd.DataFrame,
    cols_valor: list[str],
    lag: int,
    sufijo: str,
) -> pd.DataFrame:
    """Une valores rezagados exactamente `lag` meses atrás por INDICE_MES (sin fuga posicional)."""
    aux = df_base[["CODIGO_INDUSTRIA", "INDICE_MES"] + cols_valor].copy()
    aux["INDICE_MES"] = aux["INDICE_MES"] + lag
    renombres = {c: f"{c}_{sufijo}" for c in cols_valor}
    return aux.rename(columns=renombres)


def preparar_universo_nowcasting(
    df_completa: pd.DataFrame | None = None,
    forzar_snowflake: bool = False,
) -> pd.DataFrame:
    """Construye el dataset causal del corte temprano (día 15) libre de data leakage.

    - Calcula rezagos históricos sobre INDICE_MES (precios en t-1 y t-13, ventas tempranas
      deflactadas reales en t-1 y t-12, ventas definitivas reales en t-2 y t-12, y promedio
      móvil de declarantes de los últimos 12 meses previos [t-12, t-1]).
    - Filtra TIPO_CORTE == 'temprano_15d' (6,287 filas).
    - Calcula el objetivo en millones de USD reales deflactados con INDICE_PRECIO_REAL:
      `ventas_reales_definitivo_ipc_real`.
    - Elimina explícitamente columnas con fuga futura (`COLUMNAS_FUGA`).
    """
    if df_completa is None:
        df_completa = cargar_obt_completa(forzar_snowflake=forzar_snowflake)

    df = df_completa.copy()
    df["FECHA_MES"] = pd.to_datetime(df["FECHA_MES"])

    # 1. Serie de precios reales por rama y mes (única por rama x INDICE_MES)
    precios_base = (
        df[["CODIGO_INDUSTRIA", "INDICE_MES", "INDICE_PRECIO_REAL"]]
        .drop_duplicates(subset=["CODIGO_INDUSTRIA", "INDICE_MES"])
        .rename(columns={"INDICE_PRECIO_REAL": "indice_precio_real"})
    )
    precios_lag1 = _rezago_por_indice_mes(precios_base, ["indice_precio_real"], 1, "lag1")
    precios_lag13 = _rezago_por_indice_mes(precios_base, ["indice_precio_real"], 13, "lag13")

    # 2. Serie de ventas definitivas reales pasadas (lag 2 y lag 12 meses: 100% cerradas al día 15)
    def_base = df[df["TIPO_CORTE"] == "definitivo"][
        ["CODIGO_INDUSTRIA", "INDICE_MES", "VENTAS_NOMINALES_USD", "INDICE_PRECIO_REAL"]
    ].copy()
    def_base["ventas_definitivo_real"] = (
        def_base["VENTAS_NOMINALES_USD"] / (def_base["INDICE_PRECIO_REAL"] / 100.0)
    ) / 1e6
    def_lag2 = _rezago_por_indice_mes(def_base, ["ventas_definitivo_real"], 2, "lag2_musd")
    def_lag12 = _rezago_por_indice_mes(def_base, ["ventas_definitivo_real"], 12, "lag12_musd")

    # 3. Serie de ventas tempranas reales pasadas y promedio móvil de declarantes [t-12, t-1]
    temp = df[df["TIPO_CORTE"] == "temprano_15d"].copy()
    temp["ventas_temprano_real"] = (
        temp["VENTAS_NOMINALES_USD"] / (temp["INDICE_PRECIO_REAL"] / 100.0)
    ) / 1e6

    temp_lag1 = _rezago_por_indice_mes(temp, ["ventas_temprano_real"], 1, "lag1_musd")
    temp_lag12 = _rezago_por_indice_mes(temp, ["ventas_temprano_real"], 12, "lag12_musd")

    # Malla calendario completa por rama x INDICE_MES para promedio móvil causal sin sesgo por huecos
    ramas = sorted(df["CODIGO_INDUSTRIA"].unique())
    indices = np.arange(int(df["INDICE_MES"].min()), int(df["INDICE_MES"].max()) + 1)
    malla = pd.MultiIndex.from_product(
        [ramas, indices], names=["CODIGO_INDUSTRIA", "INDICE_MES"]
    ).to_frame(index=False)
    malla = malla.merge(
        temp[["CODIGO_INDUSTRIA", "INDICE_MES", "NUM_DECLARANTES"]],
        on=["CODIGO_INDUSTRIA", "INDICE_MES"],
        how="left",
    ).sort_values(["CODIGO_INDUSTRIA", "INDICE_MES"])
    malla["media_movil_declarantes_12m_lag1"] = (
        malla.groupby("CODIGO_INDUSTRIA")["NUM_DECLARANTES"]
        .transform(lambda s: s.shift(1).rolling(window=12, min_periods=3).mean())
    )
    malla = malla[["CODIGO_INDUSTRIA", "INDICE_MES", "media_movil_declarantes_12m_lag1"]]

    # 4. Definir target real deflactado con INDICE_PRECIO_REAL (en millones de USD reales)
    temp["ventas_reales_definitivo_ipc_real"] = (
        temp["VENTAS_NOMINALES_DEFINITIVO_MES"] / (temp["INDICE_PRECIO_REAL"] / 100.0)
    ) / 1e6

    # 5. Unir rezagos estrictamente causales
    temp = (
        temp.merge(precios_lag1, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
        .merge(precios_lag13, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
        .merge(def_lag2, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
        .merge(def_lag12, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
        .merge(temp_lag1, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
        .merge(temp_lag12, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
        .merge(malla, on=["CODIGO_INDUSTRIA", "INDICE_MES"], how="left")
    )

    # 6. Eliminar explícitamente columnas con fuga futura y columnas redundantes de corte
    cols_eliminar = [
        c
        for c in COLUMNAS_FUGA
        + ["CORTE_KEY", "TIPO_CORTE", "ES_DEFINITIVO", "DIAS_APROX_DESDE_CIERRE", "ventas_temprano_real"]
        if c in temp.columns
    ]
    temp = temp.drop(columns=cols_eliminar)

    temp = temp.sort_values(["FECHA_MES", "CODIGO_INDUSTRIA"]).reset_index(drop=True)
    return temp


def dividir_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición temporal única sin fuga de futuro:
    - Train: 2016-01-01 a 2024-12-01 (tras 12 meses iniciales de 2015 para lag_12 completo).
    - Test (hold-out): 2025-01-01 a 2026-06-01 (últimos 18 meses).
    """
    fechas = pd.to_datetime(df["FECHA_MES"])
    train = df[(fechas >= FECHA_TRAIN_INICIO) & (fechas <= FECHA_TRAIN_FIN)].copy()
    test = df[(fechas >= FECHA_TEST_INICIO) & (fechas <= FECHA_TEST_FIN)].copy()
    train = train.sort_values(["FECHA_MES", "CODIGO_INDUSTRIA"]).reset_index(drop=True)
    test = test.sort_values(["FECHA_MES", "CODIGO_INDUSTRIA"]).reset_index(drop=True)
    return train, test
