#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive v1 — previsão de carga elétrica diária.

Adaptação do código Naive original para a mesma base Parquet e a mesma
estrutura experimental dos modelos de carga diária do projeto.

Método preservado
-----------------
A previsão é produzida exclusivamente pela regra Naive configurada:

    y_hat(t) = y(t - NAIVE_LAG)

Para dados diários, o padrão é NAIVE_LAG = 1, isto é, a carga observada
no dia anterior é usada como previsão do dia atual. O método não ajusta
Random Forest, LASSO, Ridge ou qualquer outro modelo de aprendizagem.

A janela e os pontos de validação são usados somente para alinhar o período
out-of-sample aos demais códigos do projeto. Eles não alteram a regra Naive.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURAÇÃO EDITÁVEL
# =============================================================================

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "results_naive_load_forecasting")

#DATASETS = {
#    "Carga_2005_2018": (
#        r"/Users/macbook/My Drive/MATHEUS FELLIPE_DRIVE/DOUTORADO ECO/5 TRIMESTRE/APRENDIZADO_MAQUINA/Trabalho Final/Dados/v2/dados_consolidados_nao_tratados.parquet"
#    ),
#}

DATASETS = {
    "Carga_2005_2018": (
        r"G:\Meu Drive\MATHEUS FELLIPE_DRIVE\DOUTORADO ECO"
        r"\5 TRIMESTRE\APRENDIZADO_MAQUINA\Trabalho Final"
        r"\Dados\v2\dados_consolidados_nao_tratados.parquet"
    ),
}

# Referência da base, registrada nos relatórios.
START_DATE = "2005-01-01"
END_DATE = "2018-12-30"
EXPECTED_OBSERVATIONS = 5513

TIMESTAMP_COLUMN = "timestamp"
DEPENDENT_VARIABLE = "carga_diaria"
HORIZON = 1

PARQUET_READ_KWARGS: dict[str, Any] = {}
DROPNA_REQUIRED_COLUMNS = True
SORT_BY_TIMESTAMP = True

# Parâmetros de alinhamento com os demais modelos.
VALIDATION_POINTS = 30
MAX_TEST_ITERATIONS = None  # None para executar todas as origens disponíveis.

WINDOW_SIZES = {
    #"1-month": 30,
    # "3-month": 90,
    "1-year": 365,
}

# Regra Naive. Para carga diária, o padrão é um dia anterior.
NAIVE_LAG = 7
FORECAST_RULE = "y_hat(t) = y(t - NAIVE_LAG)"

# Mantidos como opções de compatibilidade/registro da estrutura do projeto.
GENERATE_LAGS = True
LAG_SOURCE_COLUMNS = [DEPENDENT_VARIABLE]
LAG_PERIODS = [1, 3, 7, 14, 28]


# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================


def normalise_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def detect_timestamp_column(data: pd.DataFrame) -> str | None:
    """Localiza a coluna temporal configurada ou uma alternativa usual."""
    if TIMESTAMP_COLUMN in data.columns:
        return TIMESTAMP_COLUMN

    preferred = {
        "timestamp", "datetime", "date", "data", "data_hora", "date_time", "ds", "time"
    }
    for column in data.columns:
        if normalise_name(column) in preferred:
            return str(column)

    candidates: list[tuple[float, str]] = []
    for column in data.columns:
        if column == DEPENDENT_VARIABLE:
            continue
        converted = pd.to_datetime(data[column], errors="coerce")
        ratio = float(converted.notna().mean())
        if ratio >= 0.80:
            candidates.append((ratio, str(column)))
    return max(candidates)[1] if candidates else None


def validate_configuration() -> None:
    if not isinstance(NAIVE_LAG, int) or NAIVE_LAG <= 0:
        raise ValueError("NAIVE_LAG deve ser um inteiro positivo.")
    if HORIZON != 1:
        raise ValueError("Esta versão v1 implementa previsão de um passo à frente.")
    if VALIDATION_POINTS < 0:
        raise ValueError("VALIDATION_POINTS não pode ser negativo.")
    if MAX_TEST_ITERATIONS is not None and MAX_TEST_ITERATIONS <= 0:
        raise ValueError("MAX_TEST_ITERATIONS deve ser None ou um inteiro positivo.")
    if not WINDOW_SIZES or any(int(value) <= 0 for value in WINDOW_SIZES.values()):
        raise ValueError("WINDOW_SIZES deve conter janelas positivas.")


def load_dataset(path: str) -> tuple[pd.DataFrame, str | None]:
    """Lê, ordena e limpa a base de carga diária."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Base não encontrada: {path}")

    data = pd.read_parquet(path, **PARQUET_READ_KWARGS)
    if data.empty:
        raise ValueError("A base Parquet está vazia.")
    if DEPENDENT_VARIABLE not in data.columns:
        raise KeyError(f"Variável dependente ausente: {DEPENDENT_VARIABLE}")

    timestamp_column = detect_timestamp_column(data)
    if timestamp_column is not None:
        data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
        data = data.dropna(subset=[timestamp_column])
        if SORT_BY_TIMESTAMP:
            data = data.sort_values(timestamp_column)

    data[DEPENDENT_VARIABLE] = pd.to_numeric(
        data[DEPENDENT_VARIABLE], errors="coerce"
    )
    if DROPNA_REQUIRED_COLUMNS:
        data = data.dropna(subset=[DEPENDENT_VARIABLE])

    return data.reset_index(drop=True), timestamp_column


# =============================================================================
# MÉTODO NAIVE E MÉTRICAS
# =============================================================================


def generate_naive_forecasts(
    data: pd.DataFrame,
    timestamp_column: str | None,
    window_size: int,
    validation_points: int,
    max_test_iterations: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera previsões Naive nos mesmos pontos out-of-sample do projeto.

    A primeira previsão ocorre no índice:

        validation_points + window_size

    Assim, os pontos anteriores ficam reservados para o bloco de validação e
    para a janela de treinamento conceitual usada pelos demais modelos.
    A regra Naive, porém, usa somente o valor observado no lag configurado.
    """
    total_iterations = len(data) - window_size - validation_points + 1
    if total_iterations <= 0:
        return pd.DataFrame(), pd.DataFrame()

    n_iterations = (
        total_iterations
        if max_test_iterations is None
        else min(total_iterations, int(max_test_iterations))
    )
    first_test_index = validation_points + window_size

    records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []

    for iteration in range(n_iterations):
        test_index = first_test_index + iteration
        if test_index >= len(data):
            break
        lag_index = test_index - NAIVE_LAG
        if lag_index < 0:
            continue

        test_row = data.iloc[test_index]
        lagged_value = data.iloc[lag_index][DEPENDENT_VARIABLE]
        true_value = test_row[DEPENDENT_VARIABLE]
        if pd.isna(lagged_value) or pd.isna(true_value):
            continue

        record: dict[str, Any] = {}
        if timestamp_column is not None:
            record[timestamp_column] = test_row[timestamp_column]
        record.update({
            "true_value": float(true_value),
            "forecast": float(lagged_value),
            "naive_lag": NAIVE_LAG,
        })
        records.append(record)

        parameter_record: dict[str, Any] = {
            "iteration": iteration + 1,
            "test_index": test_index,
            "lag_index": lag_index,
            "naive_lag": NAIVE_LAG,
            "forecast_rule": FORECAST_RULE,
        }
        if timestamp_column is not None:
            parameter_record["timestamp"] = test_row[timestamp_column]
        parameter_records.append(parameter_record)

    return pd.DataFrame(records), pd.DataFrame(parameter_records)


def calculate_metrics(forecast_df: pd.DataFrame) -> dict[str, float]:
    """Calcula as métricas usadas nos demais códigos do projeto."""
    y_true = forecast_df["true_value"].to_numpy(dtype=float)
    y_pred = forecast_df["forecast"].to_numpy(dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    nonzero = ~np.isclose(y_true, 0.0)
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE_percent": mape,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "mean_error_forecast_minus_true": float(np.mean(y_pred - y_true)),
    }


# =============================================================================
# EXPORTAÇÃO E EXECUÇÃO
# =============================================================================


def save_results(
    output_file: str,
    forecast_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    configuration = {
        "model": "Naive",
        "dataset": summary["dataset"],
        "dependent_variable": DEPENDENT_VARIABLE,
        "forecast_rule": FORECAST_RULE,
        "naive_lag": NAIVE_LAG,
        "horizon": HORIZON,
        "window_name": summary["window_name"],
        "window_size": summary["window_size"],
        "validation_points": VALIDATION_POINTS,
        "max_test_iterations": MAX_TEST_ITERATIONS,
        "start_date_configured": START_DATE,
        "end_date_configured": END_DATE,
        "expected_observations": EXPECTED_OBSERVATIONS,
    }
    configuration_df = pd.DataFrame(
        list(configuration.items()), columns=["parameter", "value"]
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast_True_Values", index=False)
        parameter_df.to_excel(writer, sheet_name="Naive_Parameters", index=False)
        configuration_df.to_excel(writer, sheet_name="Configuration", index=False)
        pd.DataFrame(
            list(summary.items()), columns=["metric", "value"]
        ).to_excel(writer, sheet_name="results", index=False)


def run_dataset(dataset_name: str, dataset_path: str) -> list[dict[str, Any]]:
    data, timestamp_column = load_dataset(dataset_path)
    print(
        f"\nProcessando {dataset_name}: "
        f"{len(data)} observações após a preparação."
    )

    summaries: list[dict[str, Any]] = []
    for window_name, window_size in WINDOW_SIZES.items():
        start_time = time.perf_counter()
        forecast_df, parameter_df = generate_naive_forecasts(
            data=data,
            timestamp_column=timestamp_column,
            window_size=int(window_size),
            validation_points=VALIDATION_POINTS,
            max_test_iterations=MAX_TEST_ITERATIONS,
        )
        if forecast_df.empty:
            print(f"Sem observações suficientes para {dataset_name} | {window_name}.")
            continue

        metrics = calculate_metrics(forecast_df)
        elapsed = time.perf_counter() - start_time
        summary: dict[str, Any] = {
            "model": "Naive",
            "dataset": dataset_name,
            "dependent_variable": DEPENDENT_VARIABLE,
            "window_name": window_name,
            "window_size": int(window_size),
            "n_observations_after_cleaning": len(data),
            "n_forecasts": len(forecast_df),
            "naive_lag": NAIVE_LAG,
            "validation_points": VALIDATION_POINTS,
            "max_test_iterations": MAX_TEST_ITERATIONS,
            "oos_start": forecast_df[timestamp_column].min() if timestamp_column else None,
            "oos_end": forecast_df[timestamp_column].max() if timestamp_column else None,
            "elapsed_seconds_model_window": elapsed,
            "elapsed_minutes_model_window": elapsed / 60.0,
            **metrics,
        }
        output_file = os.path.join(
            OUTPUT_DIR,
            f"result_Naive_{dataset_name}_carga_diaria_{window_name}.xlsx",
        )
        save_results(output_file, forecast_df, parameter_df, summary)
        summaries.append(summary)
        print(
            f"{dataset_name} | {window_name} | "
            f"previsões: {len(forecast_df)} | MAE: {metrics['MAE']:.6f} | "
            f"RMSE: {metrics['RMSE']:.6f}"
        )

    return summaries


def main() -> None:
    validate_configuration()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_start = time.perf_counter()
    summary_results: list[dict[str, Any]] = []

    for dataset_name, dataset_path in DATASETS.items():
        summary_results.extend(run_dataset(dataset_name, dataset_path))

    if summary_results:
        summary_file = os.path.join(
            OUTPUT_DIR, "summary_naive_carga_diaria.xlsx"
        )
        pd.DataFrame(summary_results).to_excel(
            summary_file, sheet_name="Summary", index=False
        )
        print(f"Resumo global salvo: {summary_file}")

    elapsed = time.perf_counter() - process_start
    print(f"Tempo total: {elapsed:.2f} segundos ({elapsed / 60.0:.2f} minutos)")


if __name__ == "__main__":
    main()
