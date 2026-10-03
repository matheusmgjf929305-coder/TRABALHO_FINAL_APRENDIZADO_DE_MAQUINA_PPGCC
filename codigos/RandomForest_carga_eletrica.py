#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest — previsão de carga elétrica diária.

Versão com seleção independente de variáveis em nível e variáveis que recebem
lags. A seleção em nível e a seleção de lags não incluem automaticamente todas
as colunas numéricas da base.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURAÇÃO EDITÁVEL
# =============================================================================

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "results_random_forest_load_forecasting_v2")

DATASETS = {
    "Carga_2005_2018": (
        r"/Users/macbook/My Drive/MATHEUS FELLIPE_DRIVE/DOUTORADO ECO/5 TRIMESTRE/APRENDIZADO_MAQUINA/Trabalho Final/Dados/v2/dados_consolidados_nao_tratados.parquet"
    ),
}

#DATASETS = {
 #   "Carga_2005_2018": (
  #      r"G:\Meu Drive\MATHEUS FELLIPE_DRIVE\DOUTORADO ECO"
   #     r"\5 TRIMESTRE\APRENDIZADO_MAQUINA\Trabalho Final"
    #    r"\Dados\v2\dados_consolidados_nao_tratados.parquet"
    #),
#}

START_DATE = "2005-01-01"
END_DATE = "2018-12-30"
EXPECTED_OBSERVATIONS = 5513

TIMESTAMP_COLUMN = "timestamp"
DEPENDENT_VARIABLE = "carga_diaria"
HORIZON = 1
APPLY_STANDARDIZATION = False
PARQUET_READ_KWARGS = {}
DROPNA_REQUIRED_COLUMNS = True
SORT_BY_TIMESTAMP = True

VALIDATION_POINTS = 30
MAX_TEST_ITERATIONS = None

WINDOW_SIZES = {
    # "1-month": 30,
    "6-month": 180,
    #"1-year": 365,
}

# -----------------------------------------------------------------------------
# Seleção independente de variáveis
# -----------------------------------------------------------------------------
# Variáveis usadas diretamente em nível no modelo.
# Não inclua a variável dependente nesta lista.
LEVEL_SOURCE_COLUMNS = [
     "ear_max_total",
     "intercambio_diario",
     "vazao_afluente_geral",
     "vazao_turbinada_geral",
     "volume_util_con_geral",
     "ger_termica_diaria",
     "ger_eolica_diaria",
]

# Variáveis das quais serão criadas as defasagens.
GENERATE_LAGS = True
LAG_SOURCE_COLUMNS = [
    DEPENDENT_VARIABLE,
    "ear_max_total",
    "intercambio_diario",
    "vazao_afluente_geral",
    "vazao_turbinada_geral",
    "volume_util_con_geral",
    "ger_termica_diaria",
    "ger_eolica_diaria",
]
LAG_PERIODS = [1, 3, 7, 14, 28]

N_TREES = 100
RF_MAX_FEATURES = 0.5
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF_CANDIDATES = [1, 2, 4, 8]
RF_N_JOBS = -1
RANDOM_STATE = 42
CV_N_JOBS = 1

PERMUTATION_N_REPEATS = 1
PERMUTATION_SCORING = "neg_median_absolute_error"
PERMUTATION_RANDOM_STATE = 42
GROUPED_PERMUTATION_MODE = "rel"

EXCLUDED_FROM_PREDICTORS = [TIMESTAMP_COLUMN, DEPENDENT_VARIABLE]
FORECAST_METADATA_COLUMNS = [TIMESTAMP_COLUMN, DEPENDENT_VARIABLE]

GROUP_PATTERNS = {
    "lags_carga": ["_t-", "lag_", "_lag", "carga_lag", "demanda_lag"],
    "leads_carga": ["lead_", "_t+", "carga_lead", "demanda_lead"],
    "demanda_carga": [
        "demanda_maxima_diaria", "demanda", "carga_diaria", "carga", "load",
        "peak_load", "maxima_diaria",
    ],
    "balanco_energetico_intercambio": [
        "ger_energia_total", "geracao_total", "geração_total", "intercambio",
        "intercâmbio", "exchange", "energia_total",
    ],
    "armazenamento_reservatorios": [
        "ear_max_total", "ear_total", "ear", "energia_armazenada", "reservatorio",
        "reservatório", "volume_util", "volume_armazenado",
    ],
    "hidrologia_afluencias": [
        "vazao_afluente_geral", "vazão_afluente_geral", "vazao_afluente",
        "vazão_afluente", "afluencia", "afluência", "afluente", "natural_inflow",
    ],
    "vazoes_turbinadas": [
        "vazao_turbinada_geral", "vazão_turbinada_geral", "vazao_turbinada",
        "vazão_turbinada", "turbinada", "turbined_flow",
    ],
    "geracao_hidraulica": [
        "ger_hidraulica_diaria", "ger_hidraulica_seco", "geracao_hidraulica",
        "geração_hidráulica", "hidraulica", "hidráulica", "hydro",
    ],
    "geracao_termica": [
        "ger_termica_diaria", "geracao_termica", "geração térmica", "termica",
        "térmica", "termica_diaria", "térmica_diária", "thermal",
    ],
    "geracao_eolica": [
        "ger_eolica_diaria", "geracao_eolica", "geração eólica", "eolica",
        "eólica", "eolica_diaria", "eólica_diária", "wind_generation",
    ],
    "estrutura_temporal": [
        "hora", "hour", "dia", "day", "semana", "week", "mes", "mês", "month",
        "ano", "year", "feriado", "holiday", "sin_", "cos_",
    ],
    "outras_variaveis": [],
}


def normalise_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def make_lagged_df(data: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    if not periods:
        return pd.DataFrame(index=data.index)
    frames = []
    for period in periods:
        period = int(period)
        if period <= 0:
            raise ValueError(f"Cada lag deve ser positivo; recebido: {period}")
        shifted = data.shift(period).copy()
        shifted.columns = [f"{column}_t-{period}" for column in data.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def detect_timestamp_column(data: pd.DataFrame) -> str | None:
    if TIMESTAMP_COLUMN in data.columns:
        return TIMESTAMP_COLUMN
    preferred = {"timestamp", "datetime", "date", "data", "data_hora", "date_time", "ds", "time"}
    for column in data.columns:
        if normalise_name(column) in preferred:
            return column
    candidates = []
    for column in data.columns:
        if column == DEPENDENT_VARIABLE:
            continue
        converted = pd.to_datetime(data[column], errors="coerce")
        ratio = float(converted.notna().mean())
        if ratio >= 0.80:
            candidates.append((ratio, column))
    return max(candidates)[1] if candidates else None


def _numeric_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    selected = []
    for column in columns:
        if column not in data.columns:
            continue
        if pd.api.types.is_bool_dtype(data[column]):
            data[column] = data[column].astype(float)
            selected.append(column)
        elif pd.api.types.is_numeric_dtype(data[column]):
            selected.append(column)
        else:
            converted = pd.to_numeric(data[column], errors="coerce")
            if converted.notna().mean() >= 0.95:
                data[column] = converted
                selected.append(column)
    return selected


def load_dataset(path: str) -> tuple[pd.DataFrame, list[str], str | None, list[str], list[str]]:
    data = pd.read_parquet(path, **PARQUET_READ_KWARGS)
    if data.empty:
        raise ValueError("A base Parquet está vazia.")
    if DEPENDENT_VARIABLE not in data.columns:
        raise KeyError(f"Variável dependente ausente: {DEPENDENT_VARIABLE}")

    timestamp_column = detect_timestamp_column(data)
    if timestamp_column is not None:
        data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
        if SORT_BY_TIMESTAMP:
            data = data.sort_values(timestamp_column).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    missing_level = [c for c in LEVEL_SOURCE_COLUMNS if c not in data.columns]
    if missing_level:
        raise KeyError(f"Colunas ausentes para uso em nível: {missing_level}")
    if DEPENDENT_VARIABLE in LEVEL_SOURCE_COLUMNS:
        raise ValueError(
            f"{DEPENDENT_VARIABLE!r} é a variável-alvo e não pode estar em LEVEL_SOURCE_COLUMNS. "
            "Use-a somente em LAG_SOURCE_COLUMNS se desejar seus lags."
        )

    selected_level_columns = _numeric_columns(data, LEVEL_SOURCE_COLUMNS)
    invalid_level = [c for c in LEVEL_SOURCE_COLUMNS if c not in selected_level_columns]
    if invalid_level:
        raise TypeError(f"Variáveis em nível não numéricas ou inválidas: {invalid_level}")

    selected_lag_source_columns = list(dict.fromkeys(LAG_SOURCE_COLUMNS))
    if GENERATE_LAGS:
        missing_lag = [c for c in selected_lag_source_columns if c not in data.columns]
        if missing_lag:
            raise KeyError(f"Colunas ausentes para criação de lags: {missing_lag}")
        lagged = make_lagged_df(data[selected_lag_source_columns], LAG_PERIODS)
        data = pd.concat([data, lagged], axis=1)

    data[DEPENDENT_VARIABLE] = pd.to_numeric(data[DEPENDENT_VARIABLE], errors="coerce")
    lag_columns = [
        f"{column}_t-{period}"
        for column in selected_lag_source_columns
        for period in LAG_PERIODS
    ] if GENERATE_LAGS else []
    lag_columns = [column for column in lag_columns if column in data.columns]

    predictors = selected_level_columns + lag_columns
    predictors = list(dict.fromkeys(column for column in predictors if column != DEPENDENT_VARIABLE))
    if not predictors:
        raise ValueError(
            "Nenhum preditor foi selecionado. Preencha LEVEL_SOURCE_COLUMNS e/ou "
            "LAG_SOURCE_COLUMNS."
        )

    if DROPNA_REQUIRED_COLUMNS:
        data = data.dropna(subset=predictors + [DEPENDENT_VARIABLE])
    data = data.reset_index(drop=True)
    return data, predictors, timestamp_column, selected_level_columns, lag_columns


def make_coverage_tables(predictors: list[str]):
    assignments = {column: [] for column in predictors}
    for group, patterns in GROUP_PATTERNS.items():
        if group == "outras_variaveis":
            continue
        norm_patterns = [normalise_name(pattern) for pattern in patterns]
        for column in predictors:
            norm_column = normalise_name(column)
            if any(pattern in norm_column for pattern in norm_patterns):
                assignments[column].append(group)
    for column in predictors:
        if not assignments[column]:
            assignments[column] = ["outras_variaveis"]
    group_names = []
    group_indices = []
    for group in GROUP_PATTERNS:
        indices = [i for i, column in enumerate(predictors) if group in assignments[column]]
        if indices:
            group_names.append(group)
            group_indices.append(indices)
    catalog = pd.DataFrame({
        "predictor": predictors,
        "assigned_group": [" | ".join(assignments[column]) for column in predictors],
    })
    coverage = [{"group_name": group, "n_predictors": len(indices)} for group, indices in zip(group_names, group_indices)]
    coverage.extend([
        {"group_name": "TOTAL_PREDICTORS", "n_predictors": len(predictors)},
        {"group_name": "MULTIPLE_MATCH_PREDICTORS", "n_predictors": sum(len(v) > 1 for v in assignments.values())},
        {"group_name": "UNMATCHED_PREDICTORS", "n_predictors": sum(v == ["outras_variaveis"] for v in assignments.values())},
    ])
    return pd.DataFrame(coverage), catalog, group_names, group_indices


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    nonzero = ~np.isclose(y_true, 0.0)
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE_percent": mape,
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


class RandomForestWindowModel:
    def __init__(self, min_samples_leaf: int, predictors: list[str], groups: list[str], group_indices: list[list[int]]):
        self.predictors = predictors
        self.groups = groups
        self.group_indices = group_indices
        self.model = RandomForestRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=RF_MAX_FEATURES,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS,
        )
        self.scaler = StandardScaler() if APPLY_STANDARDIZATION else None

    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.predictors].to_numpy(dtype=float)
        y = train_data[DEPENDENT_VARIABLE].to_numpy(dtype=float)
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        X = data[self.predictors].to_numpy(dtype=float)
        return self.scaler.transform(X) if self.scaler is not None else X

    def forecast(self, test_data: pd.DataFrame) -> float:
        prediction = np.asarray(self.model.predict(self.transform(test_data))).reshape(-1)
        if prediction.size == 0:
            raise ValueError("O Random Forest não retornou previsão.")
        return float(prediction[0])

    def individual_permutation_importance(self) -> pd.DataFrame:
        result = permutation_importance(
            self.model, self.X_train, self.y_train,
            scoring=PERMUTATION_SCORING,
            n_repeats=PERMUTATION_N_REPEATS,
            random_state=PERMUTATION_RANDOM_STATE,
            n_jobs=1,
        )
        return pd.DataFrame([result.importances_mean], columns=self.predictors)

    def grouped_permutation_importance(self) -> pd.DataFrame:
        if not self.groups:
            return pd.DataFrame(index=[0])
        scorer = get_scorer(PERMUTATION_SCORING)
        baseline = float(scorer(self.model, self.X_train, self.y_train))
        rng = np.random.RandomState(PERMUTATION_RANDOM_STATE)
        values = {}
        for group, indices in zip(self.groups, self.group_indices):
            drops = []
            for _ in range(PERMUTATION_N_REPEATS):
                permuted = self.X_train.copy()
                order = rng.permutation(len(permuted))
                permuted[:, indices] = self.X_train[order][:, indices]
                score = float(scorer(self.model, permuted, self.y_train))
                drops.append(baseline - score)
            value = float(np.mean(drops))
            if GROUPED_PERMUTATION_MODE == "rel":
                value /= max(abs(baseline), np.finfo(float).eps)
            values[group] = value
        return pd.DataFrame([values])


def select_min_samples_leaf(cv_data: pd.DataFrame, window_size: int, predictors: list[str], groups: list[str], group_indices: list[list[int]]) -> int:
    def evaluate(candidate: int) -> float:
        true_values, predictions = [], []
        for start in range(len(cv_data) - window_size):
            train = cv_data.iloc[start:start + window_size]
            valid = cv_data.iloc[start + window_size:start + window_size + 1]
            fitted = RandomForestWindowModel(candidate, predictors, groups, group_indices).fit(train)
            true_values.append(float(valid[DEPENDENT_VARIABLE].iloc[0]))
            predictions.append(fitted.forecast(valid))
        return mean_absolute_error(true_values, predictions)
    scores = Parallel(n_jobs=CV_N_JOBS)(delayed(evaluate)(int(candidate)) for candidate in RF_MIN_SAMPLES_LEAF_CANDIDATES)
    return int(RF_MIN_SAMPLES_LEAF_CANDIDATES[int(np.argmin(scores))])


def make_forecast_record(test_data: pd.DataFrame, timestamp_column: str | None, value: float) -> dict:
    record = {}
    if timestamp_column is not None:
        record[timestamp_column] = test_data[timestamp_column].iloc[0]
    record["true_value"] = float(test_data[DEPENDENT_VARIABLE].iloc[0])
    record["forecast"] = float(value)
    return record


def save_results(path: str, forecast_df: pd.DataFrame, parameter_df: pd.DataFrame, importance_df: pd.DataFrame, grouped_df: pd.DataFrame, coverage_df: pd.DataFrame, catalog_df: pd.DataFrame, summary: dict):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast_True_Values", index=False)
        parameter_df.to_excel(writer, sheet_name="Selected_CV_Parameters", index=False)
        importance_df.to_excel(writer, sheet_name="Individual_PI", index=False)
        grouped_df.to_excel(writer, sheet_name="Grouped_PI", index=False)
        coverage_df.to_excel(writer, sheet_name="Group_Coverage_Check", index=False)
        catalog_df.to_excel(writer, sheet_name="Predictor_Group_Map", index=False)
        pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_excel(writer, sheet_name="results", index=False)


def run_dataset(dataset_name: str, dataset_path: str) -> None:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Base não encontrada: {dataset_path}")
    data, predictors, timestamp_column, selected_level_columns, lag_columns = load_dataset(dataset_path)
    selected_lag_source_columns = list(dict.fromkeys(LAG_SOURCE_COLUMNS)) if GENERATE_LAGS else []
    print(f"\nProcessando {dataset_name}: {data.shape[0]} observações e {len(predictors)} preditores.")
    print(f"Variáveis em nível: {len(selected_level_columns)} | Variáveis com lags: {len(lag_columns)}")

    coverage_df, catalog_df, groups, group_indices = make_coverage_tables(predictors)
    minimum_required = max(WINDOW_SIZES.values()) + VALIDATION_POINTS + 1
    if len(data) < minimum_required:
        raise ValueError(f"São necessárias pelo menos {minimum_required} observações; base possui {len(data)}.")

    for window_name, window_size in WINDOW_SIZES.items():
        total_iterations = len(data) - window_size - VALIDATION_POINTS + 1
        if total_iterations <= 0:
            continue
        n_iterations = total_iterations if MAX_TEST_ITERATIONS is None else min(total_iterations, int(MAX_TEST_ITERATIONS))
        start_time = time.perf_counter()
        records, parameter_rows, importance_rows, grouped_rows = [], [], [], []

        for iteration in range(n_iterations):
            cv_data = data.iloc[iteration:iteration + window_size + VALIDATION_POINTS]
            train_start = iteration + VALIDATION_POINTS
            train_data = data.iloc[train_start:train_start + window_size]
            test_data = data.iloc[train_start + window_size:train_start + window_size + 1]
            if test_data.empty:
                continue
            best_leaf = select_min_samples_leaf(cv_data, window_size, predictors, groups, group_indices)
            fitted = RandomForestWindowModel(best_leaf, predictors, groups, group_indices).fit(train_data)
            forecast = fitted.forecast(test_data)
            records.append(make_forecast_record(test_data, timestamp_column, forecast))
            parameter_rows.append({
                "timestamp": test_data[timestamp_column].iloc[0] if timestamp_column else pd.NaT,
                "selected_min_samples_leaf": best_leaf,
            })
            individual = fitted.individual_permutation_importance()
            grouped = fitted.grouped_permutation_importance()
            if timestamp_column:
                individual.insert(0, timestamp_column, test_data[timestamp_column].iloc[0])
                grouped.insert(0, timestamp_column, test_data[timestamp_column].iloc[0])
            importance_rows.append(individual)
            grouped_rows.append(grouped)
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(f"{dataset_name} | {window_name} | iteração {iteration + 1}/{n_iterations}")

        if not records:
            continue
        forecast_df = pd.DataFrame(records)
        parameter_df = pd.DataFrame(parameter_rows)
        importance_df = pd.concat(importance_rows, ignore_index=True)
        grouped_df = pd.concat(grouped_rows, ignore_index=True)
        metrics = calculate_metrics(forecast_df["true_value"], forecast_df["forecast"])
        elapsed = time.perf_counter() - start_time
        summary = {
            "model": "RandomForest",
            "dataset": dataset_name,
            "dependent_variable": DEPENDENT_VARIABLE,
            "start_date_configured": START_DATE,
            "end_date_configured": END_DATE,
            "window_name": window_name,
            "window_size": window_size,
            "n_observations_after_cleaning": len(data),
            "n_forecasts": len(forecast_df),
            "n_predictors": len(predictors),
            "n_level_predictors": len(selected_level_columns),
            "n_lagged_predictors": len(lag_columns),
            "level_source_columns": " | ".join(selected_level_columns),
            "lag_source_columns": " | ".join(selected_lag_source_columns),
            "lag_periods": " | ".join(map(str, LAG_PERIODS)) if GENERATE_LAGS else "",
            "n_trees": N_TREES,
            "max_features": RF_MAX_FEATURES,
            "standardization": APPLY_STANDARDIZATION,
            "validation_points": VALIDATION_POINTS,
            "max_test_iterations": MAX_TEST_ITERATIONS,
            "elapsed_seconds_model_window": elapsed,
            "elapsed_minutes_model_window": elapsed / 60.0,
            **metrics,
        }
        output_file = os.path.join(OUTPUT_DIR, f"result_RF_{dataset_name}_carga_diaria_{window_name}.xlsx")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_results(output_file, forecast_df, parameter_df, importance_df, grouped_df, coverage_df, catalog_df, summary)
        print(f"Resultado salvo: {output_file}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_start = time.perf_counter()
    for dataset_name, path in DATASETS.items():
        run_dataset(dataset_name, path)
    elapsed = time.perf_counter() - process_start
    print(f"Tempo total: {elapsed:.2f} segundos ({elapsed / 60.0:.2f} minutos)")


if __name__ == "__main__":
    main()
