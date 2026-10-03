"""
Comitês de previsão de carga_diaria com LASSO, Ridge e Random Forest.

O código preserva a estrutura dos códigos originais:
- leitura de uma ou mais bases Parquet;
- seleção explícita de variáveis em nível e variáveis com lags;
- janela móvel de treinamento;
- validação temporal antes de cada previsão;
- seleção de alpha para LASSO/Ridge e min_samples_leaf para Random Forest;
- previsão de um passo à frente;
- métricas e exportação para Excel.

São geradas duas alternativas de comitê:
1. COMITE_MEDIA_SIMPLES: média das previsões finais de LASSO, Ridge e Random Forest.
2. COMITE_MEDIA_PONDERADA: média ponderada pelo inverso do MAE obtido na validação
   temporal daquela iteração. Os pesos usam somente informações anteriores ao teste.
"""

from __future__ import annotations

import inspect
import os
import re
import time
import unicodedata
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURAÇÃO EDITÁVEL
# =============================================================================

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "results_comites_carga_eletrica")
print(OUTPUT_DIR)

#DATASETS = {
 #   "Carga_2005_2018": (
  #      r"/Users/macbook/My Drive/MATHEUS FELLIPE_DRIVE/DOUTORADO ECO/5 TRIMESTRE/APRENDIZADO_MAQUINA/Trabalho Final/Dados/v2/dados_consolidados_nao_tratados.parquet"
   # ),
#}

# Alternativa para Windows:
DATASETS = {
    "Carga_2005_2018": (
        r"G:\Meu Drive\MATHEUS FELLIPE_DRIVE\DOUTORADO ECO"
        r"\5 TRIMESTRE\APRENDIZADO_MAQUINA\Trabalho Final"
        r"\Dados\v2\dados_consolidados_nao_tratados.parquet"
    ),
}

START_DATE = "2005-01-01"
END_DATE = "2018-12-30"
EXPECTED_OBSERVATIONS = 5513
TIMESTAMP_COLUMN = "timestamp"
DEPENDENT_VARIABLE = "carga_diaria"
HORIZON = 1

VALIDATION_POINTS = 30
MAX_TEST_ITERATIONS = None
WINDOW_SIZES = {"1-year": 365}

# Variáveis usadas diretamente no instante da previsão.
LEVEL_SOURCE_COLUMNS = [
    "ear_max_total",
    "intercambio_diario",
    "vazao_afluente_geral",
    "vazao_turbinada_geral",
    "volume_util_con_geral",
    "ger_termica_diaria",
    "ger_eolica_diaria",
]

# Variáveis para as quais serão criadas defasagens.
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

# LASSO e Ridge.
STANDARDIZE_LINEAR_PREDICTORS = True
N_ALPHAS = 25
ALPHAS = np.logspace(-3, 2, N_ALPHAS)
LINEAR_MODEL_KWARGS = {
    "fit_intercept": True,
    "max_iter": 10000,
    "tol": 0.01,
    "random_state": 42,
}

# Random Forest.
APPLY_STANDARDIZATION_RF = False
N_TREES = 100
RF_MAX_FEATURES = 0.5
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF_CANDIDATES = [1, 2, 4, 8]
RF_N_JOBS = -1
RANDOM_STATE = 42

# Paralelismo da seleção de hiperparâmetros.
CV_N_JOBS = 1
PARQUET_READ_KWARGS = {}
DROPNA_REQUIRED_COLUMNS = True
SORT_BY_TIMESTAMP = True

# Importância dos preditores para os três modelos base.
CALCULATE_PERMUTATION_IMPORTANCE = True
PERMUTATION_N_REPEATS = 1
PERMUTATION_RANDOM_STATE = 42
PERMUTATION_SCORING = "neg_mean_absolute_error"

GROUP_PATTERNS = {
    "lags_carga": ["_t-", "lag_", "_lag", "carga_lag", "demanda_lag"],
    "demanda_carga": ["demanda", "carga_diaria", "carga", "load"],
    "balanco_energetico_intercambio": ["ger_energia_total", "intercambio", "energia_total"],
    "armazenamento_reservatorios": ["ear", "reservatorio", "reservatório", "volume_util"],
    "hidrologia_afluencias": ["vazao_afluente", "vazão_afluente", "afluencia", "afluência", "afluente"],
    "vazoes_turbinadas": ["vazao_turbinada", "vazão_turbinada", "turbinada"],
    "geracao_hidraulica": ["ger_hidraulica", "geração_hidráulica", "hidraulica", "hydro"],
    "geracao_termica": ["ger_termica", "geracao_termica", "térmica", "termica", "thermal"],
    "geracao_eolica": ["ger_eolica", "geracao_eolica", "eólica", "eolica", "wind"],
    "outras_variaveis": [],
}


# =============================================================================
# PREPARAÇÃO DA BASE
# =============================================================================


def normalise_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")


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


def make_lagged_df(data: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    frames = []
    for period in periods:
        period = int(period)
        if period <= 0:
            raise ValueError(f"Cada lag deve ser positivo; recebido: {period}")
        shifted = data.shift(period).copy()
        shifted.columns = [f"{column}_t-{period}" for column in data.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=data.index)


def numeric_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
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


def load_dataset(path: str):
    data = pd.read_parquet(path, **PARQUET_READ_KWARGS)
    if data.empty:
        raise ValueError("A base Parquet está vazia.")
    if DEPENDENT_VARIABLE not in data.columns:
        raise KeyError(f"Variável dependente ausente: {DEPENDENT_VARIABLE}")

    timestamp_column = detect_timestamp_column(data)
    if timestamp_column is not None:
        data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
        if SORT_BY_TIMESTAMP:
            data = data.sort_values(timestamp_column)
    data = data.reset_index(drop=True)

    missing_level = [c for c in LEVEL_SOURCE_COLUMNS if c not in data.columns]
    if missing_level:
        raise KeyError(f"Colunas ausentes para uso em nível: {missing_level}")
    if DEPENDENT_VARIABLE in LEVEL_SOURCE_COLUMNS:
        raise ValueError("A variável dependente não pode estar em LEVEL_SOURCE_COLUMNS.")
    selected_level = numeric_columns(data, list(dict.fromkeys(LEVEL_SOURCE_COLUMNS)))
    if len(selected_level) != len(set(LEVEL_SOURCE_COLUMNS)):
        missing_numeric = [c for c in LEVEL_SOURCE_COLUMNS if c not in selected_level]
        raise TypeError(f"Variáveis em nível não numéricas ou inválidas: {missing_numeric}")

    selected_lag_sources = list(dict.fromkeys(LAG_SOURCE_COLUMNS))
    lag_columns = []
    if GENERATE_LAGS:
        missing_lag = [c for c in selected_lag_sources if c not in data.columns]
        if missing_lag:
            raise KeyError(f"Colunas ausentes para criação de lags: {missing_lag}")
        lagged = make_lagged_df(data[selected_lag_sources], LAG_PERIODS)
        data = pd.concat([data, lagged], axis=1)
        lag_columns = [
            f"{column}_t-{period}"
            for column in selected_lag_sources
            for period in LAG_PERIODS
        ]

    data[DEPENDENT_VARIABLE] = pd.to_numeric(data[DEPENDENT_VARIABLE], errors="coerce")
    predictors = list(dict.fromkeys(selected_level + [c for c in lag_columns if c in data.columns]))
    if not predictors:
        raise ValueError("Nenhum preditor foi selecionado.")
    if DROPNA_REQUIRED_COLUMNS:
        data = data.dropna(subset=predictors + [DEPENDENT_VARIABLE])
    return data.reset_index(drop=True), predictors, timestamp_column, selected_level, lag_columns


def make_coverage_tables(predictors: list[str]):
    assignments = {column: [] for column in predictors}
    for group, patterns in GROUP_PATTERNS.items():
        if group == "outras_variaveis":
            continue
        patterns = [normalise_name(p) for p in patterns]
        for column in predictors:
            name = normalise_name(column)
            if any(pattern in name for pattern in patterns):
                assignments[column].append(group)
    for column in predictors:
        if not assignments[column]:
            assignments[column] = ["outras_variaveis"]
    groups, indices = [], []
    for group in GROUP_PATTERNS:
        current = [i for i, column in enumerate(predictors) if group in assignments[column]]
        if current:
            groups.append(group)
            indices.append(current)
    catalog = pd.DataFrame({
        "predictor": predictors,
        "assigned_group": [" | ".join(assignments[c]) for c in predictors],
    })
    coverage = pd.DataFrame([
        {"group_name": group, "n_predictors": len(index)}
        for group, index in zip(groups, indices)
    ])
    return coverage, catalog, groups, indices


# =============================================================================
# MODELOS BASE
# =============================================================================


def make_linear_model(model_class: Callable, alpha: float):
    accepted = inspect.signature(model_class).parameters
    params = {k: v for k, v in LINEAR_MODEL_KWARGS.items() if k in accepted}
    params["alpha"] = float(alpha)
    return model_class(**params)


class LinearWindowModel:
    def __init__(self, model_class: Callable, alpha: float, predictors: list[str]):
        self.predictors = predictors
        self.model = make_linear_model(model_class, alpha)
        self.scaler = StandardScaler() if STANDARDIZE_LINEAR_PREDICTORS else None

    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.predictors].to_numpy(dtype=float)
        self.y_train = train_data[DEPENDENT_VARIABLE].to_numpy(dtype=float)
        self.X_train = self.scaler.fit_transform(X) if self.scaler is not None else X
        self.model.fit(self.X_train, self.y_train)
        return self

    def transform(self, data: pd.DataFrame):
        X = data[self.predictors].to_numpy(dtype=float)
        return self.scaler.transform(X) if self.scaler is not None else X

    def forecast(self, data: pd.DataFrame) -> float:
        return float(np.asarray(self.model.predict(self.transform(data))).reshape(-1)[0])


class RandomForestWindowModel:
    def __init__(self, min_samples_leaf: int, predictors: list[str]):
        self.predictors = predictors
        self.model = RandomForestRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=RF_MAX_FEATURES,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS,
        )
        self.scaler = StandardScaler() if APPLY_STANDARDIZATION_RF else None

    def fit(self, train_data: pd.DataFrame):
        X = train_data[self.predictors].to_numpy(dtype=float)
        self.y_train = train_data[DEPENDENT_VARIABLE].to_numpy(dtype=float)
        self.X_train = self.scaler.fit_transform(X) if self.scaler is not None else X
        self.model.fit(self.X_train, self.y_train)
        return self

    def transform(self, data: pd.DataFrame):
        X = data[self.predictors].to_numpy(dtype=float)
        return self.scaler.transform(X) if self.scaler is not None else X

    def forecast(self, data: pd.DataFrame) -> float:
        return float(np.asarray(self.model.predict(self.transform(data))).reshape(-1)[0])


def select_linear_alpha(cv_data, window_size, predictors, model_class):
    def evaluate(alpha):
        true, pred = [], []
        for start in range(len(cv_data) - window_size):
            train = cv_data.iloc[start:start + window_size]
            valid = cv_data.iloc[start + window_size:start + window_size + 1]
            fitted = LinearWindowModel(model_class, alpha, predictors).fit(train)
            true.append(float(valid[DEPENDENT_VARIABLE].iloc[0]))
            pred.append(fitted.forecast(valid))
        return mean_absolute_error(true, pred)
    scores = Parallel(n_jobs=CV_N_JOBS)(delayed(evaluate)(float(a)) for a in ALPHAS)
    return float(ALPHAS[int(np.argmin(scores))])


def select_rf_leaf(cv_data, window_size, predictors):
    def evaluate(candidate):
        true, pred = [], []
        for start in range(len(cv_data) - window_size):
            train = cv_data.iloc[start:start + window_size]
            valid = cv_data.iloc[start + window_size:start + window_size + 1]
            fitted = RandomForestWindowModel(candidate, predictors).fit(train)
            true.append(float(valid[DEPENDENT_VARIABLE].iloc[0]))
            pred.append(fitted.forecast(valid))
        return mean_absolute_error(true, pred)
    scores = Parallel(n_jobs=CV_N_JOBS)(
        delayed(evaluate)(int(candidate)) for candidate in RF_MIN_SAMPLES_LEAF_CANDIDATES
    )
    return int(RF_MIN_SAMPLES_LEAF_CANDIDATES[int(np.argmin(scores))])


def validation_forecast(model_name, cv_data, window_size, predictors, parameter):
    """Gera previsões walk-forward da validação sem usar o bloco de teste."""
    true, pred = [], []
    for start in range(len(cv_data) - window_size):
        train = cv_data.iloc[start:start + window_size]
        valid = cv_data.iloc[start + window_size:start + window_size + 1]
        if model_name in ("LASSO", "Ridge"):
            model_class = linear_model.Lasso if model_name == "LASSO" else linear_model.Ridge
            fitted = LinearWindowModel(model_class, parameter, predictors).fit(train)
        else:
            fitted = RandomForestWindowModel(parameter, predictors).fit(train)
        true.append(float(valid[DEPENDENT_VARIABLE].iloc[0]))
        pred.append(fitted.forecast(valid))
    return np.asarray(true), np.asarray(pred)


# =============================================================================
# COMITÊS E MÉTRICAS
# =============================================================================


def committee_weights(validation_errors: dict[str, float]) -> dict[str, float]:
    inverse = {name: 1.0 / max(float(error), 1e-12) for name, error in validation_errors.items()}
    total = sum(inverse.values())
    return {name: value / total for name, value in inverse.items()}


def calculate_metrics(y_true, y_pred):
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


def permutation_table(fitted, timestamp, model_name):
    if not CALCULATE_PERMUTATION_IMPORTANCE:
        return pd.DataFrame()
    result = permutation_importance(
        fitted.model,
        fitted.X_train,
        fitted.y_train,
        scoring=PERMUTATION_SCORING,
        n_repeats=PERMUTATION_N_REPEATS,
        random_state=PERMUTATION_RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame([result.importances_mean], columns=fitted.predictors)
    table.insert(0, "model", model_name)
    table.insert(1, "timestamp", timestamp)
    return table


def save_model_results(path, forecast_df, parameter_df, importance_df, coverage_df, catalog_df, summary):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast_True_Values", index=False)
        parameter_df.to_excel(writer, sheet_name="Selected_CV_Parameters", index=False)
        importance_df.to_excel(writer, sheet_name="Individual_PI", index=False)
        coverage_df.to_excel(writer, sheet_name="Group_Coverage_Check", index=False)
        catalog_df.to_excel(writer, sheet_name="Predictor_Group_Map", index=False)
        pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_excel(writer, sheet_name="results", index=False)


def save_committee_results(path, forecast_df, weights_df, summary):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast_True_Values", index=False)
        weights_df.to_excel(writer, sheet_name="Committee_Weights", index=False)
        pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_excel(writer, sheet_name="results", index=False)


# =============================================================================
# EXECUÇÃO
# =============================================================================


def run_dataset(dataset_name: str, dataset_path: str):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Base não encontrada: {dataset_path}")
    data, predictors, timestamp_column, level_columns, lag_columns = load_dataset(dataset_path)
    coverage_df, catalog_df, _, _ = make_coverage_tables(predictors)
    minimum_required = max(WINDOW_SIZES.values()) + VALIDATION_POINTS + 1
    if len(data) < minimum_required:
        raise ValueError(f"São necessárias pelo menos {minimum_required} observações; base possui {len(data)}.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for window_name, window_size in WINDOW_SIZES.items():
        total_iterations = len(data) - window_size - VALIDATION_POINTS + 1
        n_iterations = total_iterations if MAX_TEST_ITERATIONS is None else min(total_iterations, int(MAX_TEST_ITERATIONS))
        base_records = {name: [] for name in ("LASSO", "Ridge", "RandomForest")}
        base_parameters = {name: [] for name in base_records}
        base_importances = {name: [] for name in base_records}
        committee_records = {"COMITE_MEDIA_SIMPLES": [], "COMITE_MEDIA_PONDERADA": []}
        committee_weights_rows = []
        start_time = time.perf_counter()

        for iteration in range(n_iterations):
            cv_data = data.iloc[iteration:iteration + window_size + VALIDATION_POINTS]
            train_start = iteration + VALIDATION_POINTS
            train_data = data.iloc[train_start:train_start + window_size]
            test_data = data.iloc[train_start + window_size:train_start + window_size + 1]
            if test_data.empty:
                continue
            timestamp = test_data[timestamp_column].iloc[0] if timestamp_column else pd.NaT

            # Seleção e ajuste dos três modelos usando exatamente os mesmos dados.
            alpha_lasso = select_linear_alpha(cv_data, window_size, predictors, linear_model.Lasso)
            alpha_ridge = select_linear_alpha(cv_data, window_size, predictors, linear_model.Ridge)
            leaf_rf = select_rf_leaf(cv_data, window_size, predictors)
            fitted = {
                "LASSO": LinearWindowModel(linear_model.Lasso, alpha_lasso, predictors).fit(train_data),
                "Ridge": LinearWindowModel(linear_model.Ridge, alpha_ridge, predictors).fit(train_data),
                "RandomForest": RandomForestWindowModel(leaf_rf, predictors).fit(train_data),
            }
            predictions = {name: model.forecast(test_data) for name, model in fitted.items()}

            # Alternativa 1: média simples, sem privilegiar um modelo.
            simple_value = float(np.mean(list(predictions.values())))

            # Alternativa 2: pesos definidos somente pelo desempenho na validação.
            validation_errors = {}
            for name, parameter in {
                "LASSO": alpha_lasso,
                "Ridge": alpha_ridge,
                "RandomForest": leaf_rf,
            }.items():
                y_cv, p_cv = validation_forecast(name, cv_data, window_size, predictors, parameter)
                validation_errors[name] = float(mean_absolute_error(y_cv, p_cv))
            weights = committee_weights(validation_errors)
            weighted_value = float(sum(weights[name] * predictions[name] for name in predictions))

            for name, model in fitted.items():
                base_records[name].append({
                    "timestamp": timestamp,
                    "true_value": float(test_data[DEPENDENT_VARIABLE].iloc[0]),
                    "forecast": predictions[name],
                })
                base_parameters[name].append({
                    "timestamp": timestamp,
                    "selected_alpha": alpha_lasso if name == "LASSO" else alpha_ridge if name == "Ridge" else np.nan,
                    "selected_min_samples_leaf": leaf_rf if name == "RandomForest" else np.nan,
                })
                base_importances[name].append(permutation_table(model, timestamp, name))

            true_value = float(test_data[DEPENDENT_VARIABLE].iloc[0])
            for committee_name, value in {
                "COMITE_MEDIA_SIMPLES": simple_value,
                "COMITE_MEDIA_PONDERADA": weighted_value,
            }.items():
                committee_records[committee_name].append({
                    "timestamp": timestamp,
                    "true_value": true_value,
                    "forecast": value,
                    "forecast_lasso": predictions["LASSO"],
                    "forecast_ridge": predictions["Ridge"],
                    "forecast_random_forest": predictions["RandomForest"],
                })
            committee_weights_rows.append({
                "timestamp": timestamp,
                **{f"validation_mae_{name.lower()}": error for name, error in validation_errors.items()},
                **{f"weight_{name.lower()}": weight for name, weight in weights.items()},
            })
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(f"{dataset_name} | {window_name} | iteração {iteration + 1}/{n_iterations}")

        for model_name, records in base_records.items():
            if not records:
                continue
            forecast_df = pd.DataFrame(records)
            parameter_df = pd.DataFrame(base_parameters[model_name])
            importance_df = pd.concat(base_importances[model_name], ignore_index=True) if CALCULATE_PERMUTATION_IMPORTANCE else pd.DataFrame()
            metrics = calculate_metrics(forecast_df["true_value"], forecast_df["forecast"])
            summary = {
                "model": model_name,
                "dataset": dataset_name,
                "window_name": window_name,
                "window_size": window_size,
                "validation_points": VALIDATION_POINTS,
                "n_observations_after_cleaning": len(data),
                "n_forecasts": len(forecast_df),
                "n_predictors": len(predictors),
                "n_level_predictors": len(level_columns),
                "n_lagged_predictors": len(lag_columns),
                "standardize_linear_predictors": STANDARDIZE_LINEAR_PREDICTORS,
                "standardize_random_forest": APPLY_STANDARDIZATION_RF,
                "elapsed_seconds": time.perf_counter() - start_time,
                **metrics,
            }
            path = os.path.join(OUTPUT_DIR, f"result_{model_name}_{dataset_name}_{window_name}.xlsx")
            save_model_results(path, forecast_df, parameter_df, importance_df, coverage_df, catalog_df, summary)

        weights_df = pd.DataFrame(committee_weights_rows)
        for committee_name, records in committee_records.items():
            if not records:
                continue
            forecast_df = pd.DataFrame(records)
            summary = {
                "model": committee_name,
                "dataset": dataset_name,
                "window_name": window_name,
                "window_size": window_size,
                "validation_points": VALIDATION_POINTS,
                "n_observations_after_cleaning": len(data),
                "n_forecasts": len(forecast_df),
                "n_base_models": 3,
                "base_models": "LASSO | Ridge | RandomForest",
                "committee_definition": "média simples" if committee_name == "COMITE_MEDIA_SIMPLES" else "pesos normalizados pelo inverso do MAE da validação",
                **calculate_metrics(forecast_df["true_value"], forecast_df["forecast"]),
            }
            path = os.path.join(OUTPUT_DIR, f"result_{committee_name}_{dataset_name}_{window_name}.xlsx")
            save_committee_results(path, forecast_df, weights_df, summary)

        comparison_rows = []
        for name, records in base_records.items():
            if records:
                frame = pd.DataFrame(records)
                comparison_rows.append({"model": name, **calculate_metrics(frame["true_value"], frame["forecast"])})
        for name, records in committee_records.items():
            if records:
                frame = pd.DataFrame(records)
                comparison_rows.append({"model": name, **calculate_metrics(frame["true_value"], frame["forecast"])})
        pd.DataFrame(comparison_rows).to_excel(
            os.path.join(OUTPUT_DIR, f"comparativo_{dataset_name}_{window_name}.xlsx"),
            sheet_name="Comparativo",
            index=False,
        )


def main():
    process_start = time.perf_counter()
    for dataset_name, dataset_path in DATASETS.items():
        run_dataset(dataset_name, dataset_path)
    elapsed = time.perf_counter() - process_start
    print(f"Tempo total: {elapsed:.2f} segundos ({elapsed / 60:.2f} minutos)")


if __name__ == "__main__":
    main()
