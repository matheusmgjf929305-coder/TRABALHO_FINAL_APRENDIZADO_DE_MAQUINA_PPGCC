"""
Pipeline completo de previsão de carga_diaria com LASSO e Ridge,
com estimação conjunta de preditores contemporâneos e defasados (lags).

A versão v11 padroniza os preditores dentro de cada janela de treinamento,
ajustando o StandardScaler somente nos dados históricos usados no treino.

O arquivo foi preparado para a base Parquet de carga elétrica. Para reutilizar
com outra base, altere principalmente a seção CONFIGURAÇÃO EDITÁVEL.

Dependências:
    numpy, pandas, scikit-learn, joblib e um engine Parquet
    (preferencialmente pyarrow).
"""

from __future__ import annotations

import inspect
import os
import re
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import linear_model
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURAÇÃO EDITÁVEL
# =============================================================================

import os
import numpy as np
from sklearn import linear_model


BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "results_lasso_ridge_load_forecasting")


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



# Período informado para a base.
# Usado apenas como referência e registro nos resultados.
START_DATE = "2005-01-01"
END_DATE = "2018-12-30"
EXPECTED_OBSERVATIONS = 5513


# Coluna temporal.
# Se "timestamp" não existir, o código deve detectar automaticamente
# uma coluna temporal, como "data".
TIMESTAMP_COLUMN = "timestamp"

DEPENDENT_VARIABLE = "carga_diaria"


# -----------------------------------------------------------------------------
# SELEÇÃO EDITÁVEL DAS VARIÁVEIS PREDITORAS
# -----------------------------------------------------------------------------
#
# Variáveis usadas em nível: entram diretamente no modelo, sem defasagem.
# Deixe a lista vazia caso queira usar somente variáveis defasadas.
#
# IMPORTANTE: não inclua aqui a variável dependente nem a coluna temporal.
LEVEL_SOURCE_COLUMNS = [
    #"demanda_maxima_diaria",
    "ear_max_total",
    #"ger_energia_total",
    "intercambio_diario",
    "vazao_afluente_geral",
    "vazao_turbinada_geral",
    "volume_util_con_geral",
    #"ger_hidraulica_diaria",
    #"ger_hidraulica_seco",
    "ger_termica_diaria",
    "ger_eolica_diaria",
]


# Colunas selecionadas para a criação de lags.
# Uma mesma variável pode aparecer nas duas listas: nesse caso,
# ela entra em nível e também com os lags escolhidos em LAG_PERIODS.
LAG_SOURCE_COLUMNS = [
    DEPENDENT_VARIABLE,
    #"demanda_maxima_diaria",
    "ear_max_total",
    #"ger_energia_total",
    "intercambio_diario",
    "vazao_afluente_geral",
    "vazao_turbinada_geral",
    "volume_util_con_geral",
    #"ger_hidraulica_diaria",
    #"ger_hidraulica_seco",
    "ger_termica_diaria",
    "ger_eolica_diaria",
]


# Colunas que nunca devem entrar como preditoras.
CONTROL_ONLY_COLUMNS = [
    TIMESTAMP_COLUMN,
]

EXCLUDED_FROM_PREDICTORS = list(
    dict.fromkeys(
        CONTROL_ONLY_COLUMNS + [DEPENDENT_VARIABLE]
    )
)


# Colunas preservadas na planilha de previsões,
# quando existirem na base.
FORECAST_METADATA_COLUMNS = [
    TIMESTAMP_COLUMN,
    DEPENDENT_VARIABLE,
]


# Configuração de leitura do arquivo Parquet.
PARQUET_READ_KWARGS = {
    # Exemplo:
    # "engine": "pyarrow",
}


# Remover observações com valores ausentes nas colunas necessárias.
DROPNA_REQUIRED_COLUMNS = True


# Ordenar os dados pela coluna temporal.
SORT_BY_TIMESTAMP = True


# Previsão de um passo à frente.
HORIZON = 1


# Padronização dos preditores para LASSO/Ridge.
#
# Quando True, o StandardScaler é ajustado separadamente em cada janela de
# treinamento e aplicado à validação e ao ponto de teste. A variável dependente
# permanece na escala original, portanto as previsões continuam na unidade real.
STANDARDIZE_PREDICTORS = True


# Número de observações usadas na validação temporal
# para selecionar o melhor alpha.
#
# Reduzido para acelerar o teste.
VALIDATION_POINTS = 30 #30


# Limite temporário de iterações para teste.
#
# IMPORTANTE:
# essa variável somente terá efeito se o loop principal do código
# estiver configurado para utilizá-la.
MAX_TEST_ITERATIONS = None #None


# Janelas em número de observações.
#
# Para teste rápido, usa-se uma janela pequena.
# Como a base é diária, 30 observações correspondem aproximadamente
# a um mês.
WINDOW_SIZES = {
    #"1-month": 30,
    "6-month": 180,
    #"1-year": 365,
}


# =============================================================================
# LAGS: estimação conjunta com os demais preditores
# =============================================================================

# Mantém a criação de lags para a variável dependente e para
# ger_energia_total.
#
# Esses lags são estimados conjuntamente com as demais variáveis
# preditoras no mesmo modelo LASSO/Ridge.
GENERATE_LAGS = True


# LAG_SOURCE_COLUMNS foi definido na seção de seleção editável, antes desta seção.


# Para o teste rápido, cria somente a defasagem de um período.
LAG_PERIODS = [
    1,
    3,
    7,
    14,
    28
]


# =============================================================================
# HIPERPARÂMETROS DOS MODELOS
# =============================================================================

# Poucos valores de alpha para acelerar a seleção.
N_ALPHAS = 25

ALPHAS = np.logspace(
    -3,
    2,
    N_ALPHAS,
)


# Configuração dos modelos.
MODEL_KWARGS = {
    "fit_intercept": True,

    # Número reduzido apenas para o teste.
    "max_iter": EXPECTED_OBSERVATIONS - max(WINDOW_SIZES.values()) + VALIDATION_POINTS + 1,
        # len(data) - WINDOW_SIZES - VALIDATION_POINTS + 1 

    # Tolerância maior acelera a convergência.
    "tol": 0.01,

    "random_state": 42,
}


# Para teste rápido, utiliza somente Ridge.
MODELS = {
    "LASSO": linear_model.Lasso,
    "Ridge": linear_model.Ridge,
}


# Evita paralelismo excessivo durante o teste.
CV_N_JOBS = 1


# =============================================================================
# IMPORTÂNCIA DAS VARIÁVEIS
# =============================================================================

# Uma única repetição reduz bastante o tempo de processamento.
PERMUTATION_N_REPEATS = 1

PERMUTATION_SCORING = "neg_median_absolute_error"

PERMUTATION_RANDOM_STATE = 42

GROUPED_PERMUTATION_MODE = "rel"
# Opções:
# "raw" = importância em valor absoluto
# "rel"  = importância relativa


# =============================================================================
# AGRUPAMENTO DAS VARIÁVEIS
# =============================================================================

# Os padrões são procurados dentro do nome das colunas,
# sem diferenciar maiúsculas e minúsculas.
GROUP_PATTERNS = {
    # Defasagens temporais das variáveis.
    # As variáveis específicas em lag serão definidas posteriormente.
    "lags_carga": [
        "lag_",
        "l_",
        "carga_lag",
        "carga_diaria_lag",
        "demanda_lag",
        "t-",
    ],

    # Variáveis de previsão ou informação futura.
    # Devem ser utilizadas somente se estiverem disponíveis no momento
    # real da previsão, para evitar vazamento de informação.
    "leads_carga": [
        "lead_",
        "f_",
        "carga_lead",
        "carga_diaria_lead",
        "demanda_lead",
        "t+",
    ],

    # Demanda e carga do sistema elétrico.
    "demanda_carga": [
        "demanda_maxima_diaria",
        "demanda",
        "carga_diaria",
        "carga",
        "load",
        "peak_load",
        "maxima_diaria",
    ],

    # Geração total, balanço energético e intercâmbio entre regiões.
    "balanco_energetico_intercambio": [
        "ger_energia_total",
        "geracao_total",
        "geração_total",
        "intercambio_diario",
        "intercambio",
        "intercâmbio",
        "exchange",
        "energia_total",
    ],

    # Disponibilidade e armazenamento energético dos reservatórios.
    # EAR significa Energia Armazenada em Reservatórios.
    "armazenamento_reservatorios": [
        "ear_max_total",
        "ear_total",
        "ear",
        "energia_armazenada",
        "reservatorio",
        "reservatório",
        "volume_util",
        "volume_armazenado",
    ],

    # Afluências e vazões naturais que determinam a disponibilidade
    # de água para geração hidráulica.
    "hidrologia_afluencias": [
        "vazao_afluente_geral",
        "vazão_afluente_geral",
        "vazao_afluente",
        "vazão_afluente",
        "afluencia",
        "afluência",
        "afluente",
        "natural_inflow",
    ],

    # Vazão utilizada pelas usinas hidrelétricas para geração.
    "vazoes_turbinadas": [
        "vazao_turbinada_geral",
        "vazão_turbinada_geral",
        "vazao_turbinada",
        "vazão_turbinada",
        "turbinada",
        "turbined_flow",
    ],

    # Geração hidráulica, incluindo a geração total e a parcela
    # associada a condições hidrológicas mais secas.
    "geracao_hidraulica": [
        "ger_hidraulica_diaria",
        "ger_hidraulica_seco",
        "geracao_hidraulica",
        "geração_hidráulica",
        "hidraulica",
        "hidráulica",
        "hydro",
    ],

    # Geração proveniente de usinas termelétricas.
    "geracao_termica": [
        "ger_termica_diaria",
        "geracao_termica",
        "geração térmica",
        "termica",
        "térmica",
        "termica_diaria",
        "térmica_diária",
        "thermal",
    ],

    # Geração eólica.
    "geracao_eolica": [
        "ger_eolica_diaria",
        "geracao_eolica",
        "geração eólica",
        "eolica",
        "eólica",
        "eolica_diaria",
        "eólica_diária",
        "wind_generation",
    ],

    # Variáveis temporais, caso existam na base.
    "estrutura_temporal": [
        "hora",
        "hour",
        "dia",
        "day",
        "semana",
        "week",
        "mes",
        "mês",
        "month",
        "ano",
        "year",
        "feriado",
        "feriado",
        "holiday",
        "sin_",
        "cos_",
    ],

    # Grupo de segurança para variáveis que não se enquadrarem
    # nos agrupamentos específicos.
    "outras_variaveis": [],
}

# =============================================================================
# FUNÇÕES DE SUPORTE
# =============================================================================


def _normalise_name(value: object) -> str:
    """Normaliza um nome para comparação sem acentos e sem pontuação."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def make_lagged_df(df_to_lag: pd.DataFrame, lags_list: list[int]) -> pd.DataFrame:
    """Cria colunas defasadas, mantendo todas as variáveis em conjunto.

    A função segue a lógica do código original: cada variável selecionada é
    deslocada para cada lag e recebe o sufixo ``_t-<lag>``. A remoção de linhas
    incompletas ocorre depois que todos os lags foram criados.
    """
    if not lags_list:
        return pd.DataFrame(index=df_to_lag.index)
    frames = []
    for lag in lags_list:
        lag = int(lag)
        if lag <= 0:
            raise ValueError(f"Cada lag deve ser positivo; recebido: {lag}")
        shifted = df_to_lag.shift(lag).copy()
        shifted.columns = [f"{column}_t-{lag}" for column in df_to_lag.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def add_configured_lags(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona somente os lags das variáveis escolhidas."""
    if not GENERATE_LAGS or not LAG_SOURCE_COLUMNS:
        return data

    missing = [
        column for column in LAG_SOURCE_COLUMNS
        if column not in data.columns
    ]
    if missing:
        raise KeyError(
            "Colunas indicadas para criação de lags não existem na base: "
            + ", ".join(map(str, missing))
        )

    lagged = make_lagged_df(data[LAG_SOURCE_COLUMNS], LAG_PERIODS)
    new_columns = [
        column for column in lagged.columns
        if column not in data.columns
    ]
    return pd.concat([data, lagged[new_columns]], axis=1)


def detect_timestamp_column(data: pd.DataFrame) -> str | None:
    """Localiza a coluna temporal ou retorna None quando não houver uma."""
    if TIMESTAMP_COLUMN in data.columns:
        return TIMESTAMP_COLUMN

    preferred = {
        "timestamp", "datetime", "date", "data", "data_hora", "datahora",
        "date_time", "ds", "time", "tempo",
    }
    normalised = {
        column: _normalise_name(column) for column in data.columns
    }

    for column, name in normalised.items():
        if name in preferred:
            return column

    candidates = []
    for column in data.columns:
        if column == DEPENDENT_VARIABLE:
            continue
        converted = pd.to_datetime(data[column], errors="coerce")
        valid_ratio = converted.notna().mean()
        if valid_ratio >= 0.80:
            candidates.append((valid_ratio, column))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def make_model(model_class: Callable, alpha: float):
    """Cria o modelo, ignorando parâmetros não aceitos pela classe."""
    accepted = inspect.signature(model_class).parameters
    params = {
        key: value
        for key, value in MODEL_KWARGS.items()
        if key in accepted
    }
    params["alpha"] = float(alpha)
    return model_class(**params)


def load_dataset(dataset_path: str) -> tuple[pd.DataFrame, list[str], str | None]:
    """Lê Parquet, identifica a data, limpa a base e seleciona preditores."""
    data = pd.read_parquet(dataset_path, **PARQUET_READ_KWARGS)

    # Cria os lags somente após ler a base e antes de selecionar os preditores.
    data = add_configured_lags(data)

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("A base Parquet está vazia ou não retornou um DataFrame.")
    if DEPENDENT_VARIABLE not in data.columns:
        raise KeyError(
            f"A variável dependente '{DEPENDENT_VARIABLE}' não foi encontrada. "
            f"Colunas disponíveis: {list(data.columns)}"
        )

    timestamp_column = detect_timestamp_column(data)
    if timestamp_column is not None:
        data[timestamp_column] = pd.to_datetime(
            data[timestamp_column], errors="coerce"
        )
        if SORT_BY_TIMESTAMP:
            data = data.sort_values(timestamp_column)

    # A variável dependente deve ser numérica para o scikit-learn.
    data[DEPENDENT_VARIABLE] = pd.to_numeric(
        data[DEPENDENT_VARIABLE], errors="coerce"
    )

    # A matriz de preditores é formada exclusivamente pelas duas listas
    # configuradas pelo usuário: nível e lags. Nenhuma outra coluna da base
    # entra automaticamente no modelo.
    configured_level = list(dict.fromkeys(LEVEL_SOURCE_COLUMNS))
    configured_lags = []
    if GENERATE_LAGS:
        configured_lags = [
            f"{column}_t-{int(lag)}"
            for column in LAG_SOURCE_COLUMNS
            for lag in LAG_PERIODS
        ]

    candidate_columns = list(dict.fromkeys(configured_level + configured_lags))
    forbidden = {DEPENDENT_VARIABLE, TIMESTAMP_COLUMN}
    if timestamp_column is not None:
        forbidden.add(timestamp_column)
    invalid_configured = [
        column for column in candidate_columns
        if column in forbidden
    ]
    if invalid_configured:
        raise ValueError(
            "A variável dependente e a coluna temporal não podem ser usadas "
            "como preditoras: " + ", ".join(invalid_configured)
        )

    missing_configured = [
        column for column in candidate_columns
        if column not in data.columns
    ]
    if missing_configured:
        raise KeyError(
            "Variáveis selecionadas não encontradas na base após a preparação "
            "dos lags: " + ", ".join(missing_configured)
        )

    # Mantém apenas variáveis numéricas. Colunas textuais não podem ser usadas
    # diretamente pelos modelos LASSO/Ridge e são informadas ao usuário.
    predictors = []
    ignored_non_numeric = []
    for column in candidate_columns:
        if pd.api.types.is_bool_dtype(data[column]):
            data[column] = data[column].astype(float)
            predictors.append(column)
        elif pd.api.types.is_numeric_dtype(data[column]):
            predictors.append(column)
        else:
            converted = pd.to_numeric(data[column], errors="coerce")
            if converted.notna().mean() >= 0.95:
                data[column] = converted
                predictors.append(column)
            else:
                ignored_non_numeric.append(column)

    if ignored_non_numeric:
        print(
            "Colunas não numéricas ignoradas como preditoras: "
            + ", ".join(map(str, ignored_non_numeric))
        )
    if not predictors:
        raise ValueError(
            "Nenhuma variável preditora numérica foi encontrada após a limpeza."
        )

    required = predictors + [DEPENDENT_VARIABLE]
    if DROPNA_REQUIRED_COLUMNS:
        data = data.dropna(subset=required)

    data = data.reset_index(drop=True)
    return data, predictors, timestamp_column


def build_group_indices(
    predictor_columns: list[str],
    group_patterns: dict[str, list[str]],
) -> tuple[list[str], list[list[int]], list[str]]:
    """Cria grupos de colunas e lista as colunas sem grupo."""
    groups: list[str] = []
    indices: list[list[int]] = []
    matched: set[int] = set()

    for group_name, patterns in group_patterns.items():
        if group_name == "outras_variaveis":
            continue
        normalised_patterns = [pattern.lower() for pattern in patterns]
        group_indices = [
            index
            for index, column in enumerate(predictor_columns)
            if any(pattern in column.lower() for pattern in normalised_patterns)
        ]
        if group_indices:
            groups.append(group_name)
            indices.append(group_indices)
            matched.update(group_indices)

    unmatched = [
        column for index, column in enumerate(predictor_columns)
        if index not in matched
    ]
    if unmatched and "outras_variaveis" in group_patterns:
        groups.append("outras_variaveis")
        indices.append([
            index for index, column in enumerate(predictor_columns)
            if column in unmatched
        ])
        unmatched = []

    return groups, indices, unmatched


def make_coverage_tables(predictor_columns: list[str]):
    groups, indices, unmatched = build_group_indices(
        predictor_columns, GROUP_PATTERNS
    )
    group_rows = []
    lookup = {}
    for group_name, group_indices in zip(groups, indices):
        columns = [predictor_columns[index] for index in group_indices]
        group_rows.append({
            "group_name": group_name,
            "n_predictors": len(columns),
            "predictors": ", ".join(columns),
        })
        for column in columns:
            lookup[column] = group_name

    if unmatched:
        group_rows.append({
            "group_name": "UNMATCHED",
            "n_predictors": len(unmatched),
            "predictors": ", ".join(unmatched),
        })

    catalog = pd.DataFrame({
        "predictor": predictor_columns,
        "assigned_group": [lookup.get(column, "UNMATCHED") for column in predictor_columns],
    })
    return pd.DataFrame(group_rows), catalog, groups, indices


def safe_forecast_value(model, X_forecast: np.ndarray) -> float:
    prediction = np.asarray(model.predict(X_forecast)).reshape(-1)
    if prediction.size == 0:
        raise ValueError("O modelo não retornou uma previsão.")
    return float(prediction[0])


# =============================================================================
# MODELO E IMPORTÂNCIA DAS VARIÁVEIS
# =============================================================================


class MySklearningModel:
    def __init__(
        self,
        model,
        train_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        predictors: list[str],
        dependent_variable: str,
        groups: list[str],
        group_indices: list[list[int]],
    ):
        self.model = model
        self.predictors = predictors
        self.dependent_variable = dependent_variable
        self.groups = groups
        self.group_indices = group_indices
        X_train = train_data[predictors].to_numpy(dtype=float)
        self.y = train_data[dependent_variable].to_numpy(dtype=float)
        X_forecast = forecast_data[predictors].to_numpy(dtype=float)

        # O scaler é ajustado exclusivamente no conjunto de treinamento desta
        # janela. Assim, validação e teste não influenciam média ou desvio-padrão.
        self.scaler = StandardScaler() if STANDARDIZE_PREDICTORS else None
        if self.scaler is not None:
            self.X = self.scaler.fit_transform(X_train)
            self.X_forecast = self.scaler.transform(X_forecast)
        else:
            self.X = X_train
            self.X_forecast = X_forecast

    def fit(self):
        self.model.fit(self.X, self.y)
        return self

    def forecast(self) -> float:
        return safe_forecast_value(self.model, self.X_forecast)

    def individual_permutation_importance(self) -> pd.DataFrame:
        result = permutation_importance(
            self.model,
            self.X,
            self.y,
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
        baseline = float(scorer(self.model, self.X, self.y))
        rng = np.random.RandomState(PERMUTATION_RANDOM_STATE)
        values = {}

        for group_name, group_indices in zip(self.groups, self.group_indices):
            drops = []
            for _ in range(PERMUTATION_N_REPEATS):
                X_permuted = self.X.copy()
                order = rng.permutation(len(self.X))
                X_permuted[:, group_indices] = self.X[order][:, group_indices]
                permuted_score = float(scorer(self.model, X_permuted, self.y))
                drops.append(baseline - permuted_score)

            importance = float(np.mean(drops))
            if GROUPED_PERMUTATION_MODE == "rel":
                denominator = max(abs(baseline), np.finfo(float).eps)
                importance /= denominator
            values[group_name] = importance

        return pd.DataFrame([values])


def select_alpha(
    cv_data: pd.DataFrame,
    window_size: int,
    predictors: list[str],
    model_class: Callable,
    groups: list[str],
    group_indices: list[list[int]],
) -> float:
    """Seleciona alpha por validação temporal crescente dentro da janela."""
    if len(cv_data) <= window_size:
        return float(ALPHAS[0])

    def evaluate(alpha: float) -> float:
        y_true = []
        y_pred = []
        for start in range(len(cv_data) - window_size):
            train = cv_data.iloc[start:start + window_size]
            valid = cv_data.iloc[start + window_size:start + window_size + 1]
            model = MySklearningModel(
                make_model(model_class, alpha), train, valid, predictors,
                DEPENDENT_VARIABLE, groups, group_indices,
            ).fit()
            y_true.append(float(valid[DEPENDENT_VARIABLE].iloc[0]))
            y_pred.append(model.forecast())
        return mean_absolute_error(y_true, y_pred)

    scores = Parallel(n_jobs=CV_N_JOBS)(
        delayed(evaluate)(float(alpha)) for alpha in ALPHAS
    )
    return float(ALPHAS[int(np.argmin(scores))])


def make_forecast_record(
    test_data: pd.DataFrame,
    timestamp_column: str | None,
    forecast_value: float,
) -> dict:
    """Cria uma linha de resultado para uma previsão.

    O nome da função é mantido explicitamente para evitar o NameError que
    ocorria quando o laço principal tentava registrar cada previsão.
    """
    record = {}
    for column in FORECAST_METADATA_COLUMNS:
        source_column = timestamp_column if column == TIMESTAMP_COLUMN else column
        if source_column is not None and source_column in test_data.columns:
            record[column] = test_data[source_column].iloc[0]
    record["true_value"] = float(test_data[DEPENDENT_VARIABLE].iloc[0])
    record["forecast"] = float(forecast_value)
    return record


# Alias semântico para manter compatibilidade com versões anteriores.
forecast_record = make_forecast_record


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, MSE, RMSE, MAPE e R² com tratamento de casos-limite."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = ~np.isclose(y_true, 0.0)
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
    else:
        mape = float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE_percent": mape,
        "R2": r2,
    }


def save_results(
    output_file: str,
    forecast_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    individual_pi_df: pd.DataFrame,
    grouped_pi_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    summary: dict,
) -> None:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast_True_Values", index=False)
        coefficients_df.to_excel(writer, sheet_name="Coefficients", index=False)
        alpha_df.to_excel(writer, sheet_name="Selected_Alphas", index=False)
        individual_pi_df.to_excel(writer, sheet_name="Individual_PI", index=False)
        grouped_pi_df.to_excel(writer, sheet_name="Grouped_PI", index=False)
        coverage_df.to_excel(writer, sheet_name="Group_Coverage_Check", index=False)
        catalog_df.to_excel(writer, sheet_name="Predictor_Group_Map", index=False)
        pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_excel(
            writer, sheet_name="results", index=False
        )


# =============================================================================
# EXECUÇÃO
# =============================================================================


def run_dataset(dataset_name: str, dataset_path: str) -> None:
    print("\n" + "=" * 80)
    print(f"Processando: {dataset_name}")
    print(f"Path: {dataset_path}")
    print("=" * 80)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Base não encontrada: {dataset_path}")

    data, predictors, timestamp_column = load_dataset(dataset_path)
    print(f"Dimensão após limpeza: {data.shape}")
    print(f"Observações esperadas/informadas: {EXPECTED_OBSERVATIONS}")
    print(f"Número de preditores numéricos: {len(predictors)}")
    print(f"Coluna temporal utilizada: {timestamp_column or 'índice das observações'}")

    coverage_df, catalog_df, groups, group_indices = make_coverage_tables(predictors)
    unmatched = coverage_df.loc[
        coverage_df["group_name"] == "UNMATCHED", "n_predictors"
    ] if not coverage_df.empty else pd.Series(dtype=int)
    if not unmatched.empty:
        print(f"Preditores sem grupo específico: {int(unmatched.iloc[0])}")

    minimum_required = max(WINDOW_SIZES.values()) + VALIDATION_POINTS + 1
    if len(data) < minimum_required:
        raise ValueError(
            f"A base possui {len(data)} observações, mas são necessárias pelo menos "
            f"{minimum_required} para as janelas configuradas."
        )

    for window_name, window_size in WINDOW_SIZES.items():
        if window_size <= 0:
            raise ValueError(f"Janela inválida: {window_name}={window_size}")

        total_iterations = len(data) - window_size - VALIDATION_POINTS + 1
        if total_iterations <= 0:
            print(f"Janela ignorada por falta de observações: {window_name}")
            continue

        for model_name, model_class in MODELS.items():
            print(f"\nModelo={model_name} | Janela={window_name}")
            records = []
            alpha_rows = []
            coefficient_rows = []
            individual_rows = []
            grouped_rows = []

            n_iterations = (
                total_iterations
                if MAX_TEST_ITERATIONS is None
                else min(total_iterations, int(MAX_TEST_ITERATIONS))
            )

            for iteration in range(n_iterations):
                cv_start = iteration
                cv_end = iteration + window_size + VALIDATION_POINTS
                cv_data = data.iloc[cv_start:cv_end]
                train_start = iteration + VALIDATION_POINTS
                train_end = train_start + window_size
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[train_end:train_end + 1]
                if test_data.empty:
                    continue

                best_alpha = select_alpha(
                    cv_data, window_size, predictors, model_class,
                    groups, group_indices,
                )
                fitted = MySklearningModel(
                    make_model(model_class, best_alpha), train_data, test_data,
                    predictors, DEPENDENT_VARIABLE, groups, group_indices,
                ).fit()

                value = fitted.forecast()
                records.append(
                    make_forecast_record(test_data, timestamp_column, value)
                )
                alpha_record = {
                    "timestamp": (
                        test_data[timestamp_column].iloc[0]
                        if timestamp_column is not None
                        else pd.NaT
                    ),
                    "selected_alpha": best_alpha,
                }
                alpha_rows.append(alpha_record)
                coefficient_rows.append(fitted.model.coef_)

                individual = fitted.individual_permutation_importance()
                grouped = fitted.grouped_permutation_importance()
                if timestamp_column is not None:
                    timestamp = test_data[timestamp_column].iloc[0]
                    individual.insert(0, timestamp_column, timestamp)
                    grouped.insert(0, timestamp_column, timestamp)
                individual_rows.append(individual)
                grouped_rows.append(grouped)

                if (iteration + 1) % 10 == 0 or iteration == 0:
                    print(f"Iteração {iteration + 1}/{n_iterations}")

            if not records:
                print("Nenhuma previsão foi gerada.")
                continue

            forecast_df = pd.DataFrame(records)
            alpha_df = pd.DataFrame(alpha_rows, columns=["timestamp", "selected_alpha"])

            coefficients_df = pd.DataFrame(coefficient_rows, columns=predictors)
            if timestamp_column is not None and timestamp_column in forecast_df:
                coefficients_df.insert(0, timestamp_column, forecast_df[timestamp_column])

            individual_pi_df = pd.concat(individual_rows, ignore_index=True)
            grouped_pi_df = pd.concat(grouped_rows, ignore_index=True)

            metrics = calculate_metrics(
                forecast_df["true_value"].to_numpy(),
                forecast_df["forecast"].to_numpy(),
            )
            summary = {
                "model": model_name,
                "dataset": dataset_name,
                "dependent_variable": DEPENDENT_VARIABLE,
                "start_date_configured": START_DATE,
                "end_date_configured": END_DATE,
                "horizon": HORIZON,
                "window_name": window_name,
                "window_size": window_size,
                "n_observations_after_cleaning": len(data),
                "n_forecasts": len(forecast_df),
                "n_predictors": len(predictors),
                "standardize_predictors": STANDARDIZE_PREDICTORS,
                "n_lagged_predictors": sum(
                    1 for column in predictors
                    if re.search(r"(?:_t-|_lag|lag_|^l_)", str(column), re.IGNORECASE)
                ),
                **metrics,
            }

            output_file = os.path.join(
                OUTPUT_DIR,
                f"result_{model_name}_{dataset_name}_carga_diaria_{window_name}.xlsx",
            )
            save_results(
                output_file, forecast_df, coefficients_df, alpha_df,
                individual_pi_df, grouped_pi_df, coverage_df, catalog_df, summary,
            )
            print(f"Resultado salvo em: {output_file}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset_name, dataset_path in DATASETS.items():
        run_dataset(dataset_name, dataset_path)


if __name__ == "__main__":
    main()
