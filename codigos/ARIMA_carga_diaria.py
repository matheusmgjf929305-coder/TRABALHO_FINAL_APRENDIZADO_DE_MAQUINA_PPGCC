
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARIMA v1 configurável para previsão de carga elétrica diária."""
from __future__ import annotations
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO EDITÁVEL
# =============================================================================
MODELAGEM_DIR = Path(r"G:\Meu Drive\MATHEUS FELLIPE_DRIVE\DOUTORADO ECO\5 TRIMESTRE\APRENDIZADO_MAQUINA\Trabalho Final\Codigos\Meus_Codigos\modelagem")
RESULTADOS_DIR = MODELAGEM_DIR / "results_arima_load_forecasting"
DATASETS = {"Carga_2005_2018": Path(r"G:\Meu Drive\MATHEUS FELLIPE_DRIVE\DOUTORADO ECO\5 TRIMESTRE\APRENDIZADO_MAQUINA\Trabalho Final\Dados\v2\dados_consolidados_nao_tratados.parquet")}



#{
 #   "Carga_2005_2018": (
 #       r"/Users/macbook/My Drive/MATHEUS FELLIPE_DRIVE/DOUTORADO ECO/5 TRIMESTRE/APRENDIZADO_MAQUINA/Trabalho Final/Dados/v2/dados_consolidados_nao_tratados.parquet"
   # ),
#}




#
TIMESTAMP_COLUMN = "timestamp"
DEPENDENT_VARIABLE = "carga_diaria"
START_DATE = "2005-01-01"
END_DATE = "2018-12-30"
HORIZON = 1
VALIDATION_POINTS = 30
MAX_TEST_ITERATIONS = None  # None executa todas as previsões possíveis.
WINDOW_SIZES = {#"6-month": 180,
                "1-year": 365,}  # Acrescente, por exemplo, "3-month": 90.

# Modelos escolhidos para estimação.
# O SARIMA usa periodicidade semanal (m=7), conforme a sazonalidade identificada.
MODEL_SPECS = [
    #"model_type": "ARIMA", "order": (2, 1, 1), "seasonal_order": None},
    #{"model_type": "ARIMA", "order": (1, 1, 2), "seasonal_order": None},
    {"model_type": "SARIMA", "order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},
]
TREND = None
ENFORCE_STATIONARITY = True
ENFORCE_INVERTIBILITY = True
# O ajuste por espaço de estados é a opção mais compatível entre versões
# do statsmodels e permite controlar maxiter de forma consistente.
METHOD = "statespace"
MAX_ITER = 200000
DISP = 0
RUN_VALIDATION_DIAGNOSTIC = True
APPLY_STANDARDIZATION = False


def detect_timestamp_column(data):
    if TIMESTAMP_COLUMN in data.columns:
        return TIMESTAMP_COLUMN
    for column in data.columns:
        if str(column).lower() in {"timestamp", "datetime", "date", "data", "ds", "time"}:
            return column
    return None


def load_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f"Base não encontrada: {path}")
    data = pd.read_parquet(path)
    if DEPENDENT_VARIABLE not in data.columns:
        raise KeyError(f"Variável dependente ausente: {DEPENDENT_VARIABLE}")
    timestamp_column = detect_timestamp_column(data)
    if timestamp_column:
        data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
        data = data.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
        if START_DATE:
            data = data[data[timestamp_column] >= pd.Timestamp(START_DATE)]
        if END_DATE:
            data = data[data[timestamp_column] <= pd.Timestamp(END_DATE)]
    data[DEPENDENT_VARIABLE] = pd.to_numeric(data[DEPENDENT_VARIABLE], errors="coerce")
    data = data.dropna(subset=[DEPENDENT_VARIABLE]).reset_index(drop=True)
    return data, timestamp_column


def validate_order(model_type, raw_order):
    if len(raw_order) != 3 or any(int(v) < 0 for v in raw_order):
        raise ValueError(f"Ordem inválida: {raw_order}")
    p, d, q = map(int, raw_order)
    valid = {"AR": q == 0 and d == 0, "MA": p == 0 and d == 0, "ARMA": d == 0, "ARIMA": True}
    if model_type not in valid or not valid[model_type]:
        raise ValueError(f"A ordem {raw_order} não respeita a classe {model_type}.")
    return p, d, q


def fit_model(values, order, seasonal_order=None):
    """Ajusta o ARIMA com compatibilidade entre versões do statsmodels."""
    model_class = SARIMAX if seasonal_order is not None else ARIMA
    model_kwargs = {
        "order": order,
        "trend": TREND,
        "enforce_stationarity": ENFORCE_STATIONARITY,
        "enforce_invertibility": ENFORCE_INVERTIBILITY,
    }
    if seasonal_order is not None:
        model_kwargs["seasonal_order"] = seasonal_order
    model = model_class(values, **model_kwargs)

    # No ajuste statespace, maxiter e disp são encaminhados ao otimizador.
    # Para outros métodos, algumas versões não aceitam os mesmos argumentos;
    # por isso há uma segunda tentativa sem method_kwargs.
    try:
        if METHOD in (None, "statespace"):
            return model.fit(method_kwargs={"maxiter": MAX_ITER, "disp": DISP})
        return model.fit(method=METHOD, method_kwargs={"maxiter": MAX_ITER})
    except (TypeError, ValueError):
        if METHOD in (None, "statespace"):
            return model.fit()
        return model.fit(method=METHOD)


def forecast_one(values, order, seasonal_order=None):
    return float(np.asarray(fit_model(values, order, seasonal_order).forecast(steps=HORIZON)).reshape(-1)[0])


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    nz = ~np.isclose(y_true, 0)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": mse, "RMSE": float(np.sqrt(mse)), "MAPE_percent": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else np.nan, "R2": float(1 - ss_res / ss_tot) if ss_tot else np.nan, "mean_error": float(np.mean(y_pred - y_true))}


def validation_diagnostic(values, window_size, order, seasonal_order=None, context=""):
    """Executa a validação temporal e imprime o acompanhamento no console."""
    if not RUN_VALIDATION_DIAGNOSTIC or len(values) < window_size + VALIDATION_POINTS:
        print(f"      Validação temporal: não executada (dados insuficientes ou desativada).")
        return np.nan, np.nan

    history = list(values[:window_size])
    actual = values[window_size:window_size + VALIDATION_POINTS]
    predicted = []
    total_validation = len(actual)
    print(f"      Validação temporal iniciada: {total_validation} ponto(s).")
    for validation_iteration, observed in enumerate(actual, start=1):
        try:
            prediction = forecast_one(np.asarray(history), order, seasonal_order)
            predicted.append(prediction)
            history.append(float(observed))
            print(
                f"         Validação {validation_iteration:>3}/{total_validation} | "
                f"real={float(observed):,.4f} | previsão={prediction:,.4f}"
            )
        except Exception as exc:
            print(f"         Validação {validation_iteration:>3}/{total_validation} | FALHA: {exc}")
            raise
    val_mae = float(mean_absolute_error(actual[:len(predicted)], predicted))
    val_rmse = float(np.sqrt(mean_squared_error(actual[:len(predicted)], predicted)))
    print(f"      Validação concluída | MAE={val_mae:,.4f} | RMSE={val_rmse:,.4f}")
    return val_mae, val_rmse

def label(model_type, order, seasonal_order=None):
    p, d, q = order
    base = f"{model_type}_p{p}_d{d}_q{q}"
    if seasonal_order is not None:
        P, D, Q, m = seasonal_order
        return f"{base}_P{P}_D{D}_Q{Q}_m{m}"
    return base


def run_model(data, timestamp_column, dataset_name, model_type, order, seasonal_order, window_name, window_size):
    """Executa um modelo e mostra no console todo o ciclo de avaliação."""
    values = data[DEPENDENT_VARIABLE].to_numpy(float)
    total = len(values) - window_size - VALIDATION_POINTS
    if total <= 0:
        print(f"      Sem dados suficientes: total possível={total}.")
        return None

    n = total if MAX_TEST_ITERATIONS is None else min(total, int(MAX_TEST_ITERATIONS))
    model_label = label(model_type, order, seasonal_order)
    records, validation_rows, errors = [], [], []
    started = time.perf_counter()

    print(f"      Iterações planejadas: {n} de {total} possíveis.")
    print(f"      Janela de treinamento: {window_size} observações | horizonte: {HORIZON}")
    print(f"      Ajuste do modelo iniciado: {model_label}")

    for iteration in range(n):
        train_start = iteration + VALIDATION_POINTS
        train_end = train_start + window_size
        test_index = train_end
        progress = (iteration + 1) / n * 100
        print(f"\n      [{iteration + 1:>4}/{n}] ({progress:6.2f}%) | origem de previsão: {test_index}")
        print(f"      Período de treino: índices {train_start} até {train_end - 1}")
        try:
            v_mae, v_rmse = validation_diagnostic(
                values[iteration:train_end], window_size, order, seasonal_order, context=model_label
            )
            print("      Ajustando o modelo com a janela atual...")
            prediction = forecast_one(values[train_start:train_end], order, seasonal_order)
            true_value = float(values[test_index])
            error_value = prediction - true_value
            row = {
                "iteration": iteration + 1,
                "true_value": true_value,
                "forecast": prediction,
                "error": error_value,
                "absolute_error": abs(error_value),
            }
            if timestamp_column:
                row[timestamp_column] = data[timestamp_column].iloc[test_index]
            records.append(row)
            validation_rows.append({
                "iteration": iteration + 1,
                "validation_MAE": v_mae,
                "validation_RMSE": v_rmse,
            })
            print(
                f"      SUCESSO | real={true_value:,.4f} | previsão={prediction:,.4f} | "
                f"erro={error_value:,.4f} | tempo acumulado={time.perf_counter() - started:,.2f}s"
            )
        except Exception as exc:
            errors.append({"iteration": iteration + 1, "error": repr(exc)})
            print(f"      FALHA na iteração {iteration + 1}: {exc}")
            print("      A execução continuará nas próximas iterações.")

    elapsed = time.perf_counter() - started
    print(f"\n      Final do modelo {model_label} | previsões válidas={len(records)} | "
          f"falhas={len(errors)} | tempo={elapsed:,.2f}s")
    if not records:
        print(f"      Nenhuma previsão válida para {model_label}.")
        if errors:
            print(f"      Primeiro erro registrado: {errors[0]['error']}")
        return None

    forecast_df = pd.DataFrame(records)
    summary = {
        "dataset": dataset_name, "model_type": model_type, "model": model_label,
        "p": order[0], "d": order[1], "q": order[2],
        "seasonal_P": seasonal_order[0] if seasonal_order else None,
        "seasonal_D": seasonal_order[1] if seasonal_order else None,
        "seasonal_Q": seasonal_order[2] if seasonal_order else None,
        "seasonal_period": seasonal_order[3] if seasonal_order else None,
        "window": window_name,
        "window_size": window_size, "validation_points": VALIDATION_POINTS,
        "n_forecasts": len(records), "failed_iterations": len(errors),
        "elapsed_seconds": elapsed, **metrics(forecast_df.true_value, forecast_df.forecast),
    }
    result_path = RESULTADOS_DIR / f"resultado_ARIMA_{dataset_name}_{model_label}_{window_name}.xlsx"
    print(f"      Salvando planilha do modelo: {result_path.name}")
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast_True_Values", index=False)
        pd.DataFrame(validation_rows).to_excel(writer, sheet_name="Validation_Diagnostic", index=False)
        pd.DataFrame([{
            **summary, "trend": TREND,
            "enforce_stationarity": ENFORCE_STATIONARITY,
            "enforce_invertibility": ENFORCE_INVERTIBILITY,
            "seasonal_order": seasonal_order,
            "apply_standardization": APPLY_STANDARDIZATION,
            "method": METHOD, "max_iter": MAX_ITER,
        }]).to_excel(writer, sheet_name="Parameters", index=False)
        pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_excel(writer, sheet_name="results", index=False)
        pd.DataFrame(errors).to_excel(writer, sheet_name="Errors", index=False)
    print(f"      Planilha salva com sucesso: {result_path.name}")
    return summary

def main():
    execution_started = time.perf_counter()
    print("=" * 86)
    print("ARIMA — PREVISÃO DE CARGA ELÉTRICA DIÁRIA")
    print("Acompanhamento detalhado da execução")
    print("=" * 86)
    print(f"Início: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Variável dependente: {DEPENDENT_VARIABLE}")
    print(f"Período configurado: {START_DATE} até {END_DATE}")
    print(f"Horizonte: {HORIZON} | pontos de validação: {VALIDATION_POINTS}")
    print(f"Limite de iterações: {'todas' if MAX_TEST_ITERATIONS is None else MAX_TEST_ITERATIONS}")
    print(f"Janelas configuradas: {list(WINDOW_SIZES.items())}")
    print(f"Modelos configurados: {[spec["model_type"] for spec in MODEL_SPECS]}")
    print(f"Pasta de resultados: {RESULTADOS_DIR}")
    print("-" * 86)

    MODELAGEM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    dataset_count = 0
    model_count = 0

    for dataset_name, dataset_path in DATASETS.items():
        dataset_count += 1
        print(f"\n[BANCO {dataset_count}/{len(DATASETS)}] {dataset_name}")
        print(f"Local da base: {dataset_path}")
        try:
            data, timestamp_column = load_dataset(Path(dataset_path))
            print(f"Base carregada: {len(data)} observações válidas.")
            if timestamp_column:
                print(f"Coluna temporal identificada: {timestamp_column}")
            else:
                print("Coluna temporal não identificada; será usado o índice.")
        except Exception as exc:
            print(f"FALHA ao carregar a base {dataset_name}: {exc}")
            continue

        print(f"\n[MODELOS] {len(MODEL_SPECS)} especificação(ões) cadastrada(s)")
        for model_index, spec in enumerate(MODEL_SPECS, start=1):
            model_type = spec["model_type"]
            order = tuple(spec["order"])
            seasonal_order = (
                tuple(spec["seasonal_order"])
                if spec.get("seasonal_order") is not None else None
            )
            model_count += 1
            try:
                order = validate_order("ARIMA", order)
            except Exception as exc:
                print(f"  ORDEM IGNORADA {order}: {exc}")
                continue
            if seasonal_order is not None and len(seasonal_order) != 4:
                print(f"  ORDEM SAZONAL IGNORADA {seasonal_order}: deve conter (P,D,Q,m).")
                continue
            print(
                f"  [MODELO {model_index}] {model_type} | "
                f"ordem=(p,d,q)={order}"
                + (f" | ordem sazonal=(P,D,Q,m)={seasonal_order}" if seasonal_order else "")
            )
            for window_name, window_size in WINDOW_SIZES.items():
                print(f"    [JANELA] {window_name} ({window_size} observações)")
                result = run_model(
                    data, timestamp_column, dataset_name, model_type, order,
                    seasonal_order, window_name, int(window_size)
                )
                if result:
                    summaries.append(result)
                    print(
                        f"    Métricas: MAE={result['MAE']:,.4f} | "
                        f"RMSE={result['RMSE']:,.4f} | "
                        f"MAPE={result['MAPE_percent']:,.4f}% | R²={result['R2']:,.4f}"
                    )

    print("\n" + "=" * 86)
    if summaries:
        summary_path = RESULTADOS_DIR / "resumo_global_ARIMA_carga_diaria.xlsx"
        pd.DataFrame(summaries).to_excel(summary_path, index=False)
        print(f"RESUMO GLOBAL SALVO: {summary_path}")
        print(f"Modelos com resultados: {len(summaries)}")
    else:
        print("NENHUM RESULTADO FOI GERADO.")
        print("Verifique a base, as ordens configuradas e as mensagens de falha acima.")
    print(f"Tempo total da execução: {time.perf_counter() - execution_started:,.2f}s")
    print("Execução encerrada.")
    print("=" * 86)


if __name__ == "__main__":
    main()
