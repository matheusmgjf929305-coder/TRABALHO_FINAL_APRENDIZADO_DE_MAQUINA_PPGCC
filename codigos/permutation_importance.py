#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera boxplots de permutation importance a partir da planilha:
result_RF_Carga_2005_2018_carga_diaria_1-year.xlsx

Abas utilizadas:
    - Individual_PI
    - Grouped_PI

A coluna "data" é ignorada. Os gráficos são salvos em:
    PI_boxplots_all_models/Carga_2005_2018_carga_diaria/
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Considera que a planilha Excel está na mesma pasta deste código.
BASE_DIR = Path(__file__).resolve().parent

EXCEL_FILE = (
    BASE_DIR
    / "result_RF_Carga_2005_2018_carga_diaria_1-year.xlsx"
)

OUTPUT_DIR = (
    BASE_DIR
    / "PI_boxplots_all_models"
    / "Carga_2005_2018_carga_diaria"
)

TOP_N = 10
DPI = 300
SHOW_FLIERS = False

LINE_COLOR = "black"
LINE_WIDTH = 0.85
MEDIAN_LINE_WIDTH = 1.10

FONT_SIZE = 8
TITLE_SIZE = 9
LABEL_SIZE = 8


# =============================================================================
# LEITURA E PREPARAÇÃO DOS DADOS
# =============================================================================

def read_pi_sheet(excel_file: Path, sheet_name: str) -> pd.DataFrame:
    """Lê uma aba de permutation importance e retorna apenas as colunas numéricas."""
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # A base anexada usa "data"; também aceita "timestamp" caso o nome mude.
    columns_to_drop = [
        col for col in df.columns
        if str(col).strip().lower() in {"data", "timestamp"}
    ]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis=1, how="all")

    if df.empty:
        raise ValueError(
            f"A aba '{sheet_name}' não contém colunas numéricas válidas."
        )

    return df


def get_top_features_by_median(
    df: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """Seleciona as variáveis com maiores medianas de importância."""
    medians = df.median(axis=0, skipna=True).sort_values(ascending=False)
    return medians.head(top_n).index.tolist()


# =============================================================================
# FORMATAÇÃO DOS RÓTULOS
# =============================================================================

INDIVIDUAL_LABELS = {
    "ear_max_total": "EAR máxima total",
    "intercambio_diario": "Intercâmbio diário",
    "vazao_afluente_geral": "Vazão afluente geral",
    "vazao_turbinada_geral": "Vazão turbinada geral",
    "volume_util_con_geral": "Volume útil dos reservatórios",
    "ger_termica_diaria": "Geração térmica diária",
    "ger_eolica_diaria": "Geração eólica diária",
    "carga_diaria": "Carga diária",
}

GROUP_LABELS = {
    "lags_carga": "Defasagens da carga",
    "leads_carga": "Leads da carga",
    "demanda_carga": "Demanda da carga",
    "balanco_energetico_intercambio": "Balanço energético e intercâmbio",
    "armazenamento_reservatorios": "Armazenamento dos reservatórios",
    "hidrologia_afluencias": "Hidrologia e afluências",
    "vazoes_turbinadas": "Vazões turbinadas",
    "geracao_termica": "Geração térmica",
    "geracao_eolica": "Geração eólica",
    "estrutura_temporal": "Estrutura temporal",
}


def clean_individual_label(name: str) -> str:
    """Converte nomes de variáveis individuais em rótulos mais legíveis."""
    label = str(name)

    # Identifica sufixos como _t-1, _t-3, _t-7 etc.
    lag_suffix = ""
    match = re.search(r"_t-(\d+)$", label)

    if match:
        lag_suffix = f" (t−{match.group(1)})"
        label = label[:match.start()]

    label = INDIVIDUAL_LABELS.get(label, label.replace("_", " "))
    return label + lag_suffix


def clean_group_label(name: str) -> str:
    """Converte os nomes dos grupos em rótulos mais legíveis."""
    label = str(name)
    return GROUP_LABELS.get(label, label.replace("_", " ").capitalize())


# =============================================================================
# GERAÇÃO DOS GRÁFICOS
# =============================================================================

def make_horizontal_boxplot(
    df: pd.DataFrame,
    features: list[str],
    display_labels: list[str],
    title: str,
    output_file: Path,
) -> None:
    """Gera e salva um boxplot horizontal."""
    if not features:
        print(f"Gráfico ignorado: nenhuma variável válida — {title}")
        return

    plot_df = df[features].dropna(axis=0, how="all")

    if plot_df.empty:
        print(f"Gráfico ignorado: sem dados — {title}")
        return

    data = [
        plot_df[feature].dropna().to_numpy()
        for feature in features
    ]

    valid_items = [
        (label, values)
        for label, values in zip(display_labels, data)
        if len(values) > 0
    ]

    if not valid_items:
        print(f"Gráfico ignorado: séries sem valores — {title}")
        return

    labels, data = zip(*valid_items)

    fig_height = max(2.5, 0.30 * len(labels) + 0.8)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))

    ax.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        showfliers=SHOW_FLIERS,
        widths=0.55,
        boxprops={
            "facecolor": "none",
            "edgecolor": LINE_COLOR,
            "linewidth": LINE_WIDTH,
        },
        whiskerprops={
            "color": LINE_COLOR,
            "linewidth": LINE_WIDTH,
        },
        capprops={
            "color": LINE_COLOR,
            "linewidth": LINE_WIDTH,
        },
        medianprops={
            "color": LINE_COLOR,
            "linewidth": MEDIAN_LINE_WIDTH,
        },
    )

    ax.invert_yaxis()
    ax.set_title(title, fontsize=TITLE_SIZE, pad=7)
    ax.set_xlabel("Aumento do erro de previsão", fontsize=LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    print(f"Planilha: {EXCEL_FILE}")
    print(f"Pasta de saída: {OUTPUT_DIR}")

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            "A planilha não foi encontrada. "
            "Confirme se ela está na mesma pasta do código e se o nome está correto."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    title_prefix = (
        "RF | Carga 2005–2018 | carga diária | 1-year"
    )

    # -------------------------------------------------------------------------
    # Individual_PI: plota as 10 variáveis com maiores medianas
    # -------------------------------------------------------------------------
    try:
        individual_df = read_pi_sheet(EXCEL_FILE, "Individual_PI")
        top_features = get_top_features_by_median(individual_df, TOP_N)
        top_labels = [
            clean_individual_label(feature)
            for feature in top_features
        ]

        individual_output = (
            OUTPUT_DIR
            / f"RF_Carga_Individual_PI_boxplot_top{TOP_N}.png"
        )

        make_horizontal_boxplot(
            df=individual_df,
            features=top_features,
            display_labels=top_labels,
            title=f"{title_prefix} | Individual PI",
            output_file=individual_output,
        )

        print(f"Gráfico individual salvo: {individual_output.name}")

    except Exception as exc:
        print(f"Erro ao processar a aba Individual_PI: {exc}")

    # -------------------------------------------------------------------------
    # Grouped_PI: plota todos os grupos, ordenados pela mediana
    # -------------------------------------------------------------------------
    try:
        grouped_df = read_pi_sheet(EXCEL_FILE, "Grouped_PI")

        grouped_order = (
            grouped_df.median(axis=0, skipna=True)
            .sort_values(ascending=False)
            .index.tolist()
        )

        grouped_labels = [
            clean_group_label(group)
            for group in grouped_order
        ]

        grouped_output = (
            OUTPUT_DIR
            / "RF_Carga_Grouped_PI_boxplot.png"
        )

        make_horizontal_boxplot(
            df=grouped_df,
            features=grouped_order,
            display_labels=grouped_labels,
            title=f"{title_prefix} | Grouped PI",
            output_file=grouped_output,
        )

        print(f"Gráfico agrupado salvo: {grouped_output.name}")

    except Exception as exc:
        print(f"Erro ao processar a aba Grouped_PI: {exc}")

    print(f"\nConcluído. Os gráficos estão em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()