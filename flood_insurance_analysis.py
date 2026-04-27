"""
Flood Insurance Claims Analysis for Indonesia (IDN) and Malaysia (MYS)
MASA Hackathon 2026: R-Ignite
Project: Climate Risk Assessment for a Multinational Reinsurance Firm

This script:
1. Loads the EM-DAT Excel file and final_dataset.csv
2. Filters flood events for IDN/MYS
3. Aggregates insured damage by country and year
4. Merges with the WDI climate indicators dataset
5. Plots insured damage over time per country
6. Prints a correlation matrix between insured damage and WDI indicators
7. Saves the merged dataset for further modelling
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMDAT_FILE = "public_emdat_custom_request_2026-04-27_cbcefcf7-0cc5-406c-89e7-fe131d55ddf5.xlsx"
EMDAT_SHEET = "EM-DAT Data"
FINAL_DATASET_FILE = "final_dataset.csv"
OUTPUT_CSV = "merged_flood_insurance_dataset.csv"
OUTPUT_PLOT = "insured_damage_over_time.png"
CORRELATION_PLOT = "correlation_heatmap.png"
CO2_FORECAST_PLOT = "co2_forecast_2025.png"
WDI_FILTERED_FILE = "wdi_filtered.csv"

COUNTRIES_ISO = {"IDN": "Indonesia", "MYS": "Malaysia"}
INSURED_DAMAGE_COL = "Insured Damage ('000 US$)"
TOTAL_DAMAGE_COL = "Total Damage, Adjusted ('000 US$)"
DISASTER_TYPE_COL = "Disaster Type"
ISO_COL = "ISO"
YEAR_COL = "Start Year"

# WDI indicators to include in the correlation analysis
# These are expected indicator codes present in the final_dataset INDICATOR column
WDI_INDICATORS_OF_INTEREST = [
    "EN.ATM.CO2E.PC",     # CO2 emissions (metric tons per capita)
    "SP.POP.TOTL",        # Population, total
    "AG.LND.PRCP.MM",     # Average precipitation in depth (mm per year)
    "NY.GDP.PCAP.CD",     # GDP per capita (current US$)
    "EN.CLC.MDAT.ZS",     # Droughts, floods, extreme temps (% of pop. affected)
]

# ---------------------------------------------------------------------------
# 1. Load data files
# ---------------------------------------------------------------------------

def load_emdat(filepath: str) -> pd.DataFrame:
    """Load the EM-DAT Excel file and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"EM-DAT file not found: '{filepath}'\n"
            "Please place the EM-DAT Excel file in the same directory as this script."
        )
    print(f"Loading EM-DAT data from '{filepath}' (sheet: '{EMDAT_SHEET}') ...")
    df = pd.read_excel(filepath, sheet_name=EMDAT_SHEET, engine="openpyxl")
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def load_final_dataset(filepath: str) -> pd.DataFrame:
    """Load the final merged WDI/flood-count dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Final dataset file not found: '{filepath}'\n"
            "Please place final_dataset.csv in the same directory as this script."
        )
    print(f"Loading final dataset from '{filepath}' ...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# 2. Filter flood events for IDN and MYS
# ---------------------------------------------------------------------------

def filter_flood_events(emdat_df: pd.DataFrame) -> pd.DataFrame:
    """Filter EM-DAT records to flood events for Indonesia and Malaysia."""
    iso_codes = list(COUNTRIES_ISO.keys())

    # Validate required columns exist
    for col in [DISASTER_TYPE_COL, ISO_COL, YEAR_COL]:
        if col not in emdat_df.columns:
            raise KeyError(
                f"Expected column '{col}' not found in EM-DAT data.\n"
                f"Available columns: {list(emdat_df.columns)}"
            )

    # Filter by country ISO code
    country_mask = emdat_df[ISO_COL].isin(iso_codes)
    # Filter by disaster type containing 'Flood' (case-insensitive)
    flood_mask = emdat_df[DISASTER_TYPE_COL].str.contains("Flood", case=False, na=False)

    flood_df = emdat_df[country_mask & flood_mask].copy()
    print(
        f"\nFiltered flood events for {iso_codes}: {len(flood_df):,} records "
        f"({flood_df[ISO_COL].value_counts().to_dict()})"
    )

    # Map ISO code to country name for merging with final_dataset
    flood_df["Country"] = flood_df[ISO_COL].map(COUNTRIES_ISO)
    flood_df = flood_df.rename(columns={YEAR_COL: "Year"})

    return flood_df


# ---------------------------------------------------------------------------
# 3. Aggregate insured damage by country and year
# ---------------------------------------------------------------------------

def aggregate_insured_damage(flood_df: pd.DataFrame) -> pd.DataFrame:
    """Sum insured and total damages per country per year."""
    agg_cols = {"Year": "Year", "Country": "Country"}

    # Add damage columns only if they exist in the data
    for col in [INSURED_DAMAGE_COL, TOTAL_DAMAGE_COL]:
        if col in flood_df.columns:
            agg_cols[col] = col
        else:
            print(f"  Warning: column '{col}' not found – will be omitted.")

    damage_cols = [c for c in [INSURED_DAMAGE_COL, TOTAL_DAMAGE_COL] if c in flood_df.columns]

    if not damage_cols:
        raise KeyError(
            "Neither insured nor total damage columns were found in the EM-DAT data."
        )

    # Coerce damage columns to numeric (some entries may be empty strings or '--')
    for col in damage_cols:
        flood_df[col] = pd.to_numeric(flood_df[col], errors="coerce")

    grouped = (
        flood_df.groupby(["Country", "Year"])[damage_cols]
        .sum(min_count=1)  # NaN if all values in a group are NaN
        .reset_index()
    )

    print(f"\nAggregated insured damage: {len(grouped):,} country-year records.")
    print(grouped.head(10).to_string(index=False))
    return grouped


# ---------------------------------------------------------------------------
# 4. Merge with final_dataset
# ---------------------------------------------------------------------------

def merge_datasets(
    damage_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge insured damage time series with WDI indicators from final_dataset.

    final_dataset has a long format with columns:
      Country, Year, Flood_Count, REF_AREA, INDICATOR, WDI_Value

    We pivot it to wide format (one WDI column per indicator) then merge.
    """
    # Ensure Year types are compatible
    damage_df["Year"] = damage_df["Year"].astype(int)
    final_df["Year"] = final_df["Year"].astype(int)

    # Pivot final_dataset from long to wide on INDICATOR
    if "INDICATOR" in final_df.columns and "WDI_Value" in final_df.columns:
        wide_df = final_df.pivot_table(
            index=["Country", "Year", "Flood_Count"],
            columns="INDICATOR",
            values="WDI_Value",
            aggfunc="first",
        ).reset_index()
        wide_df.columns.name = None  # remove column name from multi-index
    else:
        # Already wide or different structure – use as-is
        wide_df = final_df.copy()

    merged = wide_df.merge(damage_df, on=["Country", "Year"], how="left")

    # Fill missing insured/total damage with 0 and flag them
    damage_cols = [c for c in [INSURED_DAMAGE_COL, TOTAL_DAMAGE_COL] if c in merged.columns]
    for col in damage_cols:
        n_missing = merged[col].isna().sum()
        if n_missing > 0:
            print(
                f"\n  Note: {n_missing} year(s) have no recorded insured damage for '{col}'. "
                "These are filled with 0 (no reported flood damage does not imply zero economic loss)."
            )
        merged[col] = merged[col].fillna(0)

    print(f"\nMerged dataset: {len(merged):,} rows, {merged.shape[1]} columns.")
    return merged


# ---------------------------------------------------------------------------
# 5. Plot insured damage over time
# ---------------------------------------------------------------------------

def plot_insured_damage(damage_df: pd.DataFrame, output_path: str) -> None:
    """Line chart of insured flood damage over time for each country."""
    if INSURED_DAMAGE_COL not in damage_df.columns:
        print(f"  Skipping plot: column '{INSURED_DAMAGE_COL}' not available.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {"Indonesia": "#1f77b4", "Malaysia": "#ff7f0e"}

    for country, group in damage_df.groupby("Country"):
        group_sorted = group.sort_values("Year")
        ax.plot(
            group_sorted["Year"],
            group_sorted[INSURED_DAMAGE_COL],
            marker="o",
            linewidth=2,
            label=country,
            color=palette.get(country),
        )

    ax.set_title(
        "Insured Flood Damage Over Time\n(Indonesia & Malaysia, EM-DAT)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Insured Damage ('000 US$)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(title="Country", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nInsured damage plot saved to '{output_path}'.")


# ---------------------------------------------------------------------------
# 6. Correlation analysis
# ---------------------------------------------------------------------------

def print_and_plot_correlation(merged_df: pd.DataFrame, output_path: str) -> None:
    """Compute and display the correlation matrix between insured damage and WDI indicators."""
    damage_cols = [c for c in [INSURED_DAMAGE_COL, TOTAL_DAMAGE_COL] if c in merged_df.columns]

    # Determine which WDI indicator columns are actually present
    available_wdi = [c for c in WDI_INDICATORS_OF_INTEREST if c in merged_df.columns]

    # Fall back to all numeric columns if no recognised WDI codes are found
    if not available_wdi:
        numeric_cols = merged_df.select_dtypes(include="number").columns.tolist()
        available_wdi = [
            c for c in numeric_cols
            if c not in damage_cols + ["Year", "Flood_Count"]
        ]
        if available_wdi:
            print(
                "\n  Note: Pre-defined WDI indicator codes not found in merged dataset. "
                f"Using all numeric columns instead: {available_wdi}"
            )

    corr_cols = damage_cols + available_wdi
    if len(corr_cols) < 2:
        print("  Not enough numeric columns for a correlation matrix – skipping.")
        return

    corr_matrix = merged_df[corr_cols].corr()

    print("\n" + "=" * 70)
    print("Correlation Matrix: Insured Damage vs. WDI Indicators")
    print("=" * 70)
    print(corr_matrix.to_string())
    print("=" * 70)

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(corr_cols)), max(6, len(corr_cols) - 1)))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Correlation Matrix: Insured Flood Damage vs. WDI Climate Indicators",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nCorrelation heatmap saved to '{output_path}'.")


# ---------------------------------------------------------------------------
# 7. Save merged dataset
# ---------------------------------------------------------------------------

def save_merged_dataset(merged_df: pd.DataFrame, output_path: str) -> None:
    """Save the merged dataset to CSV."""
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged dataset saved to '{output_path}' ({len(merged_df):,} rows).")


# ---------------------------------------------------------------------------
# 8. CO2 / GHG emissions prediction model
# ---------------------------------------------------------------------------

# Indicator names as they appear after pivoting wdi_filtered.csv
_TARGET = "total_ghg_kt"
_PREDICTORS = ["gdp_per_capita", "forest_pct", "renewable_pct"]

_TRAIN_YEARS = (2003, 2018)
_VAL_YEARS = (2019, 2023)
_FORECAST_YEAR = 2025
_COUNTRIES = ["Indonesia", "Malaysia"]


def build_co2_prediction_model(wdi_filepath: str = "wdi_filtered.csv") -> dict:
    """Train a linear regression model to predict total GHG emissions per country.

    Loads ``wdi_filtered.csv`` (long format with columns COUNTRY, year, indicator,
    value) and fits a multi-variate linear regression for Indonesia and Malaysia.

    Three predictors drive the model:

    * ``gdp_per_capita``  – GDP per capita (USD); a proxy for economic activity
      and energy demand, which tends to correlate positively with emissions.
    * ``forest_pct``      – Forest area as a percentage of land area; higher
      forest cover acts as a carbon sink and is expected to correlate negatively
      with net GHG emissions.
    * ``renewable_pct``   – Renewable energy share of total final energy
      consumption (%); a higher share indicates cleaner energy mix, expected to
      correlate negatively with GHG output.

    For each country the function:

    1. Loads and pivots the CSV to wide format (one column per indicator).
    2. Drops rows with NaN in the target or any predictor.
    3. Splits into training (``_TRAIN_YEARS``) and validation (``_VAL_YEARS``) sets.
    4. Fits a ``LinearRegression`` model and prints coefficients and validation RMSE.
    5. Forecasts GHG for ``_FORECAST_YEAR`` (2025) by carrying forward the last
       available (2023) values of the three predictors.
    6. Saves a line plot (``CO2_FORECAST_PLOT``) showing historical GHG, model fit,
       and the 2025 forecast with a shaded ±1 RSE prediction interval.

    Parameters
    ----------
    wdi_filepath:
        Path to the clean WDI CSV file (long format, default ``"wdi_filtered.csv"``).

    Returns
    -------
    dict
        ``{country_name: forecasted_ghg_2025}`` for each successfully modelled
        country.  Countries with insufficient data are omitted.
    """
    # ---- Load and validate the CSV ----
    if not os.path.exists(wdi_filepath):
        print(
            f"\n  Warning: WDI file not found: '{wdi_filepath}'. "
            "Skipping CO2/GHG prediction model."
        )
        return {}

    print(f"\nLoading WDI filtered data from '{wdi_filepath}' ...")
    wdi_df = pd.read_csv(wdi_filepath)
    print(f"  Loaded {len(wdi_df):,} rows, {wdi_df.shape[1]} columns.")

    required_raw_cols = {"COUNTRY", "year", "indicator", "value"}
    missing_raw = required_raw_cols - set(wdi_df.columns)
    if missing_raw:
        print(
            f"\n  Warning: expected columns {missing_raw} not found in '{wdi_filepath}'. "
            "Skipping CO2/GHG prediction model."
        )
        return {}

    # ---- Pivot long → wide ----
    pivot = wdi_df.pivot_table(
        index=["COUNTRY", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"COUNTRY": "Country", "year": "Year"})
    pivot["Year"] = pivot["Year"].astype(int)

    # Validate all required columns exist after pivoting
    all_required = [_TARGET] + _PREDICTORS
    missing_cols = [c for c in all_required if c not in pivot.columns]
    if missing_cols:
        print(
            f"\n  Warning: after pivoting, columns {missing_cols} are missing. "
            "Skipping CO2/GHG prediction model."
        )
        return {}

    forecasts: dict = {}
    palette = {"Indonesia": "#1f77b4", "Malaysia": "#ff7f0e"}

    fig, axes = plt.subplots(1, len(_COUNTRIES), figsize=(14, 6), sharey=False)
    if len(_COUNTRIES) == 1:
        axes = [axes]

    for ax, country in zip(axes, _COUNTRIES):
        country_df = pivot[pivot["Country"] == country].sort_values("Year").copy()

        if country_df.empty:
            print(f"\n  Warning [{country}]: no rows found – skipping.")
            ax.set_title(f"{country}\n(no data)", fontsize=12)
            ax.axis("off")
            continue

        # ---- Drop rows with NaN in target or any predictor ----
        country_df = country_df.dropna(subset=all_required)

        if len(country_df) < 5:
            print(
                f"\n  Warning [{country}]: only {len(country_df)} complete rows "
                "after dropping NaNs – skipping."
            )
            ax.set_title(f"{country}\n(insufficient data)", fontsize=12)
            ax.axis("off")
            continue

        # ---- Train / validation split ----
        train_df = country_df[
            country_df["Year"].between(_TRAIN_YEARS[0], _TRAIN_YEARS[1])
        ]
        val_df = country_df[
            country_df["Year"].between(_VAL_YEARS[0], _VAL_YEARS[1])
        ]

        if train_df.empty:
            print(
                f"\n  Warning [{country}]: no training rows in "
                f"{_TRAIN_YEARS[0]}–{_TRAIN_YEARS[1]} – skipping."
            )
            ax.set_title(f"{country}\n(insufficient training data)", fontsize=12)
            ax.axis("off")
            continue

        X_train = train_df[_PREDICTORS].values
        y_train = train_df[_TARGET].values

        # ---- Fit linear regression ----
        model = LinearRegression()
        model.fit(X_train, y_train)

        print(f"\n{'─' * 60}")
        print(f"GHG Prediction Model – {country}")
        print(f"{'─' * 60}")
        print(f"  Intercept : {model.intercept_:.4f}")
        for name, coef in zip(_PREDICTORS, model.coef_):
            print(f"  {name:<20}: {coef:.6f}")

        if not val_df.empty:
            X_val = val_df[_PREDICTORS].values
            y_val = val_df[_TARGET].values
            y_val_pred = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            print(f"  RMSE (validation {_VAL_YEARS[0]}–{_VAL_YEARS[1]}): {rmse:.4f}")
        else:
            print(
                f"  Note: no validation rows in "
                f"{_VAL_YEARS[0]}–{_VAL_YEARS[1]} – RMSE not computed."
            )

        # ---- Forecast 2025 ─────────────────────────────────────────────────
        # Assumption: gdp_per_capita, forest_pct, and renewable_pct for 2025
        # are assumed to equal their most recently available values (2023 where
        # present, otherwise the last observed year).  This is a simplifying
        # carry-forward assumption for short-horizon extrapolation.
        last_row = country_df.iloc[-1]
        X_forecast = np.array([[last_row[p] for p in _PREDICTORS]])
        ghg_2025 = float(model.predict(X_forecast)[0])
        forecasts[country] = ghg_2025
        print(f"  Forecast GHG for {_FORECAST_YEAR}: {ghg_2025:.4f} kt CO2-eq")

        # ---- Build plot for this country ----
        color = palette.get(country, "steelblue")

        # Residual standard error on training set for prediction interval
        y_train_pred = model.predict(X_train)
        residuals = y_train - y_train_pred
        n = len(residuals)
        p = len(_PREDICTORS)
        rse = float(np.sqrt(np.sum(residuals ** 2) / max(n - p - 1, 1)))

        # Fitted values over all historical data
        X_all = country_df[_PREDICTORS].values
        y_fitted = model.predict(X_all)

        ax.plot(
            country_df["Year"],
            country_df[_TARGET],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label="Historical GHG",
        )
        ax.plot(
            country_df["Year"],
            y_fitted,
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.75,
            label="Model fit",
        )
        # 2025 forecast point
        ax.scatter(
            [_FORECAST_YEAR],
            [ghg_2025],
            color="red",
            zorder=5,
            s=80,
            label=f"{_FORECAST_YEAR} forecast",
        )
        # Shaded prediction interval (±1 RSE) around the 2025 forecast
        ax.fill_between(
            [_FORECAST_YEAR - 0.4, _FORECAST_YEAR + 0.4],
            [ghg_2025 - rse, ghg_2025 - rse],
            [ghg_2025 + rse, ghg_2025 + rse],
            color="red",
            alpha=0.25,
            label=f"±1 RSE ({rse:.2f})",
        )
        # Vertical dashed line at forecast year
        ax.axvline(_FORECAST_YEAR, color="grey", linestyle=":", linewidth=1)

        ax.set_title(f"{country}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Total GHG Emissions (kt CO2-eq)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Historical GHG Emissions & {_FORECAST_YEAR} Forecast\n"
        "(Linear Regression: GDP/capita + Forest % + Renewable %)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(CO2_FORECAST_PLOT, dpi=150)
    plt.close()
    print(f"\nGHG forecast plot saved to '{CO2_FORECAST_PLOT}'.")

    return forecasts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Flood Insurance Analysis – Indonesia & Malaysia")
    print("MASA Hackathon 2026: R-Ignite")
    print("=" * 70 + "\n")

    # 1. Load data
    emdat_df = load_emdat(EMDAT_FILE)
    final_df = load_final_dataset(FINAL_DATASET_FILE)

    # 2. Filter flood events for IDN/MYS
    flood_df = filter_flood_events(emdat_df)

    # 3. Aggregate insured damage by country and year
    damage_agg = aggregate_insured_damage(flood_df)

    # 4. Merge with final_dataset
    merged_df = merge_datasets(damage_agg, final_df)

    # 5. Plot insured damage over time
    plot_insured_damage(damage_agg, OUTPUT_PLOT)

    # 6. Correlation matrix
    print_and_plot_correlation(merged_df, CORRELATION_PLOT)

    # 7. Save merged dataset
    save_merged_dataset(merged_df, OUTPUT_CSV)

    # 8. CO2 / GHG prediction model and 2025 forecast
    co2_forecasts = build_co2_prediction_model(WDI_FILTERED_FILE)
    if co2_forecasts:
        print("\n" + "=" * 70)
        print(f"GHG Emissions Forecast for {_FORECAST_YEAR}")
        print("=" * 70)
        for country, value in co2_forecasts.items():
            print(f"  {country}: {value:.4f} kt CO2-eq")
        print("=" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()
