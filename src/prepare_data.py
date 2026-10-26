"""Clean both datasets and save the training and test sets.

Run from the project folder:
    python -m src.prepare_data

Reads:   data/raw/sgcc/data.csv and data/raw/oedi/df.csv
Writes:  data/processed/sgcc/train.csv, test.csv
         data/processed/oedi/train.csv, test.csv
         outputs/data_quality.md   (numbers for the report)

All settings (seed, test size, folders) come from config.yaml.
"""

from src import data_oedi, data_sgcc
from src.config import load_config
from src.utils import set_seed

# Six significant digits is plenty for kWh values and keeps the CSV
# files smaller than writing every float in full.
FLOAT_FORMAT = "%.6g"


def save_split(train, test, folder):
    """Save the train and test sets as CSV files in the given folder."""
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train.csv", index=False, float_format=FLOAT_FORMAT)
    test.to_csv(folder / "test.csv", index=False, float_format=FLOAT_FORMAT)


def share_line(name, frame, label):
    """One summary line: size of a set and its theft share."""
    return f"| {name} | {len(frame):,} | {frame[label].mean():.1%} |"


def prepare_sgcc(config):
    """Load, clean, split and save SGCC. Returns lines for the summary."""
    paths = config["paths"]
    raw = data_sgcc.load_sgcc(paths["sgcc_raw"] / "data.csv")
    clean, r = data_sgcc.clean_sgcc(raw)
    train, test = data_sgcc.split_sgcc(clean, config["test_size"], config["random_seed"])
    save_split(train, test, paths["sgcc_processed"])

    return [
        "## SGCC",
        "",
        f"- Customers in raw file: {r['customers_raw']:,}; days: {r['days_raw']:,}",
        f"- Missing readings in raw file: {r['missing_share_raw']:.1%}",
        f"- Days added to complete the date range: {', '.join(r['days_added']) or 'none'}",
        f"- Customers removed (no readings at all): {r['customers_removed_empty']}",
        f"- Readings capped by the IQR rule: {r['readings_capped']:,} "
        f"(from {r['customers_capped']:,} customers)",
        f"- Customers skipped by the IQR rule (IQR = 0): {r['customers_skipped_iqr_zero']:,}",
        f"- Missing readings after cleaning: {r['missing_after_cleaning']}",
        f"- Clean data: {r['customers_clean']:,} customers x {r['days_clean']:,} days",
        "",
        "| Set | Customers | Theft share |",
        "|---|---|---|",
        share_line("Train", train, data_sgcc.LABEL_COL),
        share_line("Test", test, data_sgcc.LABEL_COL),
        "",
    ]


def prepare_oedi(config):
    """Load, clean, split and save OEDI. Returns lines for the summary."""
    paths = config["paths"]
    raw = data_oedi.load_oedi(paths["oedi_raw"] / "df.csv")
    clean, r = data_oedi.clean_oedi(raw)
    train, test = data_oedi.split_oedi(
        clean, config["oedi_test_recordings_per_type"], config["random_seed"]
    )
    save_split(train, test, paths["oedi_processed"])

    return [
        "## OEDI",
        "",
        f"- Rows (hours) in raw file: {r['rows_raw']:,}; missing values: {r['missing_values_raw']}",
        f"- Broken rows removed (building type \"0\"): {r['rows_removed_broken']}",
        f"- Clean data: {r['rows_clean']:,} hours = {r['recordings']} one-year recordings "
        f"of {r['building_types']} building types",
        f"- Theft share (all hours): {r['theft_share']:.1%}",
        f"- Test recordings: {test['recording_id'].nunique()} "
        f"(IDs {sorted(test['recording_id'].unique().tolist())})",
        "",
        "| Set | Hours | Theft share |",
        "|---|---|---|",
        share_line("Train", train, "theft"),
        share_line("Test", test, "theft"),
        "",
    ]


def main():
    config = load_config()
    set_seed(config["random_seed"])

    print("Cleaning SGCC...")
    lines = ["# Data quality summary", "", "Created by `python -m src.prepare_data`.", ""]
    lines += prepare_sgcc(config)
    print("Cleaning OEDI...")
    lines += prepare_oedi(config)

    out_dir = config["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "data_quality.md"
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
