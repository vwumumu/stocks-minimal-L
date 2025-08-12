# S&P 500 Minimal L Analysis Script

This script analyzes S&P 500 annual returns using historical baselines to perform two main tasks:

## Tasks

1. **Task 1**: Find the minimal L such that the worst L-year cumulative return is ≥ 0 (within tolerance)
2. **Task 2**: For given τ values, find the minimal L such that max CAGR - min CAGR ≤ τ (within tolerance)

## Data Sources

The script supports CSV data sources in order of preference:

1. **Local history.csv**: First tries to load a local file named "history.csv" in the current directory
2. **Remote CSV**: If local file not found, attempts to load from a CSV URL (e.g., Slickcharts)
3. **User-provided local CSV**: If remote download fails, prompts user to provide a local CSV file path

## Installation

```bash
pip3 install -r requirements.txt
```

## Usage

### Basic usage (uses default CSV URL):
```bash
python3 sp500-minimal-L.py
```

### With custom parameters:
```bash
python3 sp500-minimal-L.py \
  --baselines 1926 1957 1972 1984 \
  --taus 0.005 0.01 0.015 \
  --tol 1e-9
```

### Using custom CSV URL:
```bash
python3 sp500-minimal-L.py \
  --csv-url https://www.slickcharts.com/sp500/returns/history.csv
```

## Parameters

- `--csv-url`: CSV URL for remote data (default: Slickcharts URL)
- `--baselines`: List of baseline years to analyze (default: [1926, 1957, 1972, 1984])
- `--taus`: List of CAGR dispersion thresholds in decimal (default: [0.005, 0.01, 0.015])
- `--tol`: Tolerance for comparisons (default: 1e-9)

## Output

The script outputs analysis results for each baseline year, showing:
- Data span used
- Task 1 results (no-loss horizon) including windows tested count
- Task 2 results for each τ value including windows tested count

## Troubleshooting

### HTTP 403 Error
If you encounter a 403 Forbidden error when trying to access the Slickcharts CSV, this is because the website blocks direct programmatic access. The script will prompt you to provide a local CSV file path as an alternative.

### CSV File Formats
The script supports two CSV formats:

1. **Header format**: CSV with column headers including "Year" and a return column
2. **Headless format**: CSV without headers, with year and return values in each row (e.g., `2024,25.02`)

The local `history.csv` file uses the headless format for simplicity.

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
