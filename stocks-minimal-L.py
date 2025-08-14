#!/usr/bin/env python3
"""
Stock/Index baseline analysis (Total Return) — Tasks 1 & 2, finding minimal L

Supports analysis of:
- S&P 500 index
- NASDAQ-100 index  
- Berkshire Hathaway stock
- Bitcoin (BTC)
- Other stocks/indices via CSV files

Logic:
  * Prefer "Total Return" if available; otherwise fall back to "Price Return".
  * Work in nominal terms.
  * For each baseline year:
      - Restrict to years >= baseline.
      - If there are gaps, start at the first available year >= baseline and keep
        the contiguous run until the first gap (inclusive).
  * Task 1: Find the minimal L such that the worst L-year cumulative return is >= 0 (tolerance 1e-9).
  * Task 2: For τ ∈ {0.5%, 1.0%, 1.5%} find the minimal L s.t. max CAGR - min CAGR <= τ (tolerance 1e-9).
  * Numerical stability: compute with log1p/expm1.

Usage:
  python stocks-minimal-L.py
  python stocks-minimal-L.py --data-source nasdaq100
  python stocks-minimal-L.py --data-source brk --baselines 1990 2000 2010
  python stocks-minimal-L.py --data-source btc

Output is printed to stdout.

Notes:
  - CSV schema supports both headless format (year,return) and header format with Year column
  - Values are expected to be percentages
  - If CSV download fails, the script will prompt for a local file path.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any
import json

import numpy as np
import pandas as pd


DEFAULT_CSV_URL = "https://www.slickcharts.com/sp500/returns/history.csv"
DEFAULT_BASELINES = [1926, 1957, 1972, 1984]
DEFAULT_TAUS = [0.005, 0.01, 0.015]
DEFAULT_TOL = 1e-9

# Data source options
DATA_SOURCES = {
    'sp500': 'data/sp500-history.csv',
    'nasdaq100': 'data/ndsq100-history.csv',
    'brk': 'data/brk-history.csv',
    'btc': 'data/btc-history.csv'
}

# ----------------- I/O & parsing -----------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _find_return_column(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (colname, label) where colname exists in df and label is a friendly name.
    Prefer a column containing 'Total Return' (case-insensitive).
    Fallback to a column containing 'Price Return'.
    Raise if none found.
    """
    cols = {c.lower(): c for c in df.columns}
    # exact common names
    for k in ["total return", "total return (%)"]:
        if k in cols: return cols[k], "Total Return (%)"
    # fuzzy find
    for c in df.columns:
        cl = c.lower()
        if "total" in cl and "return" in cl:
            return c, "Total Return (%)"
    # fallback: price return
    for k in ["price return", "price return (%)"]:
        if k in cols: return cols[k], "Price Return (%)"
    for c in df.columns:
        cl = c.lower()
        if "price" in cl and "return" in cl:
            return c, "Price Return (%)"
    # fallback: any return column
    for c in df.columns:
        cl = c.lower()
        if "return" in cl:
            return c, c
    raise ValueError("No return column found in CSV.")

def _load_csv(csv_url: str) -> Tuple[pd.DataFrame, str]:
    """Load CSV from URL or local file path."""
    try:
        print(f"正在从 {csv_url} 加载CSV文件...")
        
        # Try to detect if it's a headless CSV by checking the first few lines
        try:
            # First try to read as regular CSV with headers
            df = pd.read_csv(csv_url)
            df = _normalize_cols(df)
            
            # Check if it has a Year column
            year_col = None
            for c in df.columns:
                if c.strip().lower() == "year":
                    year_col = c
                    break
            
            if year_col is None:
                # No Year column found, try as headless CSV
                print("未找到'Year'列，尝试作为无标题CSV处理...")
                df = pd.read_csv(csv_url, header=None, names=["Year", "Return"])
                df = _normalize_cols(df)
                out = df[["Year", "Return"]].rename(columns={"Return": "Return (%)"})
            else:
                # Regular CSV with headers
                ret_col, ret_label = _find_return_column(df)
                out = df[[year_col, ret_col]].rename(columns={year_col: "Year", ret_col: ret_label})
                
        except Exception:
            # If regular CSV reading fails, try as headless CSV
            print("尝试作为无标题CSV处理...")
            df = pd.read_csv(csv_url, header=None, names=["Year", "Return"])
            df = _normalize_cols(df)
            out = df[["Year", "Return"]].rename(columns={"Return": "Return (%)"})
        
        # Coerce numeric (handle stray % signs or commas if any)
        def to_float(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (int, float)): return float(x)
            s = str(x).replace(",", "").replace("%", "").strip()
            return float(s) if s else np.nan
        out["Return (%)"] = out["Return (%)"].map(to_float)
        return out, "Return (%)"
    except Exception as e:
        print(f"从 {csv_url} 加载CSV时出错: {e}")
        raise

def _prompt_for_local_file() -> str:
    """Prompt user to provide a local CSV file path."""
    print("\nCSV下载失败。请提供本地CSV文件路径。")
    print("CSV文件应包含以下任一格式:")
    print("- 包含'Year'列和收益率列 (总收益率或价格收益率), 或")
    print("- 无标题格式，包含年份,收益率对 (例如: 2024,25.02)")
    while True:
        try:
            file_path = input("请输入本地CSV文件路径: ").strip()
            if not file_path:
                print("请输入有效的文件路径。")
                continue
            
            # Test if file exists and can be read
            try:
                # First try as regular CSV
                test_df = pd.read_csv(file_path)
                if "Year" in [col.strip() for col in test_df.columns]:
                    return file_path
            except:
                pass
            
            # Try as headless CSV
            try:
                test_df = pd.read_csv(file_path, header=None, names=["Year", "Return"])
                # Check if first row looks like year data
                if test_df.iloc[0, 0] > 1900 and test_df.iloc[0, 0] < 2100:
                    return file_path
            except:
                pass
                
            print("错误: 文件必须是有效的CSV格式，包含标题或年份,收益率格式。")
            continue
                
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 未找到。请检查路径。")
        except Exception as e:
            print(f"读取文件时出错: {e}。请重试。")

def _load_data(csv_url: str = None, data_source: str = 'sp500') -> Tuple[pd.DataFrame, str]:
    """Load data from local CSV file, CSV URL, or user-provided local file."""
    # First try to load the specified local CSV file
    local_file = DATA_SOURCES.get(data_source, 'sp500-history.csv')
    try:
        print(f"正在尝试加载本地文件 {local_file}...")
        return _load_csv(local_file)
    except Exception as e:
        print(f"本地文件 {local_file} 未找到: {e}")
        
        # Then try the default URL
        if csv_url is None:
            csv_url = DEFAULT_CSV_URL
        
        try:
            return _load_csv(csv_url)
        except Exception as e:
            print(f"加载CSV数据失败: {e}")
            local_file = _prompt_for_local_file()
            return _load_csv(local_file)

# ----------------- Core math -----------------

def first_contiguous_span(years: pd.Series, baseline: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Given sorted unique years, return (start_year, end_year) for the first
    contiguous run beginning at the first year >= baseline and ending just
    before the first gap (>1 year step). Returns (None, None) if none.
    """
    ys = years[years >= baseline].dropna().astype(int).to_numpy()
    if ys.size == 0:
        return (None, None)
    start = int(ys[0])
    end = int(ys[-1])
    for i in range(len(ys) - 1):
        if ys[i + 1] - ys[i] > 1:
            end = int(ys[i])
            break
    return (start, end)

@dataclass
class Task1Result:
    L: int
    worst_window_years: Tuple[int, int]
    worst_window_cumret: float  # decimal
    windows_tested: int

@dataclass
class Task2Result:
    tau: float
    L: int
    min_window_years: Tuple[int, int]
    min_window_cagr: float  # decimal
    max_window_years: Tuple[int, int]
    max_window_cagr: float  # decimal
    spread: float           # decimal
    windows_tested: int

@dataclass
class Task3Result:
    L: int
    best_window_years: Tuple[int, int]
    best_window_cumret: float  # decimal
    best_window_cagr: float  # decimal
    windows_tested: int

@dataclass
class Task4Result:
    L: int
    worst_window_years: Tuple[int, int]
    worst_window_cumret: float  # decimal
    worst_window_cagr: float  # decimal
    windows_tested: int

def task1_no_loss_horizon(years: pd.Series, returns_decimal: pd.Series, tol: float = DEFAULT_TOL) -> Optional[Task1Result]:
    y = years.to_numpy()
    logs = np.log1p(returns_decimal.to_numpy())
    n = len(logs)
    pref = np.zeros(n + 1); pref[1:] = np.cumsum(logs)

    def cumret(i, L):
        return math.expm1(pref[i + L] - pref[i])

    total_windows_tested = 0
    for L in range(1, n + 1):
        windows_in_L = n - L + 1
        total_windows_tested += windows_in_L
        worst_val = None; worst_i = None
        for i in range(0, n - L + 1):
            v = cumret(i, L)
            if worst_val is None or v < worst_val:
                worst_val, worst_i = v, i
        if worst_val is not None and worst_val >= -tol:
            return Task1Result(
                L=L,
                worst_window_years=(int(y[worst_i]), int(y[worst_i + L - 1])),
                worst_window_cumret=float(worst_val),
                windows_tested=total_windows_tested,
            )
    return None

def task2_cagr_dispersion(years: pd.Series, returns_decimal: pd.Series, tau: float, tol: float = DEFAULT_TOL) -> Optional[Task2Result]:
    y = years.to_numpy()
    logs = np.log1p(returns_decimal.to_numpy())
    n = len(logs)
    pref = np.zeros(n + 1); pref[1:] = np.cumsum(logs)

    total_windows_tested = 0
    for L in range(1, n + 1):
        windows_in_L = n - L + 1
        total_windows_tested += windows_in_L
        min_v = None; min_i = None
        max_v = None; max_i = None
        for i in range(0, n - L + 1):
            s = pref[i + L] - pref[i]
            cagr = math.expm1(s / L)
            if min_v is None or cagr < min_v:
                min_v, min_i = cagr, i
            if max_v is None or cagr > max_v:
                max_v, max_i = cagr, i
        spread = max_v - min_v
        if spread <= tau + tol:
            return Task2Result(
                tau=tau,
                L=L,
                min_window_years=(int(y[min_i]), int(y[min_i + L - 1])),
                min_window_cagr=float(min_v),
                max_window_years=(int(y[max_i]), int(y[max_i + L - 1])),
                max_window_cagr=float(max_v),
                spread=float(spread),
                windows_tested=total_windows_tested,
            )
    return None

def task3_best_window(years: pd.Series, returns_decimal: pd.Series) -> Optional[Task3Result]:
    """Find the best performing window across all possible window lengths."""
    y = years.to_numpy()
    logs = np.log1p(returns_decimal.to_numpy())
    n = len(logs)
    pref = np.zeros(n + 1); pref[1:] = np.cumsum(logs)
    
    best_overall_val = None
    best_overall_i = None
    best_overall_L = None
    total_windows_tested = 0
    
    # Test all possible window lengths from 1 to n years
    for L in range(1, n + 1):
        windows_in_L = n - L + 1
        total_windows_tested += windows_in_L
        
        for i in range(0, n - L + 1):
            s = pref[i + L] - pref[i]
            cumret = math.expm1(s)
            
            if best_overall_val is None or cumret > best_overall_val:
                best_overall_val = cumret
                best_overall_i = i
                best_overall_L = L
    
    if best_overall_val is not None:
        # Calculate CAGR for the best window
        s = pref[best_overall_i + best_overall_L] - pref[best_overall_i]
        cagr = math.expm1(s / best_overall_L)
        
        return Task3Result(
            L=best_overall_L,
            best_window_years=(int(y[best_overall_i]), int(y[best_overall_i + best_overall_L - 1])),
            best_window_cumret=float(best_overall_val),
            best_window_cagr=float(cagr),
            windows_tested=total_windows_tested,
        )
    
    return None

def task4_worst_window(years: pd.Series, returns_decimal: pd.Series) -> Optional[Task4Result]:
    """Find the worst performing window across all possible window lengths."""
    y = years.to_numpy()
    logs = np.log1p(returns_decimal.to_numpy())
    n = len(logs)
    pref = np.zeros(n + 1); pref[1:] = np.cumsum(logs)
    
    worst_overall_val = None
    worst_overall_i = None
    worst_overall_L = None
    total_windows_tested = 0
    
    # Test all possible window lengths from 1 to n years
    for L in range(1, n + 1):
        windows_in_L = n - L + 1
        total_windows_tested += windows_in_L
        
        for i in range(0, n - L + 1):
            s = pref[i + L] - pref[i]
            cumret = math.expm1(s)
            
            if worst_overall_val is None or cumret < worst_overall_val:
                worst_overall_val = cumret
                worst_overall_i = i
                worst_overall_L = L
    
    if worst_overall_val is not None:
        # Calculate CAGR for the worst window
        s = pref[worst_overall_i + worst_overall_L] - pref[worst_overall_i]
        cagr = math.expm1(s / worst_overall_L)
        
        return Task4Result(
            L=worst_overall_L,
            worst_window_years=(int(y[worst_overall_i]), int(y[worst_overall_i + worst_overall_L - 1])),
            worst_window_cumret=float(worst_overall_val),
            worst_window_cagr=float(cagr),
            windows_tested=total_windows_tested,
        )
    
    return None

# ----------------- Reporting -----------------

def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:.2f}%"

def fmt_span(span: Tuple[int, int]) -> str:
    return f"{span[0]}–{span[1]}"

def print_report(baseline: int, span: Tuple[int, int], col_label: str,
                 t1: Optional[Task1Result],
                 t2s: Sequence[Optional[Task2Result]],
                 t3: Optional[Task3Result] = None,
                 t4: Optional[Task4Result] = None):
    print(f"\n基准年份 {baseline}")
    print(f"- 使用数据范围: {fmt_span(span)}")
    print(f"- 选择列: {col_label}")
    # Task 1
    if t1 is None:
        print("任务1 (无损失期限): 在可用范围内未找到。")
    else:
        print(f"任务1 (无损失期限): 最小L = {t1.L}年; "
              f"最差窗口 {fmt_span(t1.worst_window_years)}; "
              f"累计收益率 = {fmt_pct(t1.worst_window_cumret)}; "
              f"测试窗口数 = {t1.windows_tested}")
    # Task 2
    for res in t2s:
        if res is None:
            print("任务2: (在范围内未找到L)")
        else:
            tau_str = f"{res.tau*100:.1f}%"
            print(f"任务2 (τ = {tau_str}): 最小L = {res.L}年; "
                  f"最小 {fmt_span(res.min_window_years)} (年化收益率 {fmt_pct(res.min_window_cagr)}); "
                  f"最大 {fmt_span(res.max_window_years)} (年化收益率 {fmt_pct(res.max_window_cagr)}); "
                  f"差值 {fmt_pct(res.spread)}; "
                  f"测试窗口数 = {res.windows_tested}")
    # Task 3
    if t3 is not None:
        print(f"任务3 (最佳窗口): L = {t3.L}年; "
              f"窗口 {fmt_span(t3.best_window_years)}; "
              f"累计收益率 = {fmt_pct(t3.best_window_cumret)}; "
              f"年化收益率 = {fmt_pct(t3.best_window_cagr)}; "
              f"测试窗口数 = {t3.windows_tested}")
    # Task 4
    if t4 is not None:
        print(f"任务4 (最差窗口): L = {t4.L}年; "
              f"窗口 {fmt_span(t4.worst_window_years)}; "
              f"累计收益率 = {fmt_pct(t4.worst_window_cumret)}; "
              f"年化收益率 = {fmt_pct(t4.worst_window_cagr)}; "
              f"测试窗口数 = {t4.windows_tested}")

def collect_results(baseline: int, span: Tuple[int, int], col_label: str,
                    t1: Optional[Task1Result],
                    t2s: Sequence[Optional[Task2Result]],
                    t3: Optional[Task3Result] = None,
                    t4: Optional[Task4Result] = None) -> Dict[str, Any]:
    """Collect analysis results into a dictionary for HTML report."""
    result = {
        'baseline': baseline,
        'span': span,
        'col_label': col_label,
        'task1': None,
        'task2': [],
        'task3': None,
        'task4': None
    }
    
    if t1 is not None:
        result['task1'] = {
            'L': t1.L,
            'worst_window_years': t1.worst_window_years,
            'worst_window_cumret': t1.worst_window_cumret,
            'windows_tested': t1.windows_tested
        }
    
    for res in t2s:
        if res is not None:
            result['task2'].append({
                'tau': res.tau,
                'L': res.L,
                'min_window_years': res.min_window_years,
                'min_window_cagr': res.min_window_cagr,
                'max_window_years': res.max_window_years,
                'max_window_cagr': res.max_window_cagr,
                'spread': res.spread,
                'windows_tested': res.windows_tested
            })
        else:
            result['task2'].append(None)
    
    if t3 is not None:
        result['task3'] = {
            'L': t3.L,
            'best_window_years': t3.best_window_years,
            'best_window_cumret': t3.best_window_cumret,
            'best_window_cagr': t3.best_window_cagr,
            'windows_tested': t3.windows_tested
        }
    
    if t4 is not None:
        result['task4'] = {
            'L': t4.L,
            'worst_window_years': t4.worst_window_years,
            'worst_window_cumret': t4.worst_window_cumret,
            'worst_window_cagr': t4.worst_window_cagr,
            'windows_tested': t4.windows_tested
        }
    
    return result

def generate_html_report(all_results: Dict[str, List[Dict[str, Any]]], output_file: str = 'report.html'):
    """Generate an HTML report from analysis results."""
    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票/指数最小L分析报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.9);
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .source-section {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .source-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #f0f0f0;
        }
        .source-title {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }
        .source-icon {
            font-size: 2em;
        }
        .baseline-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .baseline-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #e9ecef;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .baseline-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .baseline-year {
            font-size: 1.3em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
        }
        .data-span {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }
        .task-section {
            margin-bottom: 15px;
        }
        .task-title {
            font-weight: 600;
            color: #343a40;
            margin-bottom: 8px;
            font-size: 1.05em;
        }
        .task-result {
            background: white;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        .metric-value {
            font-weight: 600;
            color: #2c3e50;
        }
        .highlight {
            background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .no-data {
            color: #dc3545;
            font-style: italic;
        }
        @media (max-width: 768px) {
            .baseline-container {
                grid-template-columns: 1fr;
            }
            h1 { font-size: 2em; }
            .source-title { font-size: 1.5em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 股票/指数最小L分析报告</h1>
        <div class="subtitle">寻找零损失和稳定收益的最佳投资期限</div>
'''
    
    # Add content for each data source
    source_icons = {
        'sp500': '📈',
        'nasdaq100': '💻',
        'brk': '🏦',
        'btc': '₿'
    }
    
    source_names = {
        'sp500': 'S&P 500',
        'nasdaq100': 'NASDAQ-100',
        'brk': 'Berkshire Hathaway',
        'btc': 'Bitcoin'
    }
    
    for source, results in all_results.items():
        if not results:
            continue
            
        html += f'''
        <div class="source-section">
            <div class="source-header">
                <div class="source-title">{source_names.get(source, source)}</div>
                <div class="source-icon">{source_icons.get(source, '📊')}</div>
            </div>
            
            <div class="baseline-container">
'''
        
        for result in results:
            baseline = result['baseline']
            span = result['span']
            
            html += f'''
                <div class="baseline-card">
                    <div class="baseline-year">基准年份: {baseline}</div>
                    <div class="data-span">数据范围: {span[0]}–{span[1]}</div>
'''
            
            # Task 1 results
            html += '<div class="task-section">'
            html += '<div class="task-title">任务1: 无损失期限</div>'
            if result['task1']:
                t1 = result['task1']
                html += f'''
                    <div class="task-result">
                        <div class="metric">
                            <span class="metric-label">最小L:</span>
                            <span class="metric-value highlight">{t1['L']} 年</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">最差窗口:</span>
                            <span class="metric-value">{t1['worst_window_years'][0]}–{t1['worst_window_years'][1]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">累计收益率:</span>
                            <span class="metric-value">{t1['worst_window_cumret']*100:.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">测试窗口数:</span>
                            <span class="metric-value">{t1['windows_tested']:,}</span>
                        </div>
                    </div>
'''
            else:
                html += '<div class="task-result"><span class="no-data">在可用范围内未找到</span></div>'
            html += '</div>'
            
            # Task 2 results
            html += '<div class="task-section">'
            html += '<div class="task-title">任务2: 年化收益率稳定性</div>'
            for i, t2 in enumerate(result['task2']):
                if t2:
                    tau_pct = t2['tau'] * 100
                    html += f'''
                    <div class="task-result" style="margin-bottom: 10px;">
                        <div style="font-weight: 600; color: #667eea; margin-bottom: 8px;">τ = {tau_pct:.1f}%</div>
                        <div class="metric">
                            <span class="metric-label">最小L:</span>
                            <span class="metric-value highlight">{t2['L']} 年</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">最小年化收益率:</span>
                            <span class="metric-value">{t2['min_window_cagr']*100:.2f}% ({t2['min_window_years'][0]}–{t2['min_window_years'][1]})</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">最大年化收益率:</span>
                            <span class="metric-value">{t2['max_window_cagr']*100:.2f}% ({t2['max_window_years'][0]}–{t2['max_window_years'][1]})</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">差值:</span>
                            <span class="metric-value">{t2['spread']*100:.2f}%</span>
                        </div>
                    </div>
'''
                elif i == 0:  # Only show "not found" once
                    html += '<div class="task-result"><span class="no-data">在范围内未找到L</span></div>'
            html += '</div>'
            
            # Task 3 results
            html += '<div class="task-section">'
            html += '<div class="task-title">任务3: 最佳投资窗口</div>'
            if result['task3']:
                t3 = result['task3']
                html += f'''
                    <div class="task-result">
                        <div class="metric">
                            <span class="metric-label">最佳窗口长度:</span>
                            <span class="metric-value highlight">{t3['L']} 年</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">时间段:</span>
                            <span class="metric-value">{t3['best_window_years'][0]}–{t3['best_window_years'][1]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">累计收益率:</span>
                            <span class="metric-value">{t3['best_window_cumret']*100:.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">年化收益率:</span>
                            <span class="metric-value">{t3['best_window_cagr']*100:.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">测试窗口数:</span>
                            <span class="metric-value">{t3['windows_tested']:,}</span>
                        </div>
                    </div>
'''
            else:
                html += '<div class="task-result"><span class="no-data">无数据</span></div>'
            html += '</div>'
            
            # Task 4 results
            html += '<div class="task-section">'
            html += '<div class="task-title">任务4: 最差投资窗口</div>'
            if result['task4']:
                t4 = result['task4']
                html += f'''
                    <div class="task-result">
                        <div class="metric">
                            <span class="metric-label">最差窗口长度:</span>
                            <span class="metric-value highlight">{t4['L']} 年</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">时间段:</span>
                            <span class="metric-value">{t4['worst_window_years'][0]}–{t4['worst_window_years'][1]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">累计收益率:</span>
                            <span class="metric-value">{t4['worst_window_cumret']*100:.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">年化收益率:</span>
                            <span class="metric-value">{t4['worst_window_cagr']*100:.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">测试窗口数:</span>
                            <span class="metric-value">{t4['windows_tested']:,}</span>
                        </div>
                    </div>
'''
            else:
                html += '<div class="task-result"><span class="no-data">无数据</span></div>'
            html += '</div>'
            
            html += '</div>\n'
        
        html += '''
            </div>
        </div>
'''
    
    # Add footer
    html += f'''
        <div style="text-align: center; color: white; margin-top: 30px; padding: 20px;">
            <p style="opacity: 0.9;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="opacity: 0.7; font-size: 0.9em; margin-top: 10px;">
                本报告分析历史收益率，执行四项任务：避免损失的最佳期限（任务1）、稳定收益期限（任务2）、最佳投资窗口（任务3）和最差投资窗口（任务4）。
            </p>
        </div>
    </div>
</body>
</html>
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ HTML报告已生成: {output_file}")

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Analyze index/stock annual returns by historical baselines.")
    ap.add_argument("--data-source", choices=['sp500', 'nasdaq100', 'brk', 'btc', 'all'], default='all',
                    help="Data source to analyze: sp500 (S&P 500), nasdaq100 (NASDAQ-100), brk (Berkshire Hathaway), btc (Bitcoin), or all (default: all)")
    ap.add_argument("--csv-url", default=None,
                    help="CSV URL for remote data (optional, S&P 500 only)")
    ap.add_argument("--baselines", type=int, nargs="+", default=None,
                    help="Baselines to analyze (default varies by data source)")
    ap.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS,
                    help="CAGR dispersion thresholds (decimal, default: %(default)s)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Tolerance for comparisons (default: %(default)s)")
    args = ap.parse_args()
    
    # Determine which data sources to analyze
    if args.data_source == 'all':
        data_sources = ['sp500', 'nasdaq100', 'brk', 'btc']
    else:
        data_sources = [args.data_source]
    
    # Collect all results for HTML report
    all_results = {}
    
    # Process each data source
    for data_source in data_sources:
        # Set default baselines based on data source
        if args.baselines is None:
            if data_source == 'sp500':
                baselines = [1926, 1957, 1972, 1984]
            elif data_source == 'nasdaq100':
                baselines = [1986, 1995, 2000, 2010]  # NASDAQ-100 started in 1985
            elif data_source == 'brk':
                baselines = [1981, 1990, 2000, 2010]  # Berkshire data starts from 1981
            elif data_source == 'btc':
                baselines = [2014, 2017, 2020, 2022]  # Bitcoin data starts from 2014
        else:
            baselines = args.baselines
        
        # Set default CSV URL based on data source if needed
        csv_url = args.csv_url
        if csv_url is None and data_source == 'sp500':
            csv_url = DEFAULT_CSV_URL

        # Load data
        df, label = _load_data(csv_url, data_source)

        # No need to drop current year for headless CSV format

        # Sort ascending and compute decimal/logs
        df = df.sort_values("Year").reset_index(drop=True)
        
        # Handle different column names from different data sources
        if label not in df.columns:
            # Try to find a return column
            for col in df.columns:
                if 'return' in col.lower():
                    label = col
                    break
            else:
                raise RuntimeError(f"Expected column '{label}' not present after load.")
        
        # r_y in decimal
        df["r"] = df[label] / 100.0

        years = df["Year"]
        # Print header with source note
        source_names = {
            'sp500': 'S&P 500',
            'nasdaq100': 'NASDAQ-100',
            'brk': 'Berkshire Hathaway',
            'btc': 'Bitcoin (BTC)'
        }
        source_name = source_names.get(data_source, data_source.upper())
        print(f"\n{'='*60}")
        print(f"{source_name} 年度收益率 — CSV数据分析 (名义值)")
        print(f"数据来源: {DATA_SOURCES.get(data_source, 'CSV文件')}")
        print('='*60)
        
        # Collect results for this data source
        source_results = []

        for b in baselines:
            start, end = first_contiguous_span(years, b)
            if start is None:
                print(f"\n基准年份 {b}: 没有可用数据 >= {b}。")
                continue
            sub = df[(df["Year"] >= start) & (df["Year"] <= end)].reset_index(drop=True)
            t1 = task1_no_loss_horizon(sub["Year"], sub["r"], tol=args.tol)
            t2s = [task2_cagr_dispersion(sub["Year"], sub["r"], tau, tol=args.tol) for tau in args.taus]
            t3 = task3_best_window(sub["Year"], sub["r"])
            t4 = task4_worst_window(sub["Year"], sub["r"])
            print_report(b, (start, end), label, t1, t2s, t3, t4)
            source_results.append(collect_results(b, (start, end), label, t1, t2s, t3, t4))
        
        all_results[data_source] = source_results
    
    # Generate HTML report
    generate_html_report(all_results)

if __name__ == "__main__":
    main()
