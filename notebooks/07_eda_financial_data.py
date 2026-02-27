"""
Exploratory Data Analysis (EDA) cho dữ liệu tài chính
- Phân tích xu hướng revenue, profit
- So sánh giữa các công ty
- Phân tích tăng trưởng theo thời gian
- Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib cho tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# ===== CẤU HÌNH ĐƯỜNG DẪN =====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_data_path = os.path.join(project_root, 'data', 'processed')
figures_path = os.path.join(project_root, 'reports', 'figures')

# Tạo folder figures nếu chưa có
os.makedirs(figures_path, exist_ok=True)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - FINANCIAL DATA")
print("=" * 60)
print(f"📁 Processed data: {processed_data_path}")
print(f"📁 Figures: {figures_path}\n")

# ===== 1. ĐỌC DỮ LIỆU CLEAN =====
print("=" * 60)
print("1. ĐANG ĐỌC DỮ LIỆU CLEAN...")
print("=" * 60)

income_df = pd.read_csv(os.path.join(processed_data_path, 'income_statement_clean.csv'))
balance_df = pd.read_csv(os.path.join(processed_data_path, 'balance_sheet_clean.csv'))
cashflow_df = pd.read_csv(os.path.join(processed_data_path, 'cash_flow_clean.csv'))

# Chuyển đổi date sang datetime
income_df['date'] = pd.to_datetime(income_df['date'])
balance_df['date'] = pd.to_datetime(balance_df['date'])
cashflow_df['date'] = pd.to_datetime(cashflow_df['date'])

print(f"✅ Income Statement: {income_df.shape}")
print(f"✅ Balance Sheet: {balance_df.shape}")
print(f"✅ Cash Flow: {cashflow_df.shape}")

# ===== 2. THỐNG KÊ MÔ TẢ =====
print("\n" + "=" * 60)
print("2. THỐNG KÊ MÔ TẢ")
print("=" * 60)

print("\n📊 Income Statement - Revenue Statistics:")
print(income_df.groupby('ticker')['Revenue (Bn. VND)'].describe())

print("\n📊 Data Points per Ticker:")
data_points = income_df.groupby('ticker').size().sort_values(ascending=False)
print(data_points)

print("\n📊 Date Range per Ticker:")
date_range = income_df.groupby('ticker')['date'].agg(['min', 'max'])
print(date_range)

# ===== 3. PHÂN TÍCH REVENUE THEO THỜI GIAN =====
print("\n" + "=" * 60)
print("3. PHÂN TÍCH REVENUE THEO THỜI GIAN")
print("=" * 60)

# Lọc data có revenue
income_with_revenue = income_df[income_df['Revenue (Bn. VND)'].notna()].copy()

# Chuyển revenue sang tỷ VND (dễ đọc hơn)
income_with_revenue['Revenue (Trillion VND)'] = income_with_revenue['Revenue (Bn. VND)'] / 1e12

print(f"\n✅ Records có revenue: {len(income_with_revenue)}/{len(income_df)}")

# Plot 1: Revenue theo thời gian (tất cả tickers)
plt.figure(figsize=(14, 8))
for ticker in income_with_revenue['ticker'].unique():
    ticker_data = income_with_revenue[income_with_revenue['ticker'] == ticker]
    plt.plot(ticker_data['date'], ticker_data['Revenue (Trillion VND)'], 
             marker='o', label=ticker, linewidth=2, markersize=4)

plt.title('Revenue Trends - All Tickers (2007-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue (Trillion VND)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, '01_revenue_trends_all.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: 01_revenue_trends_all.png")
plt.close()

# Plot 2: Revenue theo thời gian (từng ticker riêng)
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for idx, ticker in enumerate(sorted(income_with_revenue['ticker'].unique())):
    ticker_data = income_with_revenue[income_with_revenue['ticker'] == ticker]
    
    axes[idx].plot(ticker_data['date'], ticker_data['Revenue (Trillion VND)'], 
                   marker='o', color='steelblue', linewidth=2, markersize=4)
    axes[idx].set_title(f'{ticker} - Revenue Trend', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Date', fontsize=10)
    axes[idx].set_ylabel('Revenue (Trillion VND)', fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, '02_revenue_trends_individual.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: 02_revenue_trends_individual.png")
plt.close()

# ===== 4. SO SÁNH REVENUE GIỮA CÁC CÔNG TY =====
print("\n" + "=" * 60)
print("4. SO SÁNH REVENUE GIỮA CÁC CÔNG TY")
print("=" * 60)

# Lấy revenue trung bình của mỗi ticker
avg_revenue = income_with_revenue.groupby('ticker')['Revenue (Trillion VND)'].mean().sort_values(ascending=False)
print("\n📊 Average Revenue (Trillion VND):")
print(avg_revenue)

# Plot 3: Bar chart so sánh average revenue
plt.figure(figsize=(12, 6))
avg_revenue.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Average Revenue by Ticker (2007-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Ticker', fontsize=12)
plt.ylabel('Average Revenue (Trillion VND)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, '03_avg_revenue_comparison.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: 03_avg_revenue_comparison.png")
plt.close()

# ===== 5. PHÂN TÍCH TĂNG TRƯỞNG YoY =====
print("\n" + "=" * 60)
print("5. PHÂN TÍCH TĂNG TRƯỞNG YoY")
print("=" * 60)

# Lọc data có YoY
income_with_yoy = income_df[income_df['Revenue YoY (%)'].notna()].copy()

print(f"\n✅ Records có YoY: {len(income_with_yoy)}/{len(income_df)}")

if len(income_with_yoy) > 0:
    print("\n📊 YoY Growth Statistics:")
    print(income_with_yoy.groupby('ticker')['Revenue YoY (%)'].describe())
    
    # Plot 4: Box plot YoY growth
    plt.figure(figsize=(12, 6))
    income_with_yoy.boxplot(column='Revenue YoY (%)', by='ticker', figsize=(12, 6))
    plt.title('Revenue YoY Growth Distribution by Ticker', fontsize=16, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Ticker', fontsize=12)
    plt.ylabel('YoY Growth (%)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '04_yoy_growth_boxplot.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved: 04_yoy_growth_boxplot.png")
    plt.close()

# ===== 6. PHÂN TÍCH BALANCE SHEET =====
print("\n" + "=" * 60)
print("6. PHÂN TÍCH BALANCE SHEET")
print("=" * 60)

# Tìm cột Total Assets
asset_cols = [col for col in balance_df.columns if 'total' in col.lower() and 'asset' in col.lower()]
print(f"\n📊 Total Asset columns: {asset_cols}")

if len(asset_cols) > 0:
    asset_col = asset_cols[0]
    balance_with_assets = balance_df[balance_df[asset_col].notna()].copy()
    balance_with_assets['Total Assets (Trillion VND)'] = balance_with_assets[asset_col] / 1e12
    
    print(f"\n✅ Records có Total Assets: {len(balance_with_assets)}/{len(balance_df)}")
    
    # Plot 5: Total Assets theo thời gian
    plt.figure(figsize=(14, 8))
    for ticker in balance_with_assets['ticker'].unique():
        ticker_data = balance_with_assets[balance_with_assets['ticker'] == ticker]
        plt.plot(ticker_data['date'], ticker_data['Total Assets (Trillion VND)'], 
                 marker='o', label=ticker, linewidth=2, markersize=4)
    
    plt.title('Total Assets Trends - All Tickers (2007-2025)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Assets (Trillion VND)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '05_total_assets_trends.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved: 05_total_assets_trends.png")
    plt.close()

# ===== 7. CORRELATION ANALYSIS =====
print("\n" + "=" * 60)
print("7. CORRELATION ANALYSIS")
print("=" * 60)

# Chọn các cột số quan trọng từ income statement
numeric_cols = ['Revenue (Bn. VND)', 'Revenue YoY (%)', 
                'Attribute to parent company (Bn. VND)', 
                'Attribute to parent company YoY (%)']

# Lọc các cột tồn tại
existing_cols = [col for col in numeric_cols if col in income_df.columns]

if len(existing_cols) > 1:
    corr_data = income_df[existing_cols].corr()
    
    print("\n📊 Correlation Matrix:")
    print(corr_data)
    
    # Plot 6: Heatmap correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Income Statement', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, '06_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved: 06_correlation_heatmap.png")
    plt.close()

# ===== 8. SUMMARY REPORT =====
print("\n" + "=" * 60)
print("8. SUMMARY REPORT")
print("=" * 60)

summary = {
    'Metric': [
        'Total Records',
        'Tickers',
        'Date Range',
        'Avg Revenue (Trillion VND)',
        'Max Revenue (Trillion VND)',
        'Avg YoY Growth (%)'
    ],
    'Value': [
        len(income_df),
        len(income_df['ticker'].unique()),
        f"{income_df['date'].min()} to {income_df['date'].max()}",
        f"{income_with_revenue['Revenue (Trillion VND)'].mean():.2f}",
        f"{income_with_revenue['Revenue (Trillion VND)'].max():.2f}",
        f"{income_with_yoy['Revenue YoY (%)'].mean():.2f}" if len(income_with_yoy) > 0 else 'N/A'
    ]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Lưu summary
summary_file = os.path.join(processed_data_path, 'eda_summary.csv')
summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"\n✅ Summary saved to: {summary_file}")

print("\n" + "=" * 60)
print("🎉 HOÀN THÀNH EDA!")
print("=" * 60)
print(f"\n📁 Figures saved to: {figures_path}")
print(f"📊 Total figures: {len([f for f in os.listdir(figures_path) if f.endswith('.png')])}")
print("\n✅ Sẵn sàng cho bước modeling tiếp theo!")
