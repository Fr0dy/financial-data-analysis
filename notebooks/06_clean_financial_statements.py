"""
Làm sạch và chuẩn hóa dữ liệu báo cáo tài chính
- Xử lý missing values
- Chuẩn hóa tên cột
- Chuyển đổi kiểu dữ liệu
- Tạo features mới
"""

import pandas as pd
import numpy as np
import os

# ===== CẤU HÌNH ĐƯỜNG DẪN =====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
raw_data_path = os.path.join(project_root, 'data', 'raw')
processed_data_path = os.path.join(project_root, 'data', 'processed')

# Tạo folder processed nếu chưa có
os.makedirs(processed_data_path, exist_ok=True)

print("=" * 60)
print("CLEANING FINANCIAL DATA")
print("=" * 60)
print(f"📁 Raw data: {raw_data_path}")
print(f"📁 Processed data: {processed_data_path}\n")

# ===== 1. ĐỌC DỮ LIỆU RAW =====
print("=" * 60)
print("1. ĐANG ĐỌC DỮ LIỆU RAW...")
print("=" * 60)

# Đọc Income Statement
income_df = pd.read_csv(os.path.join(raw_data_path, 'income_statement_raw.csv'))
print(f"✅ Income Statement: {income_df.shape}")
print(f"   Columns: {len(income_df.columns)}")
print(f"   Tickers: {income_df['ticker'].unique().tolist()}")

# Đọc Balance Sheet
balance_df = pd.read_csv(os.path.join(raw_data_path, 'balance_sheet_raw.csv'))
print(f"✅ Balance Sheet: {balance_df.shape}")
print(f"   Columns: {len(balance_df.columns)}")

# Đọc Cash Flow
cashflow_df = pd.read_csv(os.path.join(raw_data_path, 'cash_flow_raw.csv'))
print(f"✅ Cash Flow: {cashflow_df.shape}")
print(f"   Columns: {len(cashflow_df.columns)}")

# ===== 2. KIỂM TRA MISSING VALUES =====
print("\n" + "=" * 60)
print("2. KIỂM TRA MISSING VALUES...")
print("=" * 60)

def check_missing(df, name):
    print(f"\n📊 {name}:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print(f"   ⚠️ Có {len(missing_df)} cột có missing values")
        print(missing_df.head(10).to_string(index=False))
    else:
        print("   ✅ Không có missing values!")
    
    return missing_df

income_missing = check_missing(income_df, "Income Statement")
balance_missing = check_missing(balance_df, "Balance Sheet")
cashflow_missing = check_missing(cashflow_df, "Cash Flow")

# ===== 3. CHUẨN HÓA DỮ LIỆU =====
print("\n" + "=" * 60)
print("3. CHUẨN HÓA DỮ LIỆU...")
print("=" * 60)

def clean_financial_data(df, name):
    print(f"\n📊 Đang xử lý {name}...")
    
    df_clean = df.copy()
    
    # 3.1. Chuẩn hóa tên cột (loại bỏ khoảng trắng thừa)
    df_clean.columns = df_clean.columns.str.strip()
    
    # 3.2. Chuyển đổi yearReport và lengthReport thành int
    if 'yearReport' in df_clean.columns:
        df_clean['yearReport'] = pd.to_numeric(df_clean['yearReport'], errors='coerce').astype('Int64')
    
    if 'lengthReport' in df_clean.columns:
        df_clean['lengthReport'] = pd.to_numeric(df_clean['lengthReport'], errors='coerce').astype('Int64')
    
    # 3.3. Tạo cột quarter và year riêng biệt
    if 'yearReport' in df_clean.columns and 'lengthReport' in df_clean.columns:
        df_clean['year'] = df_clean['yearReport']
        df_clean['quarter'] = df_clean['lengthReport']
        
        # Tạo cột date (cuối quý)
        df_clean['date'] = pd.to_datetime(
            df_clean['year'].astype(str) + '-' + 
            (df_clean['quarter'] * 3).astype(str) + '-01'
        ) + pd.offsets.MonthEnd(0)
    
    # 3.4. Chuyển đổi các cột số (loại bỏ NaN)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['ticker', 'yearReport', 'lengthReport', 'year', 'quarter']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 3.5. Sắp xếp theo ticker, year, quarter
    if 'year' in df_clean.columns and 'quarter' in df_clean.columns:
        df_clean = df_clean.sort_values(['ticker', 'year', 'quarter'], ascending=[True, True, True])
    
    # 3.6. Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"   ✅ Shape sau khi clean: {df_clean.shape}")
    print(f"   ✅ Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    
    return df_clean

income_clean = clean_financial_data(income_df, "Income Statement")
balance_clean = clean_financial_data(balance_df, "Balance Sheet")
cashflow_clean = clean_financial_data(cashflow_df, "Cash Flow")

# ===== 4. TẠO FEATURES MỚI =====
print("\n" + "=" * 60)
print("4. TẠO FEATURES MỚI...")
print("=" * 60)

# 4.1. Income Statement - Tính tỷ lệ tăng trưởng
print("\n📊 Income Statement - Growth Rates...")

# Lấy các cột revenue quan trọng
revenue_cols = [col for col in income_clean.columns if 'revenue' in col.lower() and 'yoy' not in col.lower()]
print(f"   Revenue columns: {revenue_cols[:5]}...")

# 4.2. Balance Sheet - Tính các chỉ số tài chính
print("\n📊 Balance Sheet - Financial Ratios...")

# Tìm các cột tài sản và nợ
asset_cols = [col for col in balance_clean.columns if 'asset' in col.lower()]
liability_cols = [col for col in balance_clean.columns if 'liabilit' in col.lower()]
equity_cols = [col for col in balance_clean.columns if 'equity' in col.lower()]

print(f"   Asset columns: {len(asset_cols)}")
print(f"   Liability columns: {len(liability_cols)}")
print(f"   Equity columns: {len(equity_cols)}")

# 4.3. Cash Flow - Phân loại dòng tiền
print("\n📊 Cash Flow - Cash Flow Types...")

operating_cf = [col for col in cashflow_clean.columns if 'operating' in col.lower()]
investing_cf = [col for col in cashflow_clean.columns if 'investing' in col.lower()]
financing_cf = [col for col in cashflow_clean.columns if 'financing' in col.lower()]

print(f"   Operating CF columns: {len(operating_cf)}")
print(f"   Investing CF columns: {len(investing_cf)}")
print(f"   Financing CF columns: {len(financing_cf)}")

# ===== 5. LƯU DỮ LIỆU ĐÃ CLEAN =====
print("\n" + "=" * 60)
print("5. LƯU DỮ LIỆU ĐÃ CLEAN...")
print("=" * 60)

# Lưu Income Statement
income_file = os.path.join(processed_data_path, 'income_statement_clean.csv')
income_clean.to_csv(income_file, index=False, encoding='utf-8-sig')
print(f"✅ Income Statement: {income_file}")
print(f"   Shape: {income_clean.shape}")

# Lưu Balance Sheet
balance_file = os.path.join(processed_data_path, 'balance_sheet_clean.csv')
balance_clean.to_csv(balance_file, index=False, encoding='utf-8-sig')
print(f"✅ Balance Sheet: {balance_file}")
print(f"   Shape: {balance_clean.shape}")

# Lưu Cash Flow
cashflow_file = os.path.join(processed_data_path, 'cash_flow_clean.csv')
cashflow_clean.to_csv(cashflow_file, index=False, encoding='utf-8-sig')
print(f"✅ Cash Flow: {cashflow_file}")
print(f"   Shape: {cashflow_clean.shape}")

# ===== 6. TẠO SUMMARY REPORT =====
print("\n" + "=" * 60)
print("6. TẠO SUMMARY REPORT...")
print("=" * 60)

summary = {
    'Dataset': ['Income Statement', 'Balance Sheet', 'Cash Flow'],
    'Raw Shape': [income_df.shape, balance_df.shape, cashflow_df.shape],
    'Clean Shape': [income_clean.shape, balance_clean.shape, cashflow_clean.shape],
    'Tickers': [
        len(income_clean['ticker'].unique()),
        len(balance_clean['ticker'].unique()),
        len(cashflow_clean['ticker'].unique())
    ],
    'Date Range': [
        f"{income_clean['date'].min()} to {income_clean['date'].max()}",
        f"{balance_clean['date'].min()} to {balance_clean['date'].max()}",
        f"{cashflow_clean['date'].min()} to {cashflow_clean['date'].max()}"
    ]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Lưu summary
summary_file = os.path.join(processed_data_path, 'cleaning_summary.csv')
summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"\n✅ Summary saved to: {summary_file}")

# ===== 7. HIỂN THỊ SAMPLE DATA =====
print("\n" + "=" * 60)
print("7. SAMPLE DATA (5 ROWS)")
print("=" * 60)

print("\n📊 Income Statement:")
print(income_clean[['ticker', 'year', 'quarter', 'date'] + revenue_cols[:3]].head())

print("\n📊 Balance Sheet:")
print(balance_clean[['ticker', 'year', 'quarter', 'date'] + asset_cols[:3]].head())

print("\n📊 Cash Flow:")
print(cashflow_clean[['ticker', 'year', 'quarter', 'date'] + operating_cf[:3]].head())

print("\n" + "=" * 60)
print("🎉 HOÀN THÀNH CLEANING!")
print("=" * 60)
print(f"\n📁 Processed files saved to: {processed_data_path}")
print("\n✅ Sẵn sàng cho bước phân tích tiếp theo!")
