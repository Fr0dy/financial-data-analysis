"""
Thu thập báo cáo tài chính từ VNStock với API Key
Quota: 60 requests/phút
"""

import pandas as pd
from vnstock import Vnstock
import time
import os

# ===== CẤU HÌNH API KEY =====
API_KEY = 'vnstock_8fa2b72966470ff1ba206217b0515f25'

# Set API key vào biến môi trường (vnstock sẽ tự động đọc)
os.environ['VNSTOCK_API_KEY'] = API_KEY

# ===== LẤY ĐƯỜNG DẪN GỐC =====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
raw_data_path = os.path.join(project_root, 'data', 'raw')
os.makedirs(raw_data_path, exist_ok=True)

print(f"📁 Raw data path: {raw_data_path}")
print(f"🔑 API Key: ✅ Đã cấu hình (60 requests/phút)")
print(f"🔑 Environment: {os.environ.get('VNSTOCK_API_KEY', 'Not set')[:20]}...")
print()

# Danh sách tickers
TICKERS = ['VHM', 'VIC', 'VNM', 'HPG', 'TCB', 'VCB', 'MBB', 'GAS', 'MSN', 'FPT']

# ===== 1. INCOME STATEMENT =====
print("=" * 60)
print("1. ĐANG CÀO INCOME STATEMENT (Báo cáo kết quả kinh doanh)...")
print("=" * 60)

income_statements = []

for ticker in TICKERS:
    print(f"\n📊 Đang cào {ticker}...")
    
    try:
        # Không cần truyền api_key, vnstock tự động đọc từ biến môi trường
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        income = stock.finance.income_statement(period='quarter', lang='en')
        
        if income is not None and not income.empty:
            income['ticker'] = ticker
            income_statements.append(income)
            print(f"   ✅ {ticker}: {len(income)} records")
            print(f"   📋 Columns: {income.columns.tolist()[:5]}...")
        else:
            print(f"   ⚠️ {ticker}: Không có data")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ {ticker}: Lỗi - {e}")

# Lưu Income Statement
if income_statements:
    income_df = pd.concat(income_statements, ignore_index=True)
    income_file = os.path.join(raw_data_path, 'income_statement_raw.csv')
    
    print(f"\n💾 Đang lưu vào: {income_file}")
    
    try:
        income_df.to_csv(income_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu {len(income_df)} records vào income_statement_raw.csv")
        print(f"📊 Shape: {income_df.shape}")
        print(f"📋 Columns: {income_df.columns.tolist()[:10]}...")
        
        print("\n📊 SAMPLE DATA (5 rows):")
        print(income_df.head())
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
else:
    print("\n❌ Không có Income Statement data!")

# ===== 2. BALANCE SHEET =====
print("\n" + "=" * 60)
print("2. ĐANG CÀO BALANCE SHEET (Bảng cân đối kế toán)...")
print("=" * 60)

balance_sheets = []

for ticker in TICKERS:
    print(f"\n📊 Đang cào {ticker}...")
    
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        balance = stock.finance.balance_sheet(period='quarter', lang='en')
        
        if balance is not None and not balance.empty:
            balance['ticker'] = ticker
            balance_sheets.append(balance)
            print(f"   ✅ {ticker}: {len(balance)} records")
        else:
            print(f"   ⚠️ {ticker}: Không có data")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ {ticker}: Lỗi - {e}")

# Lưu Balance Sheet
if balance_sheets:
    balance_df = pd.concat(balance_sheets, ignore_index=True)
    balance_file = os.path.join(raw_data_path, 'balance_sheet_raw.csv')
    
    print(f"\n💾 Đang lưu vào: {balance_file}")
    
    try:
        balance_df.to_csv(balance_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu {len(balance_df)} records vào balance_sheet_raw.csv")
        print(f"📊 Shape: {balance_df.shape}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
else:
    print("\n❌ Không có Balance Sheet data!")

# ===== 3. CASH FLOW =====
print("\n" + "=" * 60)
print("3. ĐANG CÀO CASH FLOW STATEMENT (Báo cáo lưu chuyển tiền tệ)...")
print("=" * 60)

cash_flows = []

for ticker in TICKERS:
    print(f"\n📊 Đang cào {ticker}...")
    
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI')
        cashflow = stock.finance.cash_flow(period='quarter', lang='en')
        
        if cashflow is not None and not cashflow.empty:
            cashflow['ticker'] = ticker
            cash_flows.append(cashflow)
            print(f"   ✅ {ticker}: {len(cashflow)} records")
        else:
            print(f"   ⚠️ {ticker}: Không có data")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ {ticker}: Lỗi - {e}")

# Lưu Cash Flow
if cash_flows:
    cashflow_df = pd.concat(cash_flows, ignore_index=True)
    cashflow_file = os.path.join(raw_data_path, 'cash_flow_raw.csv')
    
    print(f"\n💾 Đang lưu vào: {cashflow_file}")
    
    try:
        cashflow_df.to_csv(cashflow_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu {len(cashflow_df)} records vào cash_flow_raw.csv")
        print(f"📊 Shape: {cashflow_df.shape}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
else:
    print("\n❌ Không có Cash Flow data!")

# ===== SUMMARY =====
print("\n" + "=" * 60)
print("📊 TỔNG KẾT")
print("=" * 60)

if income_statements:
    print(f"✅ Income Statement: {len(income_df)} rows, {len(income_df.columns)} columns")
if balance_sheets:
    print(f"✅ Balance Sheet: {len(balance_df)} rows, {len(balance_df.columns)} columns")
if cash_flows:
    print(f"✅ Cash Flow: {len(cashflow_df)} rows, {len(cashflow_df.columns)} columns")

print(f"\n📁 Files saved to: {raw_data_path}")
print("\n🎉 HOÀN THÀNH!")

# Kiểm tra files
print("\n📋 ALL FILES:")
for filename in os.listdir(raw_data_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(raw_data_path, filename)
        size = os.path.getsize(filepath) / 1024
        print(f"   ✅ {filename} ({size:.2f} KB)")
