"""
=============================================================================
FINAL DATA COLLECTION WITH API KEY
=============================================================================
"""

import os
os.environ['VNSTOCK_API_KEY'] = 'vnstock_8fa2b72966470ff1ba206217b0515f25'

from vnstock import Vnstock
import pandas as pd
import time
from datetime import datetime

class VNStockScraper:
    def __init__(self):
        print("✅ Sử dụng API Key: vnstock_8fa2...5f25")
        self.delay = 3  # Delay 3 giây với API key
    
    def flatten_dataframe(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        df = df.reset_index()
        return df
    
    def get_financial_report(self, stock_code, period='quarter'):
        try:
            print(f"📊 {stock_code}...", end=" ", flush=True)
            
            stock = Vnstock().stock(symbol=stock_code, source='VCI')
            
            all_dfs = []
            
            report_types = [
                ('income_statement', 'Income Statement'),
                ('balance_sheet', 'Balance Sheet'),
                ('cash_flow', 'Cash Flow'),
                ('ratio', 'Financial Ratios')
            ]
            
            for method_name, report_name in report_types:
                try:
                    time.sleep(1.5)  # 1.5 giây giữa mỗi request
                    method = getattr(stock.finance, method_name)
                    df = method(period=period, lang='en', dropna=True)
                    
                    if not df.empty:
                        df = self.flatten_dataframe(df)
                        df['stock_code'] = stock_code
                        df['report_type'] = report_name
                        df['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        all_dfs.append(df)
                except Exception as e:
                    continue
            
            print(f"✅ {len(all_dfs)}/4 báo cáo")
            return all_dfs if all_dfs else None
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            return None
    
    def scrape_multiple_companies(self, stock_codes):
        print("\n" + "="*70)
        print("🚀 THU THẬP DATA VỚI API KEY")
        print("="*70)
        print(f"📋 Công ty: {', '.join(stock_codes)}")
        print(f"⏱️  Delay: {self.delay} giây/công ty")
        print(f"🔑 Rate limit: 60 requests/phút (Community)")
        print("="*70 + "\n")
        
        all_dataframes = []
        success_count = 0
        
        for i, code in enumerate(stock_codes, 1):
            print(f"[{i}/{len(stock_codes)}] ", end="")
            dfs = self.get_financial_report(code, period='quarter')
            
            if dfs:
                all_dataframes.extend(dfs)
                success_count += 1
            
            if i < len(stock_codes):
                print(f"⏳ Đợi {self.delay} giây...\n")
                time.sleep(self.delay)
        
        print("\n" + "="*70)
        print(f"✅ HOÀN THÀNH: {success_count}/{len(stock_codes)} công ty")
        print(f"📊 Tổng DataFrames: {len(all_dataframes)}")
        print("="*70)
        return all_dataframes

def main():
    print("\n" + "="*70)
    print("💼 VNSTOCK DATA SCRAPER - FINAL VERSION")
    print("="*70)
    
    scraper = VNStockScraper()
    
    # 6 công ty blue-chip
    companies = ['VNM', 'FPT', 'VCB', 'HPG', 'VHM', 'GAS']
    
    # Scrape data
    all_dataframes = scraper.scrape_multiple_companies(companies)
    
    if all_dataframes:
        # Gộp data
        try:
            df = pd.concat(all_dataframes, ignore_index=True, sort=False)
            
            # Tạo folder
            os.makedirs('../data/raw', exist_ok=True)
            
            # Lưu file
            output_file = '../data/raw/financial_data_raw.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"\n💾 ĐÃ LƯU FILE:")
            print(f"   📁 {output_file}")
            print(f"   📏 {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   💽 {os.path.getsize(output_file) / 1024:.2f} KB")
            
            # Thống kê
            if 'stock_code' in df.columns:
                print(f"\n📊 THỐNG KÊ:")
                print(f"   🏢 Công ty: {sorted(df['stock_code'].unique())}")
                print(f"\n   📋 Records/công ty:")
                for code in sorted(df['stock_code'].unique()):
                    count = len(df[df['stock_code'] == code])
                    print(f"      • {code}: {count:,} records")
                
                if 'report_type' in df.columns:
                    print(f"\n   📊 Loại báo cáo:")
                    for report in df['report_type'].unique():
                        count = len(df[df['report_type'] == report])
                        print(f"      • {report}: {count:,} records")
            
            # Preview
            print(f"\n📋 PREVIEW (5 dòng đầu):")
            preview_cols = ['stock_code', 'report_type']
            if 'yearReport' in df.columns:
                preview_cols.append('yearReport')
            if 'lengthReport' in df.columns:
                preview_cols.append('lengthReport')
            
            available_cols = [col for col in preview_cols if col in df.columns]
            if available_cols:
                print(df[available_cols].head().to_string(index=False))
            
            print("\n" + "="*70)
            print("✅ HOÀN THÀNH!")
            print("="*70)
            print("\n➡️  BƯỚC TIẾP THEO:")
            print("   python 02_data_cleaning.py")
            
        except Exception as e:
            print(f"\n❌ Lỗi khi gộp data: {e}")
            print("\n📦 Lưu từng file riêng...")
            
            for i, df in enumerate(all_dataframes, 1):
                try:
                    stock_code = df['stock_code'].iloc[0] if 'stock_code' in df.columns else f'stock_{i}'
                    report_type = df['report_type'].iloc[0] if 'report_type' in df.columns else f'report_{i}'
                    report_type = report_type.replace(' ', '_').lower()
                    
                    filename = f'../data/raw/financial_{stock_code}_{report_type}.csv'
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    print(f"   ✅ {filename}")
                except Exception as e2:
                    print(f"   ❌ Lỗi file {i}: {e2}")
    
    else:
        print("\n⚠️ KHÔNG LẤY ĐƯỢC DATA!")
        print("🔄 Chuyển sang data mẫu: python generate_realistic_data.py")

if __name__ == "__main__":
    main()
