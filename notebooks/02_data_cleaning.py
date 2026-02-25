"""
=============================================================================
DATA CLEANING & PREPROCESSING
=============================================================================
Purpose: Làm sạch và chuẩn bị data cho phân tích
Input: ../data/raw/financial_data_raw.csv
Output: ../data/processed/financial_data_clean.csv
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataCleaner:
    def __init__(self, input_file):
        print("\n" + "="*70)
        print("🧹 DATA CLEANING & PREPROCESSING")
        print("="*70)
        print(f"📁 Input: {input_file}")
        
        self.df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        print(f"📊 Original data: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        print("="*70 + "\n")
    
    def step1_basic_info(self):
        """Bước 1: Kiểm tra thông tin cơ bản"""
        print("📋 BƯỚC 1: THÔNG TIN CƠ BẢN")
        print("-" * 70)
        
        print(f"✅ Shape: {self.df.shape}")
        print(f"✅ Columns: {self.df.columns.tolist()[:10]}... (showing first 10)")
        print(f"✅ Data types:\n{self.df.dtypes.value_counts()}")
        
        print(f"\n📊 Missing values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.head(10))
        else:
            print("   ✅ Không có missing values!")
        
        print("\n✅ Bước 1 hoàn thành!\n")
    
    def step2_handle_missing(self):
        """Bước 2: Xử lý missing values"""
        print("📋 BƯỚC 2: XỬ LÝ MISSING VALUES")
        print("-" * 70)
        
        before = self.df.shape
        
        # Xóa columns có quá nhiều missing (>80%)
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        cols_to_drop = missing_pct[missing_pct > 80].index.tolist()
        
        if cols_to_drop:
            print(f"🗑️  Xóa {len(cols_to_drop)} columns có >80% missing:")
            for col in cols_to_drop[:5]:
                print(f"   • {col}")
            if len(cols_to_drop) > 5:
                print(f"   ... và {len(cols_to_drop) - 5} columns khác")
            
            self.df = self.df.drop(columns=cols_to_drop)
        
        # Fill missing cho numeric columns với 0
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        # Fill missing cho text columns với 'Unknown'
        text_cols = self.df.select_dtypes(include=['object']).columns
        self.df[text_cols] = self.df[text_cols].fillna('Unknown')
        
        after = self.df.shape
        
        print(f"✅ Before: {before[0]:,} rows × {before[1]:,} columns")
        print(f"✅ After:  {after[0]:,} rows × {after[1]:,} columns")
        print(f"✅ Dropped: {before[1] - after[1]} columns")
        print("\n✅ Bước 2 hoàn thành!\n")
    
    def step3_data_types(self):
        """Bước 3: Chuyển đổi data types"""
        print("📋 BƯỚC 3: CHUYỂN ĐỔI DATA TYPES")
        print("-" * 70)
        
        # Convert yearReport và lengthReport
        if 'yearReport' in self.df.columns:
            self.df['yearReport'] = pd.to_numeric(self.df['yearReport'], errors='coerce')
            self.df['year'] = self.df['yearReport'].astype('Int64')
            print("✅ Converted yearReport → year (integer)")
        
        if 'lengthReport' in self.df.columns:
            self.df['lengthReport'] = pd.to_numeric(self.df['lengthReport'], errors='coerce')
            self.df['quarter'] = self.df['lengthReport'].astype('Int64')
            print("✅ Converted lengthReport → quarter (integer)")
        
        # Convert scraped_at to datetime
        if 'scraped_at' in self.df.columns:
            self.df['scraped_at'] = pd.to_datetime(self.df['scraped_at'], errors='coerce')
            print("✅ Converted scraped_at → datetime")
        
        # Convert all numeric-looking columns
        for col in self.df.columns:
            if col not in ['stock_code', 'report_type', 'scraped_at', 'year', 'quarter']:
                if self.df[col].dtype == 'object':
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    except:
                        pass
        
        print(f"\n✅ Final data types:")
        print(self.df.dtypes.value_counts())
        print("\n✅ Bước 3 hoàn thành!\n")
    
    def step4_create_features(self):
        """Bước 4: Tạo features mới"""
        print("📋 BƯỚC 4: TẠO FEATURES MỚI")
        print("-" * 70)
        
        # Tạo period (YYYY-QQ)
        if 'year' in self.df.columns and 'quarter' in self.df.columns:
            self.df['period'] = self.df['year'].astype(str) + '-Q' + self.df['quarter'].astype(str)
            print("✅ Created: period (YYYY-QQ)")
        
        # Tạo company_report (unique identifier)
        if 'stock_code' in self.df.columns and 'report_type' in self.df.columns:
            self.df['company_report'] = self.df['stock_code'] + '_' + self.df['report_type'].str.replace(' ', '_')
            print("✅ Created: company_report (unique identifier)")
        
        print("\n✅ Bước 4 hoàn thành!\n")
    
    def step5_remove_duplicates(self):
        """Bước 5: Xóa duplicates"""
        print("📋 BƯỚC 5: XÓA DUPLICATES")
        print("-" * 70)
        
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        
        print(f"✅ Before: {before:,} rows")
        print(f"✅ After:  {after:,} rows")
        print(f"✅ Removed: {before - after:,} duplicates")
        print("\n✅ Bước 5 hoàn thành!\n")
    
    def step6_sort_data(self):
        """Bước 6: Sắp xếp data"""
        print("📋 BƯỚC 6: SẮP XẾP DATA")
        print("-" * 70)
        
        sort_cols = []
        if 'stock_code' in self.df.columns:
            sort_cols.append('stock_code')
        if 'report_type' in self.df.columns:
            sort_cols.append('report_type')
        if 'year' in self.df.columns:
            sort_cols.append('year')
        if 'quarter' in self.df.columns:
            sort_cols.append('quarter')
        
        if sort_cols:
            self.df = self.df.sort_values(sort_cols, ascending=[True, True, False, False])
            print(f"✅ Sorted by: {', '.join(sort_cols)}")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        print("✅ Reset index")
        
        print("\n✅ Bước 6 hoàn thành!\n")
    
    def step7_summary(self):
        """Bước 7: Tóm tắt kết quả"""
        print("📋 BƯỚC 7: TÓM TẮT KẾT QUẢ")
        print("-" * 70)
        
        print(f"📊 Final shape: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        
        if 'stock_code' in self.df.columns:
            print(f"\n🏢 Công ty ({len(self.df['stock_code'].unique())}):")
            for code in sorted(self.df['stock_code'].unique()):
                count = len(self.df[self.df['stock_code'] == code])
                print(f"   • {code}: {count:,} records")
        
        if 'report_type' in self.df.columns:
            print(f"\n📋 Loại báo cáo ({len(self.df['report_type'].unique())}):")
            for report in sorted(self.df['report_type'].unique()):
                count = len(self.df[self.df['report_type'] == report])
                print(f"   • {report}: {count:,} records")
        
        if 'year' in self.df.columns:
            print(f"\n📅 Năm: {self.df['year'].min()} - {self.df['year'].max()}")
        
        if 'quarter' in self.df.columns:
            print(f"📅 Quý: {sorted(self.df['quarter'].unique())}")
        
        print("\n✅ Bước 7 hoàn thành!\n")
    
    def save_clean_data(self, output_file):
        """Lưu data đã clean"""
        print("💾 LƯU FILE")
        print("-" * 70)
        
        # Tạo folder
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Lưu CSV
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ Đã lưu: {output_file}")
        print(f"✅ Size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        # Preview
        print(f"\n📋 PREVIEW (5 dòng đầu):")
        preview_cols = ['stock_code', 'report_type', 'year', 'quarter', 'period']
        available_cols = [col for col in preview_cols if col in self.df.columns]
        if available_cols:
            print(self.df[available_cols].head().to_string(index=False))
        
        return self.df

def main():
    print("\n" + "="*70)
    print("💼 FINANCIAL DATA CLEANING PIPELINE")
    print("="*70)
    
    # Input/Output files
    input_file = '../data/raw/financial_data_raw.csv'
    output_file = '../data/processed/financial_data_clean.csv'
    
    # Khởi tạo cleaner
    cleaner = DataCleaner(input_file)
    
    # Chạy pipeline
    cleaner.step1_basic_info()
    cleaner.step2_handle_missing()
    cleaner.step3_data_types()
    cleaner.step4_create_features()
    cleaner.step5_remove_duplicates()
    cleaner.step6_sort_data()
    cleaner.step7_summary()
    
    # Lưu file
    df_clean = cleaner.save_clean_data(output_file)
    
    print("\n" + "="*70)
    print("✅ DATA CLEANING HOÀN THÀNH!")
    print("="*70)
    print("\n➡️  BƯỚC TIẾP THEO:")
    print("   python 03_exploratory_analysis.py")
    print("="*70)

if __name__ == "__main__":
    main()
