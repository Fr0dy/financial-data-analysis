"""
=============================================================================
EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
Purpose: Phân tích khám phá dữ liệu tài chính
Input: ../data/processed/financial_data_clean.csv
Output: 
  - ../reports/eda_report.txt
  - ../reports/figures/ (các biểu đồ)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FinancialEDA:
    def __init__(self, input_file):
        print("\n" + "="*70)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        print(f"📁 Input: {input_file}")
        
        self.df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        print(f"📊 Data: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        print("="*70 + "\n")
        
        # Tạo folder cho figures
        self.figures_dir = '../reports/figures'
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Report text
        self.report = []
    
    def add_to_report(self, text):
        """Thêm text vào report"""
        self.report.append(text)
        print(text)
    
    def section1_overview(self):
        """Phần 1: Tổng quan"""
        self.add_to_report("\n" + "="*70)
        self.add_to_report("📋 PHẦN 1: TỔNG QUAN DỮ LIỆU")
        self.add_to_report("="*70)
        
        self.add_to_report(f"\n📊 Kích thước: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        
        if 'stock_code' in self.df.columns:
            companies = sorted(self.df['stock_code'].unique())
            self.add_to_report(f"\n🏢 Công ty ({len(companies)}):")
            for code in companies:
                count = len(self.df[self.df['stock_code'] == code])
                self.add_to_report(f"   • {code}: {count:,} records")
        
        if 'report_type' in self.df.columns:
            reports = sorted(self.df['report_type'].unique())
            self.add_to_report(f"\n📋 Loại báo cáo ({len(reports)}):")
            for report in reports:
                count = len(self.df[self.df['report_type'] == report])
                self.add_to_report(f"   • {report}: {count:,} records")
        
        if 'period' in self.df.columns:
            periods = sorted(self.df['period'].unique())
            self.add_to_report(f"\n📅 Kỳ báo cáo ({len(periods)}):")
            self.add_to_report(f"   • Từ: {periods[0]}")
            self.add_to_report(f"   • Đến: {periods[-1]}")
        
        self.add_to_report("\n✅ Phần 1 hoàn thành!\n")
    
    def section2_revenue_analysis(self):
        """Phần 2: Phân tích doanh thu"""
        self.add_to_report("\n" + "="*70)
        self.add_to_report("📋 PHẦN 2: PHÂN TÍCH DOANH THU")
        self.add_to_report("="*70)
        
        # Tìm cột revenue
        revenue_cols = [col for col in self.df.columns if 'revenue' in col.lower() and 'yoy' not in col.lower()]
        
        if revenue_cols:
            revenue_col = revenue_cols[0]
            self.add_to_report(f"\n💰 Cột doanh thu: {revenue_col}")
            
            # Lọc Income Statement
            if 'report_type' in self.df.columns:
                df_income = self.df[self.df['report_type'] == 'Income Statement'].copy()
                
                if revenue_col in df_income.columns and 'stock_code' in df_income.columns:
                    # Doanh thu trung bình theo công ty
                    revenue_by_company = df_income.groupby('stock_code')[revenue_col].agg(['mean', 'sum', 'count'])
                    revenue_by_company = revenue_by_company.sort_values('mean', ascending=False)
                    
                    self.add_to_report(f"\n📊 Doanh thu trung bình theo công ty (Tỷ VNĐ):")
                    for idx, row in revenue_by_company.iterrows():
                        self.add_to_report(f"   • {idx}: {row['mean']:,.2f} (TB), {row['sum']:,.2f} (Tổng), {int(row['count'])} kỳ")
                    
                    # Biểu đồ
                    plt.figure(figsize=(12, 6))
                    revenue_by_company['mean'].plot(kind='bar', color='skyblue', edgecolor='black')
                    plt.title('Doanh Thu Trung Bình Theo Công Ty', fontsize=16, fontweight='bold')
                    plt.xlabel('Công ty', fontsize=12)
                    plt.ylabel('Doanh thu (Tỷ VNĐ)', fontsize=12)
                    plt.xticks(rotation=0)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    output_file = os.path.join(self.figures_dir, '01_revenue_by_company.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.add_to_report(f"\n📊 Đã lưu biểu đồ: {output_file}")
        
        self.add_to_report("\n✅ Phần 2 hoàn thành!\n")
    
    def section3_profitability_analysis(self):
        """Phần 3: Phân tích lợi nhuận"""
        self.add_to_report("\n" + "="*70)
        self.add_to_report("📋 PHẦN 3: PHÂN TÍCH LỢI NHUẬN")
        self.add_to_report("="*70)
        
        # Tìm cột profit
        profit_cols = [col for col in self.df.columns if 'profit' in col.lower() or 'parent company' in col.lower()]
        
        if profit_cols:
            profit_col = profit_cols[0]
            self.add_to_report(f"\n💵 Cột lợi nhuận: {profit_col}")
            
            if 'report_type' in self.df.columns:
                df_income = self.df[self.df['report_type'] == 'Income Statement'].copy()
                
                if profit_col in df_income.columns and 'stock_code' in df_income.columns:
                    # Lợi nhuận theo công ty
                    profit_by_company = df_income.groupby('stock_code')[profit_col].agg(['mean', 'sum'])
                    profit_by_company = profit_by_company.sort_values('mean', ascending=False)
                    
                    self.add_to_report(f"\n📊 Lợi nhuận trung bình theo công ty (Tỷ VNĐ):")
                    for idx, row in profit_by_company.iterrows():
                        self.add_to_report(f"   • {idx}: {row['mean']:,.2f} (TB), {row['sum']:,.2f} (Tổng)")
                    
                    # Biểu đồ
                    plt.figure(figsize=(12, 6))
                    profit_by_company['mean'].plot(kind='bar', color='lightgreen', edgecolor='black')
                    plt.title('Lợi Nhuận Trung Bình Theo Công Ty', fontsize=16, fontweight='bold')
                    plt.xlabel('Công ty', fontsize=12)
                    plt.ylabel('Lợi nhuận (Tỷ VNĐ)', fontsize=12)
                    plt.xticks(rotation=0)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    output_file = os.path.join(self.figures_dir, '02_profit_by_company.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.add_to_report(f"\n📊 Đã lưu biểu đồ: {output_file}")
        
        self.add_to_report("\n✅ Phần 3 hoàn thành!\n")
    
    def section4_financial_ratios(self):
        """Phần 4: Phân tích chỉ số tài chính"""
        self.add_to_report("\n" + "="*70)
        self.add_to_report("📋 PHẦN 4: PHÂN TÍCH CHỈ SỐ TÀI CHÍNH")
        self.add_to_report("="*70)
        
        if 'report_type' in self.df.columns:
            df_ratios = self.df[self.df['report_type'] == 'Financial Ratios'].copy()
            
            # Tìm các chỉ số quan trọng
            key_ratios = ['ROE', 'ROA', 'EPS', 'P/E', 'Debt/Equity']
            
            found_ratios = []
            for ratio in key_ratios:
                matching_cols = [col for col in df_ratios.columns if ratio.lower() in col.lower()]
                if matching_cols:
                    found_ratios.append(matching_cols[0])
            
            if found_ratios and 'stock_code' in df_ratios.columns:
                self.add_to_report(f"\n📊 Chỉ số tài chính tìm thấy: {len(found_ratios)}")
                for ratio in found_ratios:
                    self.add_to_report(f"   • {ratio}")
                
                # Phân tích từng chỉ số
                for i, ratio in enumerate(found_ratios[:3], 1):  # Top 3 chỉ số
                    ratio_by_company = df_ratios.groupby('stock_code')[ratio].mean().sort_values(ascending=False)
                    
                    self.add_to_report(f"\n📊 {ratio} trung bình:")
                    for idx, val in ratio_by_company.items():
                        self.add_to_report(f"   • {idx}: {val:.2f}")
                    
                    # Biểu đồ
                    plt.figure(figsize=(12, 6))
                    ratio_by_company.plot(kind='bar', color='coral', edgecolor='black')
                    plt.title(f'{ratio} Trung Bình Theo Công Ty', fontsize=16, fontweight='bold')
                    plt.xlabel('Công ty', fontsize=12)
                    plt.ylabel(ratio, fontsize=12)
                    plt.xticks(rotation=0)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    output_file = os.path.join(self.figures_dir, f'03_ratio_{i}_{ratio.replace("/", "_").replace(" ", "_")}.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.add_to_report(f"📊 Đã lưu: {output_file}")
        
        self.add_to_report("\n✅ Phần 4 hoàn thành!\n")
    
    def section5_trend_analysis(self):
        """Phần 5: Phân tích xu hướng"""
        self.add_to_report("\n" + "="*70)
        self.add_to_report("📋 PHẦN 5: PHÂN TÍCH XU HƯỚNG")
        self.add_to_report("="*70)
        
        # Tìm cột revenue và period
        revenue_cols = [col for col in self.df.columns if 'revenue' in col.lower() and 'yoy' not in col.lower()]
        
        if revenue_cols and 'period' in self.df.columns and 'stock_code' in self.df.columns:
            revenue_col = revenue_cols[0]
            
            if 'report_type' in self.df.columns:
                df_income = self.df[self.df['report_type'] == 'Income Statement'].copy()
                
                # Pivot table
                pivot = df_income.pivot_table(
                    values=revenue_col,
                    index='period',
                    columns='stock_code',
                    aggfunc='mean'
                )
                
                # Sắp xếp theo period
                pivot = pivot.sort_index()
                
                self.add_to_report(f"\n📊 Xu hướng doanh thu theo kỳ:")
                self.add_to_report(str(pivot.tail()))
                
                # Biểu đồ line chart
                plt.figure(figsize=(14, 7))
                for col in pivot.columns:
                    plt.plot(pivot.index, pivot[col], marker='o', label=col, linewidth=2)
                
                plt.title('Xu Hướng Doanh Thu Theo Kỳ', fontsize=16, fontweight='bold')
                plt.xlabel('Kỳ báo cáo', fontsize=12)
                plt.ylabel('Doanh thu (Tỷ VNĐ)', fontsize=12)
                plt.legend(title='Công ty', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                output_file = os.path.join(self.figures_dir, '04_revenue_trend.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.add_to_report(f"\n📊 Đã lưu biểu đồ: {output_file}")
        
        self.add_to_report("\n✅ Phần 5 hoàn thành!\n")
    
    def section6_correlation_analysis(self):
        """Phần 6: Phân tích tương quan"""
        self.add_to_report("\n" + "="*70)
        self.add_to_report("📋 PHẦN 6: PHÂN TÍCH TƯƠNG QUAN")
        self.add_to_report("="*70)
        
        # Chọn các cột numeric quan trọng
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Loại bỏ các cột không cần thiết
        exclude_cols = ['index', 'yearReport', 'lengthReport', 'year', 'quarter']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) > 1:
            # Tính correlation
            corr_matrix = self.df[numeric_cols].corr()
            
            # Top 10 correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'corr': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('corr', ascending=False, key=abs)
            
            self.add_to_report(f"\n📊 Top 10 tương quan mạnh nhất:")
            for idx, row in corr_df.head(10).iterrows():
                self.add_to_report(f"   • {row['var1'][:30]} <-> {row['var2'][:30]}: {row['corr']:.3f}")
            
            # Heatmap (chỉ top 15 biến)
            if len(numeric_cols) > 15:
                # Chọn 15 cột có variance cao nhất
                variances = self.df[numeric_cols].var().sort_values(ascending=False)
                top_cols = variances.head(15).index.tolist()
            else:
                top_cols = numeric_cols
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                self.df[top_cols].corr(),
                annot=False,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            plt.title('Ma Trận Tương Quan (Top 15 Biến)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_file = os.path.join(self.figures_dir, '05_correlation_heatmap.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.add_to_report(f"\n📊 Đã lưu heatmap: {output_file}")
        
        self.add_to_report("\n✅ Phần 6 hoàn thành!\n")
    
    def save_report(self, output_file):
        """Lưu report"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print(f"\n💾 Đã lưu report: {output_file}")

def main():
    print("\n" + "="*70)
    print("💼 FINANCIAL DATA - EXPLORATORY ANALYSIS")
    print("="*70)
    
    # Input file
    input_file = '../data/processed/financial_data_clean.csv'
    report_file = '../reports/eda_report.txt'
    
    # Khởi tạo EDA
    eda = FinancialEDA(input_file)
    
    # Chạy các phần phân tích
    eda.section1_overview()
    eda.section2_revenue_analysis()
    eda.section3_profitability_analysis()
    eda.section4_financial_ratios()
    eda.section5_trend_analysis()
    eda.section6_correlation_analysis()
    
    # Lưu report
    eda.save_report(report_file)
    
    print("\n" + "="*70)
    print("✅ EDA HOÀN THÀNH!")
    print("="*70)
    print(f"\n📊 Đã tạo:")
    print(f"   • Report: {report_file}")
    print(f"   • Figures: {eda.figures_dir}/")
    print("\n➡️  BƯỚC TIẾP THEO:")
    print("   • Xem report: notepad {report_file}")
    print("   • Xem figures: explorer {eda.figures_dir}")
    print("   • Dashboard: python 04_dashboard.py")
    print("="*70)

if __name__ == "__main__":
    main()
