"""
=============================================================================
INTERACTIVE FINANCIAL DASHBOARD
=============================================================================
Purpose: Dashboard tương tác để phân tích dữ liệu tài chính
Run: streamlit run 04_dashboard.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load cleaned data"""
    # Tự động tìm đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'processed', 'financial_data_clean.csv')
    
    # Kiểm tra file tồn tại
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        return df
    else:
        st.error("❌ Không tìm thấy file data! Vui lòng kiểm tra đường dẫn.")
        st.info(f"Đang tìm file tại: {data_path}")
        st.stop()

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Financial Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Filters")
    
    # Company filter
    companies = sorted(df['stock_code'].unique())
    selected_companies = st.sidebar.multiselect(
        "Chọn công ty:",
        companies,
        default=companies
    )
    
    # Report type filter
    report_types = sorted(df['report_type'].unique())
    selected_report = st.sidebar.selectbox(
        "Chọn loại báo cáo:",
        report_types
    )
    
    # Period filter
    if 'period' in df.columns:
        periods = sorted(df['period'].unique())
        # Lọc bỏ '0-Q0'
        periods = [p for p in periods if p != '0-Q0']
        
        if len(periods) > 0:
            selected_period_range = st.sidebar.select_slider(
                "Chọn khoảng thời gian:",
                options=periods,
                value=(periods[0], periods[-1])
            )
    
    # Filter data
    df_filtered = df[
        (df['stock_code'].isin(selected_companies)) &
        (df['report_type'] == selected_report)
    ]
    
    if 'period' in df.columns and len(periods) > 0:
        df_filtered = df_filtered[
            (df_filtered['period'] >= selected_period_range[0]) &
            (df_filtered['period'] <= selected_period_range[1])
        ]
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Info")
    st.sidebar.info(f"""
    **Records:** {len(df_filtered):,}  
    **Companies:** {len(selected_companies)}  
    **Report:** {selected_report}
    """)
    
    # Main content
    if len(df_filtered) == 0:
        st.warning("⚠️ Không có dữ liệu cho bộ lọc này!")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "💰 Revenue & Profit", 
        "📊 Financial Ratios",
        "🔍 Detailed Analysis"
    ])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.header("📈 Tổng Quan")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏢 Số công ty",
                value=len(selected_companies)
            )
        
        with col2:
            st.metric(
                label="📋 Số records",
                value=f"{len(df_filtered):,}"
            )
        
        with col3:
            if 'period' in df_filtered.columns:
                periods_count = len(df_filtered['period'].unique())
                st.metric(
                    label="📅 Số kỳ báo cáo",
                    value=periods_count
                )
        
        with col4:
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            st.metric(
                label="📊 Số chỉ số",
                value=len(numeric_cols)
            )
        
        st.markdown("---")
        
        # Company comparison
        st.subheader("🏢 So Sánh Công Ty")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Records per company
            company_counts = df_filtered['stock_code'].value_counts().sort_index()
            
            fig = px.bar(
                x=company_counts.index,
                y=company_counts.values,
                labels={'x': 'Công ty', 'y': 'Số records'},
                title='Số Records Theo Công Ty',
                color=company_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Period distribution
            if 'period' in df_filtered.columns:
                period_counts = df_filtered['period'].value_counts().sort_index()
                
                fig = px.line(
                    x=period_counts.index,
                    y=period_counts.values,
                    labels={'x': 'Kỳ', 'y': 'Số records'},
                    title='Phân Bố Theo Kỳ',
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: REVENUE & PROFIT ====================
    with tab2:
        st.header("💰 Doanh Thu & Lợi Nhuận")
        
        # Tìm cột revenue và profit
        revenue_cols = [col for col in df_filtered.columns if 'revenue' in col.lower() and 'yoy' not in col.lower()]
        profit_cols = [col for col in df_filtered.columns if 'profit' in col.lower() or 'parent company' in col.lower()]
        
        if revenue_cols and 'stock_code' in df_filtered.columns:
            revenue_col = revenue_cols[0]
            
            # Revenue by company
            st.subheader("📊 Doanh Thu Theo Công Ty")
            
            revenue_by_company = df_filtered.groupby('stock_code')[revenue_col].agg(['mean', 'sum', 'count']).reset_index()
            revenue_by_company = revenue_by_company.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    revenue_by_company,
                    x='stock_code',
                    y='mean',
                    title='Doanh Thu Trung Bình (Tỷ VNĐ)',
                    labels={'stock_code': 'Công ty', 'mean': 'Doanh thu TB'},
                    color='mean',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    revenue_by_company,
                    values='sum',
                    names='stock_code',
                    title='Tỷ Trọng Doanh Thu',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Revenue trend
            if 'period' in df_filtered.columns:
                st.subheader("📈 Xu Hướng Doanh Thu")
                
                pivot = df_filtered.pivot_table(
                    values=revenue_col,
                    index='period',
                    columns='stock_code',
                    aggfunc='mean'
                ).sort_index()
                
                fig = go.Figure()
                
                for col in pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot.index,
                        y=pivot[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Xu Hướng Doanh Thu Theo Kỳ',
                    xaxis_title='Kỳ',
                    yaxis_title='Doanh thu (Tỷ VNĐ)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Profit analysis
        if profit_cols and 'stock_code' in df_filtered.columns:
            profit_col = profit_cols[0]
            
            st.subheader("💵 Lợi Nhuận Theo Công Ty")
            
            profit_by_company = df_filtered.groupby('stock_code')[profit_col].agg(['mean', 'sum']).reset_index()
            profit_by_company = profit_by_company.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    profit_by_company,
                    x='stock_code',
                    y='mean',
                    title='Lợi Nhuận Trung Bình (Tỷ VNĐ)',
                    labels={'stock_code': 'Công ty', 'mean': 'Lợi nhuận TB'},
                    color='mean',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Profit margin (if revenue exists)
                if revenue_cols:
                    margin_data = df_filtered.groupby('stock_code').agg({
                        revenue_col: 'mean',
                        profit_col: 'mean'
                    }).reset_index()
                    margin_data['margin'] = (margin_data[profit_col] / margin_data[revenue_col] * 100)
                    margin_data = margin_data.sort_values('margin', ascending=False)
                    
                    fig = px.bar(
                        margin_data,
                        x='stock_code',
                        y='margin',
                        title='Biên Lợi Nhuận (%)',
                        labels={'stock_code': 'Công ty', 'margin': 'Margin (%)'},
                        color='margin',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: FINANCIAL RATIOS ====================
    with tab3:
        st.header("📊 Chỉ Số Tài Chính")
        
        # Tìm các chỉ số
        key_ratios = ['ROE', 'ROA', 'EPS', 'P/E', 'Debt/Equity']
        
        found_ratios = []
        for ratio in key_ratios:
            matching_cols = [col for col in df_filtered.columns if ratio.lower() in col.lower()]
            if matching_cols:
                found_ratios.append(matching_cols[0])
        
        if found_ratios and 'stock_code' in df_filtered.columns:
            # Ratio selector
            selected_ratio = st.selectbox(
                "Chọn chỉ số:",
                found_ratios
            )
            
            # Ratio by company
            ratio_by_company = df_filtered.groupby('stock_code')[selected_ratio].mean().sort_values(ascending=False).reset_index()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    ratio_by_company,
                    x='stock_code',
                    y=selected_ratio,
                    title=f'{selected_ratio} Trung Bình Theo Công Ty',
                    labels={'stock_code': 'Công ty', selected_ratio: selected_ratio},
                    color=selected_ratio,
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Top 3")
                for idx, row in ratio_by_company.head(3).iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{row['stock_code']}</h4>
                        <h2>{row[selected_ratio]:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            
            # Ratio trend
            if 'period' in df_filtered.columns:
                st.subheader(f"📈 Xu Hướng {selected_ratio}")
                
                pivot = df_filtered.pivot_table(
                    values=selected_ratio,
                    index='period',
                    columns='stock_code',
                    aggfunc='mean'
                ).sort_index()
                
                fig = go.Figure()
                
                for col in pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot.index,
                        y=pivot[col],
                        mode='lines+markers',
                        name=col
                    ))
                
                fig.update_layout(
                    title=f'Xu Hướng {selected_ratio} Theo Kỳ',
                    xaxis_title='Kỳ',
                    yaxis_title=selected_ratio,
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Compare multiple ratios
            if len(found_ratios) >= 2:
                st.subheader("🔄 So Sánh Nhiều Chỉ Số")
                
                selected_ratios_compare = st.multiselect(
                    "Chọn các chỉ số để so sánh:",
                    found_ratios,
                    default=found_ratios[:3]
                )
                
                if selected_ratios_compare:
                    comparison_data = df_filtered.groupby('stock_code')[selected_ratios_compare].mean().reset_index()
                    
                    fig = go.Figure()
                    
                    for ratio in selected_ratios_compare:
                        fig.add_trace(go.Bar(
                            name=ratio,
                            x=comparison_data['stock_code'],
                            y=comparison_data[ratio]
                        ))
                    
                    fig.update_layout(
                        title='So Sánh Các Chỉ Số Tài Chính',
                        xaxis_title='Công ty',
                        yaxis_title='Giá trị',
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 4: DETAILED ANALYSIS ====================
    with tab4:
        st.header("🔍 Phân Tích Chi Tiết")
        
        # Data table
        st.subheader("📋 Dữ Liệu Chi Tiết")
        
        # Column selector
        all_cols = df_filtered.columns.tolist()
        default_cols = ['stock_code', 'report_type', 'period', 'year', 'quarter']
        default_cols = [col for col in default_cols if col in all_cols]
        
        selected_cols = st.multiselect(
            "Chọn cột hiển thị:",
            all_cols,
            default=default_cols[:5] if len(default_cols) >= 5 else default_cols
        )
        
        if selected_cols:
            st.dataframe(
                df_filtered[selected_cols].head(100),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df_filtered[selected_cols].to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"financial_data_{selected_report.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # Statistics
        st.subheader("📊 Thống Kê Mô Tả")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_stat_col = st.selectbox(
                "Chọn cột để xem thống kê:",
                numeric_cols
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Thống Kê Cơ Bản")
                stats = df_filtered[selected_stat_col].describe()
                st.dataframe(stats, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Phân Phối")
                fig = px.histogram(
                    df_filtered,
                    x=selected_stat_col,
                    nbins=30,
                    title=f'Phân Phối {selected_stat_col}',
                    color_discrete_sequence=['skyblue']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Financial Analysis Dashboard | Data from VNStock API | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
