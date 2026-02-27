"""
Financial Data Analysis Dashboard
Interactive Streamlit App for Revenue Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from pathlib import Path

# ===== SETUP PATHS =====
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / 'data' / 'processed'
MODEL_DIR = project_root / 'models'
FIGURES_DIR = project_root / 'reports' / 'figures'

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Financial Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
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
</style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        features_file = DATA_DIR / 'features_engineered.csv'
        test_pred_file = DATA_DIR / 'test_predictions.csv'
        ticker_perf_file = DATA_DIR / 'ticker_performance.csv'
        feat_imp_file = DATA_DIR / 'feature_importance.csv'
        
        if not features_file.exists():
            st.error(f"❌ File not found: {features_file}")
            return None, None, None, None
        
        features_df = pd.read_csv(features_file)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        test_pred = None
        if test_pred_file.exists():
            test_pred = pd.read_csv(test_pred_file)
            test_pred['date'] = pd.to_datetime(test_pred['date'])
        
        ticker_perf = None
        if ticker_perf_file.exists():
            ticker_perf = pd.read_csv(ticker_perf_file)
        
        feat_imp = None
        if feat_imp_file.exists():
            feat_imp = pd.read_csv(feat_imp_file)
        
        return features_df, test_pred, ticker_perf, feat_imp
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model_file = MODEL_DIR / 'rf_revenue_model.pkl'
        features_file = MODEL_DIR / 'model_features.txt'
        
        if not model_file.exists():
            st.warning(f"⚠️ Model file not found: {model_file}")
            return None, None
        
        model = joblib.load(model_file)
        
        feature_names = None
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        
        return model, feature_names
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ===== MAIN APP =====
def main():
    st.markdown('<h1 class="main-header">📊 Financial Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Revenue Prediction for Top 10 Vietnamese Companies (2007-2025)")
    st.markdown("---")
    
    with st.spinner("Loading data..."):
        features_df, test_pred, ticker_perf, feat_imp = load_data()
        model, feature_names = load_model()
    
    if features_df is None:
        st.error("❌ Cannot load data. Please check if data files exist.")
        st.info(f"📁 Expected data directory: `{DATA_DIR}`")
        st.stop()
    
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["📈 Overview", "🔍 Company Analysis", "🤖 Model Performance", "🎯 Make Prediction"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Project Info:**
    - 📅 Period: 2007-2025
    - 🏢 Companies: {len(features_df['ticker'].unique())}
    - 📊 Records: {len(features_df):,}
    - 🤖 Model: Random Forest
    - 📈 Best R²: 0.98 (VNM, FPT)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Companies:**
    - VHM (Vinhomes)
    - VIC (Vingroup)
    - VNM (Vinamilk)
    - HPG (Hòa Phát)
    - TCB (Techcombank)
    - VCB (Vietcombank)
    - MBB (MB Bank)
    - GAS (PV Gas)
    - MSN (Masan)
    - FPT (FPT Corp)
    """)
    
    if page == "📈 Overview":
        st.header("📈 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(features_df):,}")
        with col2:
            st.metric("Companies", len(features_df['ticker'].unique()))
        with col3:
            avg_revenue = features_df['Revenue (Trillion VND)'].mean()
            st.metric("Avg Revenue", f"{avg_revenue:.2f}T VND")
        with col4:
            date_range = f"{features_df['date'].min().year}-{features_df['date'].max().year}"
            st.metric("Time Period", date_range)
        
        st.markdown("---")
        st.subheader("📊 Revenue Trends (All Companies)")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        for ticker in sorted(features_df['ticker'].unique()):
            ticker_data = features_df[features_df['ticker'] == ticker]
            ax.plot(ticker_data['date'], ticker_data['Revenue (Trillion VND)'], 
                   marker='o', label=ticker, linewidth=2, markersize=3, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Revenue (Trillion VND)', fontsize=12)
        ax.set_title('Revenue Trends - All Companies (2007-2025)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=5)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        st.subheader("🏆 Top 5 Companies by Average Revenue")
        
        top_companies = features_df.groupby('ticker')['Revenue (Trillion VND)'].mean().sort_values(ascending=False).head(5)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Rankings")
            for i, (ticker, revenue) in enumerate(top_companies.items(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                st.markdown(f"{emoji} **{i}. {ticker}:** {revenue:.2f} Trillion VND")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['gold', 'silver', '#CD7F32', 'steelblue', 'lightsteelblue']
            ax.barh(range(len(top_companies)), top_companies.values, color=colors, edgecolor='black')
            ax.set_yticks(range(len(top_companies)))
            ax.set_yticklabels(top_companies.index)
            ax.set_xlabel('Average Revenue (Trillion VND)', fontsize=11)
            ax.set_title('Top 5 Companies by Revenue', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    elif page == "🔍 Company Analysis":
        st.header("🔍 Company Analysis")
        
        ticker = st.selectbox("Select Company:", sorted(features_df['ticker'].unique()), index=0)
        ticker_data = features_df[features_df['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) == 0:
            st.error(f"No data available for {ticker}")
            st.stop()
        
        st.subheader(f"📊 {ticker} - Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_rev = ticker_data['Revenue (Trillion VND)'].mean()
            st.metric("Avg Revenue", f"{avg_rev:.2f}T VND")
        
        with col2:
            latest_rev = ticker_data['Revenue (Trillion VND)'].iloc[-1]
            st.metric("Latest Revenue", f"{latest_rev:.2f}T VND")
        
        with col3:
            if 'Revenue_YoY_Change' in ticker_data.columns:
                avg_yoy = ticker_data['Revenue_YoY_Change'].mean()
                st.metric("Avg YoY Growth", f"{avg_yoy:.1f}%")
            else:
                st.metric("Avg YoY Growth", "N/A")
        
        with col4:
            if 'ROA' in ticker_data.columns:
                avg_roa = ticker_data['ROA'].mean()
                st.metric("Avg ROA", f"{avg_roa:.2f}%")
            else:
                st.metric("Avg ROA", "N/A")
        
        st.markdown("---")
        st.subheader("📈 Revenue Trend")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ticker_data['date'], ticker_data['Revenue (Trillion VND)'], 
               marker='o', linewidth=2.5, markersize=6, color='steelblue', label='Revenue')
        
        if 'Revenue_MA_4Q' in ticker_data.columns:
            ax.plot(ticker_data['date'], ticker_data['Revenue_MA_4Q'], 
                   linestyle='--', linewidth=2, color='orange', alpha=0.7, label='MA 4Q')
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Revenue (Trillion VND)', fontsize=11)
        ax.set_title(f'{ticker} - Revenue Trend', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        if 'Revenue_YoY_Change' in ticker_data.columns:
            st.subheader("📊 Year-over-Year Growth")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            yoy_data = ticker_data[ticker_data['Revenue_YoY_Change'].notna()]
            
            if len(yoy_data) > 0:
                colors = ['green' if x > 0 else 'red' for x in yoy_data['Revenue_YoY_Change']]
                ax.bar(yoy_data['date'], yoy_data['Revenue_YoY_Change'], 
                      color=colors, edgecolor='black', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                ax.set_xlabel('Date', fontsize=11)
                ax.set_ylabel('YoY Growth (%)', fontsize=11)
                ax.set_title(f'{ticker} - Year-over-Year Growth', fontsize=13, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance")
        
        st.subheader("📊 Overall Test Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", "4.12 Trillion VND")
        with col2:
            st.metric("RMSE", "14.45 Trillion VND")
        with col3:
            st.metric("R² Score", "0.4480")
        
        st.info("📝 Overall R² is moderate due to high volatility in real estate sector (VIC, VHM).")
        
        st.markdown("---")
        st.subheader("🎯 Performance by Company")
        
        if ticker_perf is not None:
            ticker_perf_sorted = ticker_perf.sort_values('R2', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if x < 0 else 'orange' if x < 0.5 else 'yellow' if x < 0.8 else 'green' 
                     for x in ticker_perf_sorted['R2']]
            ax.barh(range(len(ticker_perf_sorted)), ticker_perf_sorted['R2'], 
                   color=colors, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(ticker_perf_sorted)))
            ax.set_yticklabels(ticker_perf_sorted['Ticker'])
            ax.set_xlabel('R² Score', fontsize=12)
            ax.set_title('Model R² Score by Company', fontsize=13, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.axvline(x=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (R²≥0.8)')
            ax.grid(axis='x', alpha=0.3)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            st.subheader("📋 Detailed Performance Metrics")
            st.dataframe(ticker_perf.sort_values('R2', ascending=False), use_container_width=True)
        
        if feat_imp is not None:
            st.markdown("---")
            st.subheader("🔍 Feature Importance")
            
            top_features = feat_imp.head(15)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue', edgecolor='black')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    elif page == "🎯 Make Prediction":
        st.header("🎯 Make Revenue Prediction")
        
        st.info("ℹ️ **Demo Mode:** Predictions based on latest available data.")
        
        ticker = st.selectbox("Select Company:", sorted(features_df['ticker'].unique()), index=0)
        ticker_data = features_df[features_df['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) == 0:
            st.error(f"No data available for {ticker}")
            st.stop()
        
        latest_data = ticker_data.iloc[-1]
        
        st.subheader(f"📊 Latest Data for {ticker}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**📅 Date:** {latest_data['date'].strftime('%Y-%m-%d')}")
            st.markdown(f"**💰 Latest Revenue:** {latest_data['Revenue (Trillion VND)']:.2f} Trillion VND")
            if 'Revenue_MA_4Q' in latest_data:
                st.markdown(f"**📈 Revenue MA 4Q:** {latest_data['Revenue_MA_4Q']:.2f} Trillion VND")
        
        with col2:
            if 'ROA' in latest_data:
                st.markdown(f"**📊 ROA:** {latest_data['ROA']:.2f}%")
            if 'Profit_Margin' in latest_data:
                st.markdown(f"**💹 Profit Margin:** {latest_data['Profit_Margin']:.2f}%")
            if 'Revenue_YoY_Change' in latest_data:
                st.markdown(f"**📈 YoY Growth:** {latest_data['Revenue_YoY_Change']:.2f}%")
        
        st.markdown("---")
        
        if st.button("🔮 Predict Next Quarter Revenue", type="primary", use_container_width=True):
            if model is not None and feature_names is not None:
                try:
                    X = latest_data[feature_names].values.reshape(1, -1)
                    prediction = model.predict(X)[0]
                    
                    st.success(f"## 🎯 Predicted Revenue: **{prediction:.2f} Trillion VND**")
                    
                    actual_latest = latest_data['Revenue (Trillion VND)']
                    change = ((prediction - actual_latest) / actual_latest) * 100
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Quarter", f"{actual_latest:.2f}T VND")
                    with col2:
                        st.metric("Predicted Next Quarter", f"{prediction:.2f}T VND")
                    with col3:
                        st.metric("Expected Change", f"{change:+.1f}%")
                    
                    if ticker_perf is not None:
                        ticker_r2 = ticker_perf[ticker_perf['Ticker'] == ticker]['R2'].values
                        if len(ticker_r2) > 0:
                            r2_score = ticker_r2[0]
                            confidence = "Very High" if r2_score >= 0.95 else "High" if r2_score >= 0.80 else "Moderate" if r2_score >= 0.50 else "Low"
                            color = "green" if r2_score >= 0.95 else "blue" if r2_score >= 0.80 else "orange" if r2_score >= 0.50 else "red"
                            
                            st.markdown("---")
                            st.markdown(f"### 🎯 Prediction Confidence: :{color}[{confidence}]")
                            st.info(f"**Model R² for {ticker}:** {r2_score:.4f}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.error("❌ Model not loaded.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Financial Data Analysis Dashboard | Built with Streamlit</p>
        <p>Data: VNStock API | Model: Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
