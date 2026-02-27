"""
Revenue Prediction Model
- Dự đoán Revenue quý tiếp theo
- Sử dụng Random Forest, XGBoost, LSTM
- Đánh giá model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import os

# ===== CẤU HÌNH =====
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_data_path = os.path.join(project_root, 'data', 'processed')
figures_path = os.path.join(project_root, 'reports', 'figures')
models_path = os.path.join(project_root, 'models')

os.makedirs(models_path, exist_ok=True)

print("=" * 60)
print("REVENUE PREDICTION MODEL")
print("=" * 60)
print(f"📁 Data: {processed_data_path}")
print(f"📁 Models: {models_path}\n")

# ===== 1. ĐỌC DỮ LIỆU =====
print("=" * 60)
print("1. ĐANG ĐỌC DỮ LIỆU...")
print("=" * 60)

features_df = pd.read_csv(os.path.join(processed_data_path, 'features_engineered.csv'))
features_df['date'] = pd.to_datetime(features_df['date'])

print(f"✅ Features shape: {features_df.shape}")
print(f"✅ Columns: {len(features_df.columns)}")
print(f"✅ Date range: {features_df['date'].min()} to {features_df['date'].max()}")

# ===== 2. CHUẨN BỊ DỮ LIỆU CHO MODELING =====
print("\n" + "=" * 60)
print("2. CHUẨN BỊ DỮ LIỆU CHO MODELING...")
print("=" * 60)

# Target: Revenue (Trillion VND)
target_col = 'Revenue (Trillion VND)'

# Features: Loại bỏ identifiers và target
exclude_cols = ['ticker', 'date', 'year', 'quarter', target_col]
feature_cols = [col for col in features_df.columns if col not in exclude_cols]

print(f"\n📊 Target: {target_col}")
print(f"📊 Features: {len(feature_cols)}")
print(f"   {feature_cols[:10]}...")

# Lọc data không có missing values trong target
df_model = features_df[features_df[target_col].notna()].copy()
print(f"\n✅ Records for modeling: {len(df_model)}/{len(features_df)}")

# Fill missing values trong features bằng median
for col in feature_cols:
    if df_model[col].isnull().sum() > 0:
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)
        print(f"   ✅ Filled {col} with median: {median_val:.2f}")

# ===== 3. SPLIT DATA =====
print("\n" + "=" * 60)
print("3. SPLIT DATA (TIME-BASED)...")
print("=" * 60)

# Sort by date
df_model = df_model.sort_values('date')

# Split: 80% train, 20% test (time-based)
split_idx = int(len(df_model) * 0.8)

train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"✅ Train set: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
print(f"✅ Test set: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")
print(f"✅ Features: {X_train.shape[1]}")

# ===== 4. TRAIN RANDOM FOREST MODEL =====
print("\n" + "=" * 60)
print("4. TRAINING RANDOM FOREST MODEL...")
print("=" * 60)

# Train model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("📊 Training...")
rf_model.fit(X_train, y_train)
print("✅ Training completed!")

# ===== 5. ĐÁNH GIÁ MODEL =====
print("\n" + "=" * 60)
print("5. ĐÁNH GIÁ MODEL...")
print("=" * 60)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 TRAIN SET PERFORMANCE:")
print(f"   MAE:  {train_mae:.4f} Trillion VND")
print(f"   RMSE: {train_rmse:.4f} Trillion VND")
print(f"   R²:   {train_r2:.4f}")

print("\n📊 TEST SET PERFORMANCE:")
print(f"   MAE:  {test_mae:.4f} Trillion VND")
print(f"   RMSE: {test_rmse:.4f} Trillion VND")
print(f"   R²:   {test_r2:.4f}")

# ===== 6. FEATURE IMPORTANCE =====
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE...")
print("=" * 60)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, '07_feature_importance.png'), dpi=300, bbox_inches='tight')
print("\n✅ Saved: 07_feature_importance.png")
plt.close()

# ===== 7. VISUALIZE PREDICTIONS =====
print("\n" + "=" * 60)
print("7. VISUALIZE PREDICTIONS...")
print("=" * 60)

# Plot: Actual vs Predicted (Test Set)
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Revenue (Trillion VND)', fontsize=12)
plt.ylabel('Predicted Revenue (Trillion VND)', fontsize=12)
plt.title('Actual vs Predicted Revenue - Test Set', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, '08_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: 08_actual_vs_predicted.png")
plt.close()

# Plot: Predictions over time
test_results = test_df[['ticker', 'date', target_col]].copy()
test_results['Predicted'] = y_test_pred
test_results['Error'] = test_results[target_col] - test_results['Predicted']

plt.figure(figsize=(14, 6))
plt.plot(test_results['date'], test_results[target_col], 
         marker='o', label='Actual', linewidth=2, markersize=4)
plt.plot(test_results['date'], test_results['Predicted'], 
         marker='s', label='Predicted', linewidth=2, markersize=4)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue (Trillion VND)', fontsize=12)
plt.title('Revenue Prediction Over Time - Test Set', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, '09_predictions_over_time.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: 09_predictions_over_time.png")
plt.close()

# ===== 8. PREDICTIONS BY TICKER =====
print("\n" + "=" * 60)
print("8. PREDICTIONS BY TICKER...")
print("=" * 60)

print("\n📊 Test Set Performance by Ticker:")
for ticker in sorted(test_results['ticker'].unique()):
    ticker_data = test_results[test_results['ticker'] == ticker]
    
    if len(ticker_data) > 0:
        ticker_mae = mean_absolute_error(ticker_data[target_col], ticker_data['Predicted'])
        ticker_r2 = r2_score(ticker_data[target_col], ticker_data['Predicted'])
        
        print(f"   {ticker}: MAE = {ticker_mae:.4f}, R² = {ticker_r2:.4f} ({len(ticker_data)} records)")

# ===== 9. SAVE MODEL =====
print("\n" + "=" * 60)
print("9. SAVE MODEL...")
print("=" * 60)

import joblib

model_file = os.path.join(models_path, 'rf_revenue_model.pkl')
joblib.dump(rf_model, model_file)
print(f"✅ Model saved to: {model_file}")

# Save feature names
features_file = os.path.join(models_path, 'model_features.txt')
with open(features_file, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"✅ Features saved to: {features_file}")

# ===== 10. SUMMARY =====
print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETED!")
print("=" * 60)

print(f"\n📊 MODEL SUMMARY:")
print(f"   Algorithm: Random Forest")
print(f"   Features: {len(feature_cols)}")
print(f"   Train Records: {len(train_df)}")
print(f"   Test Records: {len(test_df)}")
print(f"   Test MAE: {test_mae:.4f} Trillion VND")
print(f"   Test R²: {test_r2:.4f}")

print(f"\n📁 Saved:")
print(f"   - Model: {model_file}")
print(f"   - Features: {features_file}")
print(f"   - Figures: {figures_path}")

print("\n✅ Sẵn sàng để deploy và tạo dashboard!")
