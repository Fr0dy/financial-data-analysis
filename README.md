# 📊 Financial Data Analysis & Revenue Prediction

Dự án phân tích dữ liệu tài chính và dự đoán doanh thu của 10 công ty hàng đầu Việt Nam (2007-2025)

## 🎯 Mục tiêu

- Thu thập và xử lý dữ liệu tài chính từ VNStock API
- Phân tích xu hướng doanh thu, tăng trưởng, và các chỉ số tài chính
- Xây dựng model Machine Learning để dự đoán doanh thu quý tiếp theo
- Tạo dashboard và báo cáo trực quan

## 📂 Cấu trúc Project

[Project 2026] Financial Data Analysis/
├── data/
│   ├── raw/              # Dữ liệu thô
│   └── processed/        # Dữ liệu đã xử lý
├── notebooks/            # Scripts phân tích
├── models/               # Models đã train
├── reports/
│   └── figures/          # Visualizations
└── README.md


## 🏆 Kết quả

- **516 records** từ 10 công ty (2007-2025)
- **19 features** được tạo từ feature engineering
- **Random Forest Model** với R² = 0.45 (overall test)
- **Best predictions**: VNM (R²=0.98), FPT (R²=0.98)
- **19 visualizations** và 1 báo cáo chi tiết

## 🔧 Technologies

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- vnstock3

## 📊 Key Insights

- **VIC, HPG, GAS** dẫn đầu về doanh thu (20-28 Trillion VND)
- **VHM, VIC** có tăng trưởng cao nhất nhưng biến động mạnh
- **VNM, FPT** ổn định và dễ dự đoán nhất
- Model hoạt động tốt cho các công ty ổn định, khó khăn với BĐS

## 📧 Contact

[Pham Ngoc Khanh] - [khanhpn.forwork@gmail.com]

Project Link: [https://github.com/Fr0dy/financial-data-analysis]
