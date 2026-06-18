# Tóm tắt tự động phục vụ viết báo cáo

## 1. Tổng quan bộ dữ liệu
- Số dòng: 14100
- Số cột: 30
- Số quốc gia: 25
- Số vùng địa lý: 8
- Khoảng thời gian: 2015-01-04 đến 2025-10-19
- Số giá trị thiếu: 0

## 2. Phương pháp đề xuất
Phương pháp đề xuất là **AQM-Lag**, kết hợp:
1. Các biến khí hậu, chất lượng không khí và kinh tế - xã hội gốc.
2. Đặc trưng độ trễ 1, 2 và 4 tuần.
3. Trung bình trượt của các biến hô hấp và môi trường quan trọng.
4. Mã hóa mùa vụ bằng sin/cos của số tuần trong năm.
5. Chỉ số tổng hợp **aqm_stress_index** xây dựng từ PM2.5, AQI, bất thường nhiệt độ, số ngày nắng nóng và sự kiện thời tiết cực đoan.
6. Các biến tương tác đơn giản: PM2.5 x nhiệt độ, AQI x nắng nóng, lượng mưa x lũ lụt.

## 3. Phân chia dữ liệu
- Huấn luyện: 2015-2021
- Kiểm định: 2022-2023
- Kiểm tra: 2024-2025

## 4. Mô hình tốt nhất trên tập kiểm định
- Bộ đặc trưng: Bộ đặc trưng AQM-Lag đề xuất
- Mô hình: Hồi quy Ridge

## 5. Chỉ số đánh giá trên tập kiểm tra
- MAE: 7.9421
- RMSE: 9.9694
- R2: 0.5764
- MAPE (%): 12.4891

## 6. Lưu ý diễn giải quan trọng
AQM-Lag là chiến lược tổ chức bộ đặc trưng có cấu trúc. Cần diễn giải đây là cách kết hợp thông tin độ trễ, trung bình trượt, mùa vụ, chỉ số căng thẳng và tương tác; không phải một thuật toán học máy độc lập mới.

## 7. Gợi ý sử dụng đầu ra theo chương
- Chương 1: table_01_dataset_overview, table_02_variable_groups, table_03_missing_values, table_05_outlier_summary
- Chương 3: table_08_time_split, supp_aqm_stress_weights, table_09_feature_catalog
- Chương 4: table_15_ablation_best_by_feature_set, table_11_validation_model_ranking, table_12_test_model_ranking, supp_top10_permutation_importance
- Hình kết quả: figure_10_actual_vs_predicted, figure_12_global_actual_vs_predicted_over_time, figure_13_top_permutation_importance, figure_14_validation_rmse_by_feature_set
- Demo/sản phẩm: dashboard/dashboard_climate_respiratory.html và dashboard/app.py
- Ghi chú bảng điều khiển: không dùng bản đồ nền choropleth mặc định; so sánh không gian được thể hiện bằng biểu đồ vùng/quốc gia.
- Tài nguyên phụ lục: các tệp có tiền tố supp_*
- Danh mục biểu đồ bảng điều khiển: supp_16_dashboard_chart_catalog
- Ánh xạ đầu ra cho Word: supp_17_word_output_mapping
- Ảnh bảng sẵn sàng chèn Word: word_table_images/table_XX_*.png

## 8. Ghi chú
- Các bảng chính dùng tiền tố table_XX_*.
- Các đầu ra trung gian/phụ lục dùng tiền tố supp_XX_*.
- Khi thay đổi phương pháp hoặc cấu hình mô hình, cần chạy lại notebook để toàn bộ bảng và hình nhất quán.
