# Hướng dẫn chạy và xem sản phẩm đồ án

## 1. Truy cập bảng điều khiển đã triển khai trên Streamlit

Bảng điều khiển động của đồ án đã được triển khai trên Streamlit và có thể truy cập tại:

`https://datnhuyenngoc.streamlit.app`

Bảng điều khiển cho phép xem trực quan các kết quả phân tích, so sánh mô hình và kết quả dự báo nguy cơ bệnh đường hô hấp trên tập kiểm tra.

Lưu ý: Bảng điều khiển là sản phẩm minh họa kết quả phân tích và dự báo từ bộ dữ liệu nghiên cứu. Đây chưa phải là hệ thống dự báo thời gian thực hoặc hệ thống cảnh báo y tế vận hành chính thức.

## 2. Cấu trúc dữ liệu đầu vào

Đặt file dữ liệu `global_climate_health_impact_tracker_2015_2025.csv` theo một trong các vị trí sau:

- Cùng thư mục với notebook.
- Thư mục `data/`.
- Thư mục `datn_huph_input/`.
- Trên Google Colab: `/content/drive/MyDrive/` hoặc `/content/drive/MyDrive/datn_huph_input/`.

## 3. Chạy notebook để tái lập kết quả

Mở `DATN.ipynb` bằng Google Colab hoặc VS Code/Jupyter, sau đó chọn **Run All**.

Kết quả sẽ được lưu trong thư mục:

`datn_huph_output_plus/`

Notebook sẽ tự động sinh các bảng, biểu đồ, mô hình đã lưu và các tệp phục vụ báo cáo.

## 4. Chạy bảng điều khiển trên máy cá nhân nếu cần

Trường hợp muốn chạy dashboard trên máy cá nhân, cài thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

Sau đó chạy Streamlit:

```bash
python -m streamlit run datn_huph_output_plus/dashboard/app.py
```

Mở địa chỉ sau trên trình duyệt:

`http://localhost:8501`

## 5. Các thư mục output

- `figures/`: hình biểu đồ dùng trong báo cáo.
- `tables/`: bảng CSV nguồn để mở bằng Excel hoặc kiểm tra lại dữ liệu.
- `word_table_docx/`: bảng Word `.docx` đã định dạng, có thể copy trực tiếp vào báo cáo Word.
- `word_table_images/`: ảnh PNG của các bảng để chèn nhanh vào Word nếu không cần chỉnh sửa bảng.
- `models/`: mô hình học máy đã lưu.
- `report/`: tệp tóm tắt kết quả và thông tin môi trường chạy.
- `dashboard/`: bảng điều khiển HTML và ứng dụng Streamlit.

## 6. Ghi chú phương pháp

Bảng điều khiển sử dụng kết quả phân tích được tạo ra từ notebook. Các kết quả dự báo được trình bày trên tập kiểm tra nhằm minh họa khả năng ứng dụng mô hình học máy trong đánh giá nguy cơ bệnh đường hô hấp.

Dashboard chưa được kết nối với dữ liệu thời gian thực, do đó không nên diễn giải là hệ thống cảnh báo y tế vận hành chính thức.

## 7. Gợi ý sử dụng trong báo cáo Word

- Với biểu đồ, dùng file `.png` trong thư mục `figures/`.
- Với bảng, ưu tiên mở file `.docx` trong `word_table_docx/`, sau đó copy bảng và dán vào báo cáo Word chính.
- File `.csv` trong `tables/` là dữ liệu nguồn để kiểm tra hoặc xử lý lại bằng Excel.
- Ảnh bảng trong `word_table_images/` chỉ nên dùng khi cần chèn nhanh và không cần chỉnh sửa nội dung bảng.
- Xem `tables/supp_17_word_output_mapping.csv` để biết bảng/hình nào nên đặt ở phần nào trong báo cáo Word.

## 8. Ghi chú khi nộp bài

File `datn_huph_output_plus.zip` được tạo tự động sau khi chạy notebook. File này chứa toàn bộ kết quả đầu ra của đồ án, bao gồm bảng, biểu đồ, mô hình, báo cáo phụ trợ và dashboard.

Nếu đã triển khai dashboard trên Streamlit, khi nộp bài có thể cung cấp thêm đường dẫn trực tuyến:

`https://datnhuyenngoc.streamlit.app`
