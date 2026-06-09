
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

APP_TITLE = "Bảng điều khiển phân tích chất lượng không khí, khí tượng và bệnh hô hấp"

CHART_TITLES = {
    "D01": "D01. Xu hướng tỷ lệ bệnh hô hấp theo năm và vùng",
    "D02": "D02. PM2.5 và tỷ lệ bệnh hô hấp",
    "D03": "D03. Tỷ lệ bệnh hô hấp trung bình theo vùng",
    "D04": "D04. Xếp hạng quốc gia theo tỷ lệ bệnh hô hấp trung bình",
    "D05": "D05. Phân bố tỷ lệ bệnh hô hấp theo nhóm thu nhập",
    "D06": "D06. Giá trị thực tế và dự báo trên tập kiểm tra",
    "D07": "D07. Xu hướng trung bình thực tế và dự báo trên tập kiểm tra",
    "D08": "D08. RMSE trên tập kiểm tra theo vùng",
}

LABELS = {
    "year": "Năm",
    "date": "Thời gian",
    "region": "Vùng địa lý",
    "country_name": "Quốc gia",
    "income_level": "Nhóm thu nhập",
    "region_vi": "Vùng địa lý",
    "income_level_vi": "Nhóm thu nhập",
    "pm25_ugm3": "PM2.5 (µg/m³)",
    "air_quality_index": "Chỉ số chất lượng không khí",
    "respiratory_disease_rate": "Tỷ lệ bệnh hô hấp",
    "mean_resp": "Tỷ lệ bệnh hô hấp trung bình",
    "prediction": "Giá trị dự báo",
    "actual": "Giá trị thực tế",
    "predicted": "Giá trị dự báo",
    "residual": "Phần dư",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "R2": "R²",
    "n_records": "Số bản ghi",
    "n_countries": "Số quốc gia",
    "mean_pm25": "PM2.5 trung bình (µg/m³)",
    "mean_aqi": "AQI trung bình",
}


INCOME_LABELS = {
    "High": "Thu nhập cao",
    "High income": "Thu nhập cao",
    "Low": "Thu nhập thấp",
    "Low income": "Thu nhập thấp",
    "Lower-Middle": "Thu nhập trung bình thấp",
    "Lower middle income": "Thu nhập trung bình thấp",
    "Upper-Middle": "Thu nhập trung bình cao",
    "Upper middle income": "Thu nhập trung bình cao",
    "Middle": "Thu nhập trung bình",
}

REGION_LABELS = {
    "Africa": "Châu Phi",
    "Asia": "Châu Á",
    "Asia Pacific": "Châu Á - Thái Bình Dương",
    "East Asia": "Đông Á",
    "South Asia": "Nam Á",
    "Southeast Asia": "Đông Nam Á",
    "Central Asia": "Trung Á",
    "Europe": "Châu Âu",
    "North America": "Bắc Mỹ",
    "South America": "Nam Mỹ",
    "Latin America": "Mỹ Latinh",
    "Latin America & Caribbean": "Mỹ Latinh và Caribe",
    "Middle East": "Trung Đông",
    "Middle East & North Africa": "Trung Đông và Bắc Phi",
    "Oceania": "Châu Đại Dương",
    "Sub-Saharan Africa": "Châu Phi cận Sahara",
}

def vi_region(value):
    return REGION_LABELS.get(value, value)

def vi_income(value):
    return INCOME_LABELS.get(value, value)

def add_vietnamese_display_columns(dataframe):
    display_df = dataframe.copy()
    if "region" in display_df.columns:
        display_df["region_vi"] = display_df["region"].map(vi_region)
    if "income_level" in display_df.columns:
        display_df["income_level_vi"] = display_df["income_level"].map(vi_income)
    return display_df

DASHBOARD_MAPPING = pd.DataFrame([
    {"Mã": "D01", "Biểu đồ": CHART_TITLES["D01"], "Liên hệ báo cáo Word/tệp đầu ra": "figure_02_annual_respiratory_trend; figure_04_respiratory_by_region"},
    {"Mã": "D02", "Biểu đồ": CHART_TITLES["D02"], "Liên hệ báo cáo Word/tệp đầu ra": "figure_06_pm25_vs_respiratory"},
    {"Mã": "D03", "Biểu đồ": CHART_TITLES["D03"], "Liên hệ báo cáo Word/tệp đầu ra": "figure_04_respiratory_by_region"},
    {"Mã": "D04", "Biểu đồ": CHART_TITLES["D04"], "Liên hệ báo cáo Word/tệp đầu ra": "table_07_country_summary"},
    {"Mã": "D05", "Biểu đồ": CHART_TITLES["D05"], "Liên hệ báo cáo Word/tệp đầu ra": "figure_03_respiratory_by_income"},
    {"Mã": "D06", "Biểu đồ": CHART_TITLES["D06"], "Liên hệ báo cáo Word/tệp đầu ra": "figure_10_actual_vs_predicted"},
    {"Mã": "D07", "Biểu đồ": CHART_TITLES["D07"], "Liên hệ báo cáo Word/tệp đầu ra": "figure_12_global_actual_vs_predicted_over_time"},
    {"Mã": "D08", "Biểu đồ": CHART_TITLES["D08"], "Liên hệ báo cáo Word/tệp đầu ra": "supp_10_test_error_by_region"},
])

AQM_LAG_FEATURE_TABLE = pd.DataFrame([
    {
        "Nhóm đặc trưng": "Biến gốc hiện tại",
        "Biến/thành phần": "country_name, region, income_level, year, month, week, temperature_celsius, temp_anomaly_celsius, precipitation_mm, heat_wave_days, drought_indicator, flood_indicator, extreme_weather_events, pm25_ugm3, air_quality_index, healthcare_access_index, gdp_per_capita_usd, mental_health_index, food_security_index",
        "Ý nghĩa": "Phản ánh điều kiện môi trường, khí tượng và bối cảnh quốc gia tại tuần quan sát."
    },
    {
        "Nhóm đặc trưng": "Đặc trưng trễ",
        "Biến/thành phần": "Các độ trễ 1, 2 và 4 tuần của respiratory_disease_rate, pm25_ugm3, air_quality_index, temperature_celsius, temp_anomaly_celsius, precipitation_mm, heat_wave_days, extreme_weather_events",
        "Ý nghĩa": "Giúp mô hình nắm bắt tác động trì hoãn của ô nhiễm không khí và khí tượng lên bệnh hô hấp."
    },
    {
        "Nhóm đặc trưng": "Trung bình trượt 4 tuần",
        "Biến/thành phần": "respiratory_disease_rate_roll4_mean, pm25_ugm3_roll4_mean, air_quality_index_roll4_mean, temp_anomaly_celsius_roll4_mean",
        "Ý nghĩa": "Phản ánh mức phơi nhiễm hoặc xu hướng nền ngắn hạn trước tuần dự báo."
    },
    {
        "Nhóm đặc trưng": "Mã hóa mùa vụ",
        "Biến/thành phần": "season_sin, season_cos",
        "Ý nghĩa": "Biểu diễn chu kỳ mùa vụ theo tuần trong năm, giúp mô hình hiểu tính lặp lại theo thời gian."
    },
    {
        "Nhóm đặc trưng": "Chỉ số tổng hợp",
        "Biến/thành phần": "aqm_stress_index",
        "Ý nghĩa": "Tóm tắt áp lực môi trường hô hấp từ PM2.5, AQI, bất thường nhiệt, nắng nóng và thời tiết cực đoan."
    },
    {
        "Nhóm đặc trưng": "Biến tương tác",
        "Biến/thành phần": "pm25_x_temperature, aqi_x_heatwave, precip_x_flood",
        "Ý nghĩa": "Cho phép mô hình nhận diện tác động kết hợp giữa các yếu tố môi trường."
    },
])

AQM_STRESS_WEIGHTS = pd.DataFrame([
    {"Thành phần": "PM2.5", "Biến": "pm25_ugm3", "Trọng số": 0.35},
    {"Thành phần": "AQI", "Biến": "air_quality_index", "Trọng số": 0.25},
    {"Thành phần": "Bất thường nhiệt độ", "Biến": "temp_anomaly_celsius", "Trọng số": 0.15},
    {"Thành phần": "Số ngày nắng nóng", "Biến": "heat_wave_days", "Trọng số": 0.15},
    {"Thành phần": "Thời tiết cực đoan", "Biến": "extreme_weather_events", "Trọng số": 0.10},
])


st.set_page_config(page_title=APP_TITLE, layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dashboard_data.csv"
PRED_PATH = BASE_DIR / "test_predictions.csv"

if not DATA_PATH.exists():
    st.error(f"Không tìm thấy file dữ liệu: {DATA_PATH}")
    st.stop()

if not PRED_PATH.exists():
    st.error(f"Không tìm thấy file dự báo: {PRED_PATH}")
    st.stop()

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    return data, pred

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

df, pred_df = load_data()
df = add_vietnamese_display_columns(df)
pred_df = add_vietnamese_display_columns(pred_df)

st.title(APP_TITLE)
st.caption("Sản phẩm tương tác")
st.sidebar.header("Bộ lọc dữ liệu")

year_min = int(df["year"].min())
year_max = int(df["year"].max())
year_range = st.sidebar.slider("Chọn khoảng năm", min_value=year_min, max_value=year_max, value=(year_min, year_max))

regions = sorted(df["region_vi"].dropna().unique())
selected_regions = st.sidebar.multiselect("Chọn vùng địa lý", regions, default=regions)

income_levels = sorted(df["income_level_vi"].dropna().unique())
selected_income = st.sidebar.multiselect("Chọn nhóm thu nhập", income_levels, default=income_levels)

countries = sorted(df["country_name"].dropna().unique())
selected_countries = st.sidebar.multiselect("Chọn quốc gia", countries, default=countries)

filtered_df = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1]) &
    (df["region_vi"].isin(selected_regions)) &
    (df["income_level_vi"].isin(selected_income)) &
    (df["country_name"].isin(selected_countries))
].copy()

filtered_pred = pred_df[
    (pred_df["date"].dt.year >= year_range[0]) &
    (pred_df["date"].dt.year <= year_range[1]) &
    (pred_df["region_vi"].isin(selected_regions)) &
    (pred_df["income_level_vi"].isin(selected_income)) &
    (pred_df["country_name"].isin(selected_countries))
].copy()

if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Số bản ghi", f"{len(filtered_df):,}")
col2.metric("Số quốc gia", filtered_df["country_name"].nunique())
col3.metric("PM2.5 trung bình (µg/m³)", f"{filtered_df['pm25_ugm3'].mean():.2f}")
col4.metric("Tỷ lệ bệnh hô hấp trung bình", f"{filtered_df['respiratory_disease_rate'].mean():.2f}")

st.info(
    "Các biểu đồ dự báo trong dashboard được xây dựng trên bộ đặc trưng AQM-Lag, "
    "tức là không chỉ sử dụng biến môi trường tại tuần hiện tại mà còn bổ sung "
    "độ trễ theo thời gian, trung bình trượt, mùa vụ, chỉ số tổng hợp và biến tương tác."
)

with st.expander("Giải thích bộ đặc trưng AQM-Lag được sử dụng trong mô hình dự báo", expanded=True):
    st.markdown("""
    ### AQM-Lag là gì?

    **AQM-Lag** là bộ đặc trưng đề xuất của đồ án, trong đó **AQM** là viết tắt của
    *Air Quality and Meteorology* — chất lượng không khí và khí tượng, còn **Lag**
    nhấn mạnh việc bổ sung các đặc trưng trễ theo thời gian.

    Thay vì chỉ sử dụng các biến tại tuần hiện tại, AQM-Lag kết hợp thêm:
    - Đặc trưng trễ 1, 2 và 4 tuần;
    - Trung bình trượt 4 tuần;
    - Mã hóa mùa vụ bằng `season_sin` và `season_cos`;
    - Chỉ số tổng hợp `aqm_stress_index`;
    - Các biến tương tác giữa ô nhiễm không khí và khí tượng.

    Cách thiết kế này phù hợp vì dữ liệu của đề tài có cấu trúc theo tuần và theo quốc gia.
    Ảnh hưởng của ô nhiễm không khí hoặc biến động khí tượng đến bệnh hô hấp có thể không
    xuất hiện ngay lập tức, mà có thể có độ trễ hoặc tác động tích lũy ngắn hạn.
    """)

    st.dataframe(AQM_LAG_FEATURE_TABLE, use_container_width=True, hide_index=True)

    st.markdown("""
    ### AQM Stress Index

    `aqm_stress_index` là chỉ số tổng hợp phản ánh mức căng thẳng môi trường hô hấp
    trong từng tuần. Chỉ số này kết hợp các yếu tố ô nhiễm không khí, bất thường nhiệt,
    nắng nóng và thời tiết cực đoan.
    """)

    st.dataframe(AQM_STRESS_WEIGHTS, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Vì sao chọn AQM-Lag cho các biểu đồ dự báo?

    Bộ đặc trưng này được chọn vì nó giúp mô hình nhìn bài toán theo nhiều lớp thông tin hơn:

    1. **Lớp hiện tại**: điều kiện môi trường và bối cảnh quốc gia trong tuần quan sát.
    2. **Lớp thời gian**: các giá trị trễ và trung bình trượt giúp phản ánh hiệu ứng trì hoãn.
    3. **Lớp mùa vụ**: `season_sin` và `season_cos` giúp mô hình nhận diện chu kỳ trong năm.
    4. **Lớp tổng hợp**: `aqm_stress_index` tóm tắt áp lực môi trường hô hấp.
    5. **Lớp tương tác**: các biến như `pm25_x_temperature` hoặc `precip_x_flood`
       biểu diễn tình huống nhiều yếu tố môi trường xuất hiện đồng thời.

    Vì vậy, AQM-Lag không chỉ phục vụ mục tiêu dự báo mà còn giúp giải thích vì sao
    các yếu tố môi trường có thể liên hệ với tỷ lệ bệnh hô hấp.
    """)

st.divider()

left, right = st.columns(2)

with left:
    annual_trend = filtered_df.groupby(["year", "region_vi"], as_index=False)["respiratory_disease_rate"].mean()
    fig_d01_trend = px.line(
        annual_trend,
        x="year",
        y="respiratory_disease_rate",
        color="region_vi",
        markers=True,
        title=CHART_TITLES["D01"],
        labels=LABELS,
    )
    st.plotly_chart(fig_d01_trend, use_container_width=True)

with right:
    fig_d02_pm25 = px.scatter(
        filtered_df,
        x="pm25_ugm3",
        y="respiratory_disease_rate",
        color="income_level_vi",
        hover_name="country_name",
        title=CHART_TITLES["D02"],
        labels=LABELS,
        opacity=0.6,
    )
    st.plotly_chart(fig_d02_pm25, use_container_width=True)

left, right = st.columns(2)

with left:
    region_summary = (
        filtered_df
        .groupby("region_vi", as_index=False)
        .agg(
            n_records=("respiratory_disease_rate", "size"),
            n_countries=("country_name", "nunique"),
            mean_resp=("respiratory_disease_rate", "mean"),
            mean_pm25=("pm25_ugm3", "mean"),
            mean_aqi=("air_quality_index", "mean"),
        )
        .sort_values("mean_resp", ascending=False)
    )
    fig_d03_region = px.bar(
        region_summary,
        x="region_vi",
        y="mean_resp",
        color="region_vi",
        hover_data={"n_records": True, "n_countries": True, "mean_pm25": ":.2f", "mean_aqi": ":.2f", "mean_resp": ":.2f"},
        title=CHART_TITLES["D03"],
        labels=LABELS,
    )
    fig_d03_region.update_layout(showlegend=False)
    st.plotly_chart(fig_d03_region, use_container_width=True)

with right:
    country_summary_filtered = (
        filtered_df
        .groupby(["country_code", "country_name", "region_vi", "income_level_vi"], as_index=False)
        .agg(mean_resp=("respiratory_disease_rate", "mean"), mean_pm25=("pm25_ugm3", "mean"), mean_aqi=("air_quality_index", "mean"))
        .sort_values("mean_resp", ascending=False)
    )
    fig_d04_country = px.bar(
        country_summary_filtered,
        x="country_name",
        y="mean_resp",
        color="region_vi",
        hover_data={"country_code": True, "income_level_vi": True, "mean_pm25": ":.2f", "mean_aqi": ":.2f", "mean_resp": ":.2f"},
        title=CHART_TITLES["D04"],
        labels=LABELS,
    )
    fig_d04_country.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_d04_country, use_container_width=True)

fig_d05_income = px.box(
    filtered_df,
    x="income_level_vi",
    y="respiratory_disease_rate",
    color="income_level_vi",
    title=CHART_TITLES["D05"],
    labels=LABELS,
)
st.plotly_chart(fig_d05_income, use_container_width=True)

st.divider()

st.subheader("Đánh giá dự báo trên tập kiểm tra")
st.caption("Các giá trị dự báo được tạo từ mô hình cuối cùng sử dụng bộ đặc trưng AQM_Lag_proposed.")

if filtered_pred.empty:
    st.info("Không có bản ghi dự báo phù hợp với bộ lọc hiện tại.")
else:
    mae = mean_absolute_error(filtered_pred["respiratory_disease_rate"], filtered_pred["prediction"])
    model_rmse = rmse(filtered_pred["respiratory_disease_rate"], filtered_pred["prediction"])

    if len(filtered_pred) >= 2:
        model_r2 = r2_score(filtered_pred["respiratory_disease_rate"], filtered_pred["prediction"])
        r2_text = f"{model_r2:.3f}"
    else:
        r2_text = "NA"

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{model_rmse:.3f}")
    c3.metric("R²", r2_text)

    left, right = st.columns(2)

    with left:
        fig_d06_pred_scatter = px.scatter(
            filtered_pred,
            x="respiratory_disease_rate",
            y="prediction",
            color="region_vi",
            hover_name="country_name",
            title=CHART_TITLES["D06"],
            labels=LABELS,
        )
        fig_d06_pred_scatter.add_shape(
            type="line",
            x0=filtered_pred["respiratory_disease_rate"].min(),
            y0=filtered_pred["respiratory_disease_rate"].min(),
            x1=filtered_pred["respiratory_disease_rate"].max(),
            y1=filtered_pred["respiratory_disease_rate"].max(),
            line=dict(dash="dash"),
        )
        fig_d06_pred_scatter.update_layout(xaxis_title="Giá trị thực tế", yaxis_title="Giá trị dự báo")
        st.plotly_chart(fig_d06_pred_scatter, use_container_width=True)

    with right:
        pred_trend = filtered_pred.groupby("date", as_index=False).agg(actual=("respiratory_disease_rate", "mean"), predicted=("prediction", "mean"))
        fig_d07_pred_trend = px.line(
            pred_trend.rename(columns={"actual": "Giá trị thực tế", "predicted": "Giá trị dự báo"}),
            x="date",
            y=["Giá trị thực tế", "Giá trị dự báo"],
            markers=True,
            title=CHART_TITLES["D07"],
            labels=LABELS,
        )
        st.plotly_chart(fig_d07_pred_trend, use_container_width=True)

    error_by_region = (
        filtered_pred
        .groupby("region_vi")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "MAE": mean_absolute_error(g["respiratory_disease_rate"], g["prediction"]),
            "RMSE": rmse(g["respiratory_disease_rate"], g["prediction"]),
            "mean_residual": g["residual"].mean(),
        }))
        .reset_index()
        .sort_values("RMSE", ascending=False)
    )

    fig_d08_error = px.bar(
        error_by_region,
        x="region_vi",
        y="RMSE",
        color="region_vi",
        hover_data={"n": True, "MAE": ":.3f", "RMSE": ":.3f", "mean_residual": ":.3f"},
        title=CHART_TITLES["D08"],
        labels=LABELS,
    )
    fig_d08_error.update_layout(showlegend=False)
    st.plotly_chart(fig_d08_error, use_container_width=True)

    with st.expander("Bảng sai số dự báo theo vùng"):
        st.dataframe(error_by_region.rename(columns=LABELS), use_container_width=True)

with st.expander("Bảng ánh xạ bảng điều khiển"):
    st.dataframe(DASHBOARD_MAPPING, use_container_width=True)
with st.expander("Ghi chú diễn giải kết quả AQM-Lag"):
    st.markdown("""
    Trong thực nghiệm của đồ án, AQM-Lag được xem là phần đề xuất chính về mặt thiết kế đặc trưng.
    Kết quả ablation cho thấy bộ đặc trưng này chưa tạo ra cải thiện RMSE rõ rệt so với bộ biến gốc.
    Tuy nhiên, trong phân tích độ quan trọng hoán vị, `aqm_stress_index` xuất hiện trong nhóm đặc trưng
    quan trọng nhất sau PM2.5. Điều này cho thấy AQM-Lag có giá trị ở khả năng tổ chức và giải thích
    không gian đặc trưng, thay vì chỉ được đánh giá bằng mức cải thiện sai số dự báo.
    """)


