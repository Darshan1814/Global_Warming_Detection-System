import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from prophet import Prophet
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="🌌 Advanced Global Warming Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #f5f5f5;
    } 
    .stApp {
        background-color: #121212;
    }
    h1, h2, h3, h4 {
        color: #f5f5f5;
    }
    .stSidebar {
        background-color: #1f1f1f;
    }
    .stSidebar .st-radio {
        color: #f5f5f5;
    }
    .css-1aumxhk {
        background-color: #1f1f1f;
    }
    .stDataFrame {
        border: 1px solid #f5f5f5;
    }
    .css-1ekf893 {
        color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.title("✨ Navigation Menu")
menu_options = [
    "🏠 Home",
    "📊 Scenario Analysis",
    "📈 Advanced Visualizations",
    "🔮 Time Series Forecast (ARIMA & Prophet)",
    "📥 Upload & Analyze Data",
    "📋 Generate Reports",
    "ℹ️ About"
]
menu_choice = st.sidebar.radio("Navigate", menu_options)

# Load the dataset
@st.cache_data
def load_data(file_path="fully_cleaned_global_warming_sim_dataset.csv"):
    return pd.read_csv(file_path)

df = load_data()

# Function to generate custom scenario predictions
def generate_scenario(df, co2_change, ch4_change, n2o_change):
    scenario_df = df.copy()
    scenario_df["CO2_Concentration_ppm"] += co2_change
    scenario_df["CH4_Concentration_ppb"] += ch4_change
    scenario_df["N2O_Concentration_ppb"] += n2o_change
    X = scenario_df[["CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb"]]
    y = scenario_df["Temperature_Anomaly_C"]
    model = LinearRegression()
    model.fit(X, y)
    scenario_df["Predicted_Temperature_Anomaly_C"] = model.predict(X)
    return scenario_df

# Home Page
if menu_choice == "🏠 Home":
    st.title("🌌 Advanced Global Warming Analysis")
    st.markdown("""
    <h3>Welcome to the Advanced Global Warming Analysis Tool</h3>
    <p>This platform allows you to explore and analyze climate data interactively with advanced tools and visualizations.</p>
    """, unsafe_allow_html=True)
    st.image("https://cdn.mos.cms.futurecdn.net/6ZW3VY5dZJbYSD7FeAsKe6-1200-80.jpg", use_container_width=True)
    st.write("### Dataset Preview")
    st.dataframe(df.head(10))

# Scenario Analysis
elif menu_choice == "📊 Scenario Analysis":
    st.header("📊 Advanced Scenario Analysis")

    st.write("### Customize Greenhouse Gas Changes:")
    co2_change = st.slider("CO2 Change (ppm)", -10.0, 10.0, 0.0, step=0.5)
    ch4_change = st.slider("CH4 Change (ppb)", -50.0, 50.0, 0.0, step=5.0)
    n2o_change = st.slider("N2O Change (ppb)", -5.0, 5.0, 0.0, step=0.5)

    scenario_df = generate_scenario(df, co2_change, ch4_change, n2o_change)

    st.write("### Scenario Results")
    fig = px.line(
        scenario_df,
        x="Year",
        y=["Temperature_Anomaly_C", "Predicted_Temperature_Anomaly_C"],
        labels={"value": "Temperature Anomaly (°C)", "variable": "Scenario"},
        title="Scenario Analysis of Temperature Anomalies"
    )
    st.plotly_chart(fig)

# Advanced Visualizations
elif menu_choice == "📈 Advanced Visualizations":
    st.header("📈 Advanced Visualizations")

    # Altair Scatter Plot
    st.write("### Altair Interactive Scatter Plot")
    alt_chart = alt.Chart(df).mark_circle(size=60).encode(
        x='CO2_Concentration_ppm',
        y='Temperature_Anomaly_C',
        color='Year:N',
        tooltip=['Year', 'CO2_Concentration_ppm', 'Temperature_Anomaly_C']
    ).interactive()
    st.altair_chart(alt_chart, use_container_width=True)

    # Heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

# Time Series Forecast
elif menu_choice == "🔮 Time Series Forecast (ARIMA & Prophet)":
    st.markdown("""
    <style>
    .forecast-container {
        background: linear-gradient(to bottom, #1c1c1c, #121212);
        color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.8);
    }
    .forecast-container h2 {
        font-size: 2.5rem;
        color: #00ff8a;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .forecast-container p {
        font-size: 1.2rem;
        color: #dcdcdc;
        line-height: 1.8;
    }
    .forecast-chart {
        margin-top: 20px;
        background: #1f1f1f;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        font-size: 1rem;
    }
    .analyze-button {
        background: #00d4ff;
        color: #000000;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
    }
    .analyze-button:hover {
        transform: scale(1.1);
        background: #00ff8a;
    }
    </style>
    """, unsafe_allow_html=True)

    # Forecasting Başlığı
    st.markdown("""
    <div class="forecast-container">
        <h2>🔮 Time Series Forecasting</h2>
        <p>
            Analyze and predict future temperature anomalies using ARIMA and Prophet models.<br>
            These forecasts are designed to help visualize long-term climate trends interactively.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ARIMA Forecast
    st.write("### ARIMA Forecast")
    arima_model = ARIMA(df["Temperature_Anomaly_C"], order=(2, 1, 2))
    arima_result = arima_model.fit()
    forecast_years = 50
    forecast_index = pd.date_range(start="2025", periods=forecast_years, freq="YE")
    forecast = arima_result.forecast(steps=forecast_years)

    # ARIMA Grafiği
    st.markdown("""
    <div class="forecast-chart">
    """, unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Temperature_Anomaly_C"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_index.year, y=forecast, mode="lines", name="Forecast"))
    fig.update_layout(
        title="ARIMA Forecast (Next 50 Years)",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prophet Forecast
    st.write("### Prophet Forecast")
    prophet_df = df.rename(columns={"Year": "ds", "Temperature_Anomaly_C": "y"})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=50, freq="YE")
    forecast = prophet_model.predict(future)

    # Prophet Grafiği
    st.markdown("""
    <div class="forecast-chart">
    """, unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Actual"))
    fig2.add_trace(go.Scatter(x=future["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    fig2.update_layout(
        title="Prophet Forecast (Next 50 Years)",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        template="plotly_dark"
    )
    st.plotly_chart(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    # Özet ve Gelecek Planları
    st.markdown("""
    <div class="forecast-container">
        <p>
            These time series models provide valuable insights into long-term climate trends.<br>
            Further improvements, including model tuning and additional forecasting metrics, are planned for future releases.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Upload and Analyze Data
elif menu_choice == "📥 Upload & Analyze Data":
    st.markdown("""
    <style>
    .upload-container {
        background: linear-gradient(to bottom, #1c1c1c, #121212);
        color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.8);
    }
    .upload-container h2 {
        font-size: 2.5rem;
        color: #00ff8a;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .upload-container p {
        font-size: 1.2rem;
        color: #dcdcdc;
        line-height: 1.8;
    }
    .data-preview {
        margin-top: 20px;
        background: #1f1f1f;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        font-size: 1rem;
        text-align: left;
        overflow-x: auto;
    }
    .analyze-button {
        background: #00d4ff;
        color: #000000;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
    }
    .analyze-button:hover {
        transform: scale(1.1);
        background: #00ff8a;
    }
    .no-data-warning {
        color: #ff6666;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Veri Yükleme Başlığı
    st.markdown("""
    <div class="upload-container">
        <h2>📥 Upload and Analyze Your Data</h2>
        <p>
            Upload your CSV dataset to perform interactive analysis.  
            The uploaded dataset will be previewed, and summary statistics will be generated for further insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dosya Yükleme Aracı
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file:
        # Yüklenen Dosya ile Çalışma
        user_df = pd.read_csv(uploaded_file)

        # Veri Seti Önizlemesi
        st.write("### Uploaded Dataset Preview")
        st.markdown("""
        <div class="data-preview">
        """, unsafe_allow_html=True)
        st.dataframe(user_df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        # Veri Seti Tanımı
        st.write("### Dataset Description")
        st.markdown("""
        <div class="data-preview">
        """, unsafe_allow_html=True)
        st.write(user_df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

        # Ek Analiz Seçenekleri
        st.write("### Explore Data Further")
        analyze_choice = st.selectbox(
            "Choose an analysis option:",
            ["Correlation Heatmap", "Histogram", "Scatter Plot"]
        )

        # Korelasyon Isı Haritası
        if analyze_choice == "Correlation Heatmap":
            st.write("#### Correlation Heatmap")
            corr = user_df.corr()
            st.write("Correlation Matrix:", corr)
            st.markdown("""
            <div class="data-preview">
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        # Histogram
        elif analyze_choice == "Histogram":
            st.write("#### Histogram")
            column_to_plot = st.selectbox("Choose a column for the histogram:", user_df.columns)
            bins = st.slider("Number of bins:", 5, 50, 20)
            st.markdown("""
            <div class="data-preview">
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            user_df[column_to_plot].hist(bins=bins, ax=ax, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogram of {column_to_plot}")
            ax.set_xlabel(column_to_plot)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        # Scatter Plot
        elif analyze_choice == "Scatter Plot":
            st.write("#### Scatter Plot")
            x_col = st.selectbox("Select X-axis column:", user_df.columns)
            y_col = st.selectbox("Select Y-axis column:", user_df.columns)
            st.markdown("""
            <div class="data-preview">
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            user_df.plot.scatter(x=x_col, y=y_col, ax=ax, color="orange")
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="no-data-warning">
            Please upload a CSV file to begin analysis.
        </div>
        """, unsafe_allow_html=True)

# Generate Reports
elif menu_choice == "📋 Generate Reports":
    st.markdown("""
    <style>
    .report-container {
        background: linear-gradient(to bottom, #1c1c1c, #121212);
        color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.8);
    }
    .report-container h2 {
        font-size: 2.5rem;
        color: #00ff8a;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .report-container p {
        font-size: 1.2rem;
        color: #dcdcdc;
        line-height: 1.8;
    }
    .report-button {
        background: #00d4ff;
        color: #000000;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
    }
    .report-button:hover {
        transform: scale(1.1);
        background: #00ff8a;
    }
    </style>
    """, unsafe_allow_html=True)

    # Raporlama Bölümü
    st.markdown("""
    <div class="report-container">
        <h2>📋 Generate Reports</h2>
        <p>
            Export the dataset in various formats, including CSV, Excel, and PDF.<br>
            Download and analyze the reports locally for deeper insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Veri Setini CSV Formatında İndirme
    buffer_csv = BytesIO()
    df.to_csv(buffer_csv, index=False)
    buffer_csv.seek(0)

    # Excel İndir
    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="GlobalWarmingData")
    buffer_excel.seek(0)

    # İndirilebilir Seçenekler
    st.write("### Download Options")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download CSV",
            data=buffer_csv,
            file_name="global_warming_analysis.csv",
            mime="text/csv",
            help="Download the dataset in CSV format.",
            key="csv_download"
        )
    with col2:
        st.download_button(
            label="Download Excel",
            data=buffer_excel,
            file_name="global_warming_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the dataset in Excel format.",
            key="excel_download"
        )

    # Ek Raporlama ve Bilgilendirme
    st.markdown("""
    <div class="report-container">
        <p>
            Download your analysis in CSV or Excel format for detailed insights and further analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# About Page
elif menu_choice == "ℹ️ About":
    st.markdown("""
    <style>
    /* Enhanced General Style */
    body {
        background: linear-gradient(135deg, #1a1a1a, #121212);
        color: #f5f5f5;
        font-family: 'Roboto', sans-serif;
    }
    .about-container {
        background: linear-gradient(135deg, #212121, #1a1a1a);
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.7);
        color: #ffffff;
        max-width: 900px;
        margin: auto;
        text-align: center;
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .about-container h2 {
        color: #00c8ff;
        font-size: 32px;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.8);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 5px #00c8ff, 0 0 10px #00c8ff; }
        to { text-shadow: 0 0 10px #00c8ff, 0 0 20px #00c8ff, 0 0 30px #00c8ff; }
    }
    .about-container p {
        color: #cccccc;
        line-height: 1.8;
        margin-bottom: 20px;
        font-size: 16px;
    }
    .tech-box {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 15px;
        margin: 20px 0;
    }
    .tech-item {
        background: #1f1f1f;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #444;
        text-align: center;
        font-size: 14px;
        color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        font-weight: bold;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .tech-item:hover {
        background: #00c8ff;
        color: #1a1a1a;
        transform: scale(1.1);
    }
    .developer-box {
        margin-top: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #2a2a2a, #1f1f1f);
        border-radius: 10px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.6);
        animation: slideIn 1s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .developer-box h3 {
        color: #00ff8a;
        margin-bottom: 15px;
        font-size: 24px;
        text-shadow: 0 0 10px rgba(0, 255, 138, 0.5);
    }
    .developer-box p {
        color: #e0e0e0;
        margin: 5px 0;
    }
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 30px 0;
        padding: 20px;
    }
    .team-card {
        background: linear-gradient(145deg, #2a2a2a, #1f1f1f);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    .team-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(0, 200, 255, 0.1), transparent);
        transform: translateX(-100%);
        transition: 0.5s;
    }
    .team-card:hover::before {
        transform: translateX(100%);
    }
    .team-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 25px rgba(0, 200, 255, 0.2);
    }
    .team-card h4 {
        color: #00c8ff;
        font-size: 1.5em;
        margin: 10px 0;
        text-shadow: 0 0 10px rgba(0, 200, 255, 0.3);
    }
    .team-card p {
        color: #e0e0e0;
        margin: 5px 0;
        font-size: 1.1em;
    }
    .team-card .prn {
        color: #00ff8a;
        font-family: monospace;
        font-size: 0.9em;
        padding: 5px 10px;
        background: rgba(0, 255, 138, 0.1);
        border-radius: 5px;
        display: inline-block;
        margin-top: 10px;
    }
    .team-card .role {
        color: #ffd700;
        font-size: 0.9em;
        margin-top: 5px;
    }
    .team-section {
        animation: fadeInUp 1s ease-out;
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #cccccc;
        font-size: 18px;
        padding-top: 20px;
        border-top: 1px solid #444;
        animation: fadeIn 2s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-container">
        <h2>🌍 About Eco-Guardians</h2>
        <div class="mission-statement">
            <p>We are a team of passionate Computer Engineering students dedicated to creating innovative solutions for a sustainable future. Our mission is to leverage technology to combat climate change and raise awareness about global warming.</p>
        </div>
        <div class="team-section">
            <h3>👥 Meet Our Team</h3>
            <div class="team-grid">
                <div class="team-card">
                    <h4>Darshan</h4>
                    <p class="role">Team Lead</p>
                    <p class="prn">202301040255</p>
                </div>
                <div class="team-card">
                    <h4>Lucky</h4>
                    <p class="role">Data Analyst</p>
                    <p class="prn">202301040253</p>
                </div>
                <div class="team-card">
                    <h4>Arya</h4>
                    <p class="role">Visualization Expert</p>
                    <p class="prn">202301040265</p>
                </div>
                <div class="team-card">
                    <h4>Atharva</h4>
                    <p class="role">Backend Developer</p>
                    <p class="prn">202301040254</p>
                </div>
            </div>
        </div>
        <div class="developer-box">
            <h3>🎓 Education</h3>
            <p>Computer Engineering (MITAOE)</p>
            <p>📍 Pune, Alandi 412105</p>
        </div>
        <h3>🎯 Our Vision</h3>
        <ul style='text-align: left; margin-left: 40px;'>
            <li>Create an intuitive platform for analyzing global warming trends</li>
            <li>Enable users to simulate different environmental scenarios</li>
            <li>Provide advanced visualizations and dynamic forecasting tools</li>
            <li>Make climate science accessible to everyone</li>
        </ul>
        <h3>🛠️ Technologies We Use</h3>
        <div class='tech-box'>
            <div class='tech-item'>Python</div>
            <div class='tech-item'>Streamlit</div>
            <div class='tech-item'>Plotly</div>
            <div class='tech-item'>Altair</div>
            <div class='tech-item'>Matplotlib</div>
            <div class='tech-item'>Seaborn</div>
            <div class='tech-item'>Prophet</div>
            <div class='tech-item'>ARIMA</div>
        </div>
        <h3>🚀 Future Roadmap</h3>
        <ul style='text-align: left; margin-left: 40px;'>
            <li>Integration with live environmental data APIs (NASA, NOAA)</li>
            <li>Advanced forecasting models using deep learning</li>
            <li>Interactive global map with real-time data visualization</li>
            <li>Enhanced report generation capabilities</li>
        </ul>
        <div class='footer'>
            <p>🌱 Together, We Can Make a Difference!</p>
            <p>Designed with ❤️ by <span style='color:#00c8ff;font-weight:bold;'>Team Eco-Guardians</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
