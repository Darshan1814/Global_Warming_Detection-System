import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Load the cleaned dataset
df = pd.read_csv("fully_cleaned_global_warming_sim_dataset.csv")

# Create report directory if it doesn't exist
os.makedirs("report", exist_ok=True)

# 1. Advanced Correlation Heatmap with Clustering
def advanced_correlation_heatmap(data, output_path="report/advanced_correlation_heatmap.png"):
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.clustermap(corr, annot=True, fmt=".2f", cmap="coolwarm", figsize=(12, 10), cbar_kws={'label': 'Correlation'})
    plt.title("Advanced Correlation Heatmap with Clustering", pad=20)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# 2. Interactive 3D Scatter Plot
def interactive_3d_scatter(data, x, y, z, color, output_path="report/interactive_3d_scatter.html"):
    fig = px.scatter_3d(
        data, x=x, y=y, z=z, color=color, 
        title="3D Scatter Plot of Greenhouse Gases vs Temperature Anomaly",
        template="plotly_dark"
    )
    fig.write_html(output_path)
    print(f"Saved: {output_path}")

# 3. Time Series Decomposition
def time_series_decomposition(data, column, output_path="report/time_series_decomposition.png"):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(data[column], model="additive", period=12)
    result.plot()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# 4. Line Chart with Rolling Averages
def line_chart_with_rolling_averages(data, column, window=12, output_path="report/rolling_average_chart.png"):
    plt.figure(figsize=(12, 6))
    data[f"{column}_Rolling_Avg"] = data[column].rolling(window=window).mean()
    plt.plot(data["Year"], data[column], label=f"Actual {column}", linestyle="--", color="blue")
    plt.plot(data["Year"], data[f"{column}_Rolling_Avg"], label=f"{window}-Month Rolling Average", color="red")
    plt.title(f"{column} with {window}-Month Rolling Average")
    plt.xlabel("Year")
    plt.ylabel(column)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# 5. Heatmap of Annual Temperature Anomalies
def annual_temperature_heatmap(data, year_col="Year", temp_col="Temperature_Anomaly_C", output_path="report/annual_temperature_heatmap.png"):
    pivot = data.pivot_table(values=temp_col, index=year_col, aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Temperature Anomaly (°C)"})
    plt.title("Annual Temperature Anomaly Heatmap")
    plt.xlabel("Year")
    plt.ylabel("Average Anomaly")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# 6. Interactive Line Chart with Plotly
def interactive_line_chart(data, x, y, color, output_path="report/interactive_line_chart.html"):
    fig = px.line(data, x=x, y=y, color=color, title="Interactive Line Chart of Temperature Anomalies", template="plotly_white")
    fig.write_html(output_path)
    print(f"Saved: {output_path}")

# Call functions to generate advanced visualizations
advanced_correlation_heatmap(df)
interactive_3d_scatter(
    df, x="CO2_Concentration_ppm", y="CH4_Concentration_ppb", z="Temperature_Anomaly_C", color="Year"
)
time_series_decomposition(df, column="Temperature_Anomaly_C")
line_chart_with_rolling_averages(df, column="Temperature_Anomaly_C")
annual_temperature_heatmap(df)
interactive_line_chart(df, x="Year", y="Temperature_Anomaly_C", color="Year")

print("All advanced visualizations generated and saved in the 'report/' directory.")
