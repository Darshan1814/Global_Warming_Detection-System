import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import os

# Load the cleaned dataset
df = pd.read_csv("fully_cleaned_global_warming_sim_dataset.csv")

# Feature Selection
X = df[["CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb"]]
y = df["Temperature_Anomaly_C"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  R² Score: {r2:.4f}")

# Advanced Statistical Analysis
print("\nAdvanced Statistical Analysis:")
corr_matrix = df.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Heatmap of Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.savefig("report/correlation_matrix_heatmap.png")
plt.close()
print("Saved: report/correlation_matrix_heatmap.png")

# Function to simulate scenarios
def simulate_scenario(model, df, scenario_name, co2_change, ch4_change, n2o_change):
    scenario_data = df.copy()
    scenario_data["CO2_Concentration_ppm"] += co2_change
    scenario_data["CH4_Concentration_ppb"] += ch4_change
    scenario_data["N2O_Concentration_ppb"] += n2o_change
    scenario_data["Predicted_Temperature_Anomaly_C"] = model.predict(
        scenario_data[["CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb"]]
    )
    return scenario_data

# Define scenarios
scenarios = {
    "No_Policy_Change": (2, 10, 1),  # Gradual increase in emissions
    "Carbon_Neutral_2050": (-1, -5, -0.5),  # Slow decrease in emissions
    "Global_Collaboration": (-2, -10, -1),  # Rapid decrease in emissions
    "Extreme_Mitigation": (-3, -15, -1.5),  # Extremely rapid decrease in emissions
    "Worst_Case_Scenario": (3, 15, 1.5)  # Extremely rapid increase in emissions
}

# Simulate scenarios
scenario_results = {}
for scenario_name, (co2_change, ch4_change, n2o_change) in scenarios.items():
    scenario_results[scenario_name] = simulate_scenario(model, df, scenario_name, co2_change, ch4_change, n2o_change)

# Plot and save results for scenarios
for scenario_name, scenario_data in scenario_results.items():
    plt.figure(figsize=(12, 6))
    plt.plot(df["Year"], df["Temperature_Anomaly_C"], label="Actual", linestyle="--", color="blue")
    plt.plot(scenario_data["Year"], scenario_data["Predicted_Temperature_Anomaly_C"], label=f"{scenario_name}", color="red")
    plt.title(f"Scenario: {scenario_name.replace('_', ' ')}")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.legend()
    plt.grid()
    plt.savefig(f"report/{scenario_name}_Temperature_Anomaly.png")
    plt.close()
    print(f"Saved: report/{scenario_name}_Temperature_Anomaly.png")

# Advanced Time Series Forecasting with ARIMA
print("\nTime Series Forecasting with ARIMA:")
arima_model = ARIMA(df["Temperature_Anomaly_C"], order=(2, 1, 2))
arima_result = arima_model.fit()

# Forecast for the next 50 years
forecast_years = 50
forecast_index = pd.date_range(start="2025", periods=forecast_years, freq="Y")
forecast = arima_result.forecast(steps=forecast_years)

# Plot ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Temperature_Anomaly_C"], label="Actual Data", color="blue")
plt.plot(forecast_index.year, forecast, label="Forecast", color="orange", linestyle="--")
plt.title("ARIMA Forecast for Temperature Anomaly (Next 50 Years)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.grid()
plt.savefig("report/temperature_anomaly_forecast.png")
plt.close()
print("Saved: report/temperature_anomaly_forecast.png")

# Scenario Comparisons
plt.figure(figsize=(12, 8))
for scenario_name, scenario_data in scenario_results.items():
    plt.plot(
        scenario_data["Year"], 
        scenario_data["Predicted_Temperature_Anomaly_C"], 
        label=f"{scenario_name.replace('_', ' ')}"
    )
plt.plot(df["Year"], df["Temperature_Anomaly_C"], label="Actual", linestyle="--", color="black")
plt.title("Scenario Comparisons for Temperature Anomaly")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.grid()
plt.savefig("report/scenario_comparisons.png")
plt.close()
print("Saved: report/scenario_comparisons.png")

# Generate summary table for scenarios
scenario_summary = pd.DataFrame({
    "Scenario": list(scenario_results.keys()),
    "Average_Temperature_Anomaly": [
        scenario_data["Predicted_Temperature_Anomaly_C"].mean() for scenario_data in scenario_results.values()
    ]
})
scenario_summary.to_csv("report/scenario_summary.csv", index=False)
print("Saved: report/scenario_summary.csv")

print("\nAdvanced analysis, forecasting, and visualizations saved in the 'report/' directory.")
