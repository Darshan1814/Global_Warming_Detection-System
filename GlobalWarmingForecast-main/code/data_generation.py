import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Timeframe for the data
years = np.arange(2000, 2025)
months = np.arange(1, 13)
time_index = pd.date_range(start="2000-01-01", end="2024-12-31", freq="M")

# Function to simulate seasonal variation
def seasonal_variation(base, amplitude, period, phase_shift, noise_factor=0.1):
    return base + amplitude * np.sin(2 * np.pi * time_index.month / period + phase_shift) + \
           np.random.normal(0, noise_factor, len(time_index))

# Generate CO2, CH4, and N2O levels
co2_concentration = seasonal_variation(
    base=370,  # Base concentration in ppm
    amplitude=2,
    period=12,
    phase_shift=0.5,
    noise_factor=0.2
) + np.linspace(0, 50, len(time_index))  # Gradual increase

ch4_concentration = seasonal_variation(
    base=1800,  # Base concentration in ppb
    amplitude=10,
    period=12,
    phase_shift=1,
    noise_factor=5
) + np.linspace(0, 200, len(time_index))

n2o_concentration = seasonal_variation(
    base=310,  # Base concentration in ppb
    amplitude=1,
    period=12,
    phase_shift=1.5,
    noise_factor=0.5
) + np.linspace(0, 20, len(time_index))

# Generate global average temperature anomaly (°C)
temperature_anomaly = seasonal_variation(
    base=0.5,  # Baseline temperature anomaly in °C
    amplitude=0.1,
    period=12,
    phase_shift=0,
    noise_factor=0.05
) + np.linspace(0, 1.5, len(time_index))  # Gradual increase

# Generate renewable and fossil fuel energy usage patterns
renewable_energy_usage = np.clip(
    seasonal_variation(
        base=20,  # Starting at 20% of total energy usage
        amplitude=3,
        period=12,
        phase_shift=2,
        noise_factor=0.5
    ) + np.linspace(0, 30, len(time_index)), 0, 100
)  # Cannot exceed 100%

fossil_energy_usage = np.clip(
    100 - renewable_energy_usage + np.random.normal(0, 2, len(time_index)), 0, 100
)  # Remaining percentage for fossil fuels

# Generate forest area (in hectares, gradual decrease due to deforestation)
forest_area = np.clip(
    seasonal_variation(
        base=4_000_000,  # Base forest area in hectares
        amplitude=50_000,
        period=12,
        phase_shift=3,
        noise_factor=20_000
    ) - np.linspace(0, 300_000, len(time_index)), 0, None
)

# Generate annual natural disasters
natural_disasters = np.clip(
    seasonal_variation(
        base=5,  # Average of 5 disasters per year initially
        amplitude=1,
        period=12,
        phase_shift=1,
        noise_factor=0.2
    ) + np.linspace(0, 10, len(time_index)), 0, None
).astype(int)

# Generate glacier melting rates (km²)
glacier_melting_rate = seasonal_variation(
    base=50,  # Initial melting rate in km²
    amplitude=5,
    period=12,
    phase_shift=4,
    noise_factor=1
) + np.linspace(0, 100, len(time_index))

# Compile the data into a DataFrame
data = {
    "Year": time_index.year,
    "Month": time_index.month,
    "CO2_Concentration_ppm": co2_concentration,
    "CH4_Concentration_ppb": ch4_concentration,
    "N2O_Concentration_ppb": n2o_concentration,
    "Temperature_Anomaly_C": temperature_anomaly,
    "Renewable_Energy_Usage_Percentage": renewable_energy_usage,
    "Fossil_Energy_Usage_Percentage": fossil_energy_usage,
    "Forest_Area_Hectares": forest_area,
    "Natural_Disasters_Count": natural_disasters,
    "Glacier_Melting_Rate_km2": glacier_melting_rate
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("global_warming_sim_dataset.csv", index=False)

print("Dataset successfully created and saved as 'global_warming_sim_dataset.csv'.")
