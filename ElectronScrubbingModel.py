import numpy as np
import matplotlib.pyplot as plt

# Constants
e_charge = 1.602e-19  # Elementary charge (Coulombs)
R_air = 287.0         # Specific gas constant for air (J/kg·K)

# --------------------------
# Core Functions
# --------------------------
def calculate_beam_power(voltage_MeV, current_mA):
    """
    Calculate electron beam power in kilowatts (kW).
    :param voltage_MeV: Electron energy in megaelectronvolts (MeV)
    :param current_mA: Beam current in milliamperes (mA)
    """
    current_A = current_mA / 1000  # Convert mA to Amperes
    energy_per_electron_J = voltage_MeV * 1e6 * e_charge  # Convert MeV to Joules
    power_W = energy_per_electron_J * current_A / e_charge  # P = (Energy/e⁻) * (e⁻/s)
    return power_W / 1000  # Convert to kW

def gas_density(temp_C):
    """
    Calculate exhaust gas density (kg/m³) using ideal gas law.
    :param temp_C: Exhaust temperature (°C)
    """
    temp_K = temp_C + 273.15
    pressure = 101325  # Atmospheric pressure (Pa)
    return pressure / (R_air * temp_K)

def calculate_dose(beam_power_kW, gas_flow_rate_kg_per_s, efficiency=0.6):
    """
    Calculate energy dose (kGy) absorbed by flue gas.
    :param efficiency: Fraction of beam energy absorbed by gas (0–1)
    """
    energy_absorbed_kJ = beam_power_kW * efficiency  # kW = kJ/s
    return energy_absorbed_kJ / gas_flow_rate_kg_per_s  # kGy = kJ/kg

def removal_efficiency(k, dose, concentration):
    """
    Calculate pollutant removal efficiency using exponential decay model.
    """
    return 1 - np.exp(-k * dose * concentration)

# --------------------------
# Input Parameters (Realistic Values)
# --------------------------
# Ship Engine Parameters
exhaust_flow_rate_m3_per_s = 300  # Typical for large cargo ships (100–500 m³/s)
exhaust_temp_C = 300              # Exhaust gas temperature (°C)
SOx_ppm = 2000                    # SOx concentration (ppm) for 3.5% sulfur HFO
NOx_ppm = 1500                    # NOx concentration (ppm) for marine diesel engines

# E-Beam System Parameters
voltage_MeV = 0.8     # Electron energy (0.5–1.5 MeV typical)
current_mA = 200      # Beam current (100–300 mA typical)
humidity = 0.15       # Humidity ratio (10–20% optimal)
NH3_ppm = 2000        # Ammonia concentration (ppm)

# Reaction Kinetics (Adjusted from literature)
k_SOx = 8.0  # Reaction rate constant for SOx (kGy⁻¹)
k_NOx = 3.0  # Reaction rate constant for NOx (kGy⁻¹)

# --------------------------
# Calculations
# --------------------------
# Step 1: Gas properties
gas_density_kg_per_m3 = gas_density(exhaust_temp_C)
gas_flow_rate_kg_per_s = exhaust_flow_rate_m3_per_s * gas_density_kg_per_m3

# Step 2: Beam power and dose
beam_power_kW = calculate_beam_power(voltage_MeV, current_mA)
dose = calculate_dose(beam_power_kW, gas_flow_rate_kg_per_s, efficiency=0.6)

# Step 3: Removal efficiencies
eta_SOx = removal_efficiency(k_SOx, dose, humidity)
eta_NOx = removal_efficiency(k_NOx, dose, NH3_ppm / 1e6)  # Convert ppm to fraction

# Step 4: Byproduct yield (kg/h)
m_SOx_removed = (SOx_ppm * 1e-6) * gas_flow_rate_kg_per_s * 3600 * eta_SOx
m_NOx_removed = (NOx_ppm * 1e-6) * gas_flow_rate_kg_per_s * 3600 * eta_NOx

m_ammonium_sulfate = m_SOx_removed * (132.14 / 64.06)  # (NH4)2SO4 (132.14 g/mol) / SO2 (64.06 g/mol)
m_ammonium_nitrate = m_NOx_removed * (80.04 / 46.01)   # NH4NO3 (80.04 g/mol) / NO2 (46.01 g/mol)

# --------------------------
# Results Output
# --------------------------
print(f"[1] Beam Power: {beam_power_kW:.1f} kW")
print(f"[2] Dose: {dose:.1f} kGy")
print(f"[3] SOx Removal Efficiency: {eta_SOx * 100:.1f}%")
print(f"[4] NOx Removal Efficiency: {eta_NOx * 100:.1f}%")
print(f"[5] Ammonium Sulfate Yield: {m_ammonium_sulfate:.1f} kg/h")
print(f"[6] Ammonium Nitrate Yield: {m_ammonium_nitrate:.1f} kg/h")

# --------------------------
# Energy Penalty Analysis
# --------------------------
def energy_penalty(beam_power_kW, ship_power_MW=15, fuel_specific_energy=42.7, generator_efficiency=0.5):
    """
    Calculate additional fuel consumption and CO₂ emissions from e-beam operation.
    """
    fuel_kg_per_h = (beam_power_kW * 3.6) / (fuel_specific_energy * generator_efficiency)  # 3.6 converts kW to MJ/h
    co2_kg_per_h = fuel_kg_per_h * 3.206  # CO₂ emission factor (kg CO₂/kg fuel)
    return fuel_kg_per_h, co2_kg_per_h

fuel_use, co2_emissions = energy_penalty(beam_power_kW)
print(f"\n[7] Additional Fuel Use: {fuel_use:.1f} kg/h")
print(f"[8] Additional CO₂ Emissions: {co2_emissions:.1f} kg/h")

def run_sensitivity_analysis():
    # Baseline parameters
    params = {
        "voltage_MeV": 0.8,
        "current_mA": 200,
        "exhaust_flow_m3_per_s": 300,
        "exhaust_temp_C": 300,
        "humidity": 0.15,
        "NH3_ppm": 2000,
        "k_SOx": 8.0,
        "k_NOx": 3.0
    }

    # Focus on critical parameters
    variables = {
        "Beam Current (mA)": {
            "param": "current_mA",
            "range": np.linspace(100, 300, 10),
            "xlabel": "Beam Current (mA)"
        },
        "Exhaust Flow Rate (m³/s)": {
            "param": "exhaust_flow_m3_per_s",
            "range": np.linspace(100, 500, 10),
            "xlabel": "Exhaust Flow Rate (m³/s)"
        },
        "Humidity (%)": {
            "param": "humidity",
            "range": np.linspace(0.05, 0.25, 10),
            "xlabel": "Humidity (fraction)"
        }
    }

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)

    for idx, (title, var) in enumerate(variables.items()):
        param_name = var["param"]
        param_values = var["range"]
        xlabel = var["xlabel"]
        
        eta_SOx_list = []
        eta_NOx_list = []
        dose_list = []
        
        for val in param_values:
            # Update parameter
            local_params = params.copy()
            local_params[param_name] = val
            
            # Recalculate
            gas_dens = gas_density(local_params["exhaust_temp_C"])
            gas_flow_kg_s = local_params["exhaust_flow_m3_per_s"] * gas_dens
            beam_power = calculate_beam_power(local_params["voltage_MeV"], local_params["current_mA"])
            dose = calculate_dose(beam_power, gas_flow_kg_s)
            
            eta_SOx = removal_efficiency(local_params["k_SOx"], dose, local_params["humidity"])
            eta_NOx = removal_efficiency(local_params["k_NOx"], dose, local_params["NH3_ppm"] / 1e6)
            
            eta_SOx_list.append(eta_SOx * 100)
            eta_NOx_list.append(eta_NOx * 100)
            dose_list.append(dose)
        
        # Plot efficiency and dose trends
        ax = axes[idx]
        ax.plot(param_values, eta_SOx_list, 'b-o', label="SOx Removal", linewidth=2)
        ax.plot(param_values, eta_NOx_list, 'r--s', label="NOx Removal", linewidth=2)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Removal Efficiency (%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Annotate critical thresholds
        if param_name == "current_mA":
            ax.axvline(200, color="gray", linestyle=":", label="Baseline (200 mA)")
            ax.annotate("Diminishing returns >200 mA", xy=(220, 85), fontsize=10, color="blue")
        elif param_name == "exhaust_flow_m3_per_s":
            ax.axvline(300, color="gray", linestyle=":", label="Baseline (300 m³/s)")
            ax.annotate("High flow → Low dose", xy=(400, 40), fontsize=10, color="red")
        elif param_name == "humidity":
            ax.axvline(0.15, color="gray", linestyle=":", label="Baseline (15%)")
            ax.annotate("Optimal humidity: 10–20%", xy=(0.18, 70), fontsize=10, color="purple")
        
        # Twin axis for dose
        ax2 = ax.twinx()
        ax2.plot(param_values, dose_list, 'g-^', label="Dose (kGy)", alpha=0.7)
        ax2.set_ylabel("Dose (kGy)", fontsize=12)
        ax2.legend(loc="upper right" if idx == 2 else "lower right")

        ax.legend(loc="upper left")

    plt.suptitle("Key Sensitivity Plots: Beam Current, Exhaust Flow, and Humidity", y=1.05, fontsize=14)
    plt.show()

run_sensitivity_analysis()