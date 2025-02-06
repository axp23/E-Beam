import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from matplotlib.colors import LinearSegmentedColormap

# ========================
# Working on 2/5/25
# Fix Plots on NOx/SOx efficiency
# ========================

# ========================
# CONSTANTS & PARAMETERS
# ========================
e_charge = 1.602e-19  # Coulombs
R_air = 287.0         # J/kg·K
k_SOx = 8.0           # kGy⁻¹
k_NOx = 3.0           # kGy⁻¹

BASE_PARAMS = {
    'voltage_MeV': 0.8,
    'current_mA': 200,
    'exhaust_flow_m3_per_s': 300,
    'exhaust_temp_C': 300,
    'humidity': 0.15,
    'NH3_ppm': 2000,
    'reactor_length': 5.0,
    'reactor_width': 2.0,
    'beam_sigma': 0.5,
    'gas_velocity': 5.0
}

# ========================
# CORE FUNCTIONS (VALIDATED)
# ========================
def calculate_beam_power(voltage_MeV, current_mA):
    """Calculate beam power in kW with proper unit conversions"""
    current_A = current_mA / 1000  # mA to A
    energy_per_electron_J = voltage_MeV * 1e6 * e_charge  # MeV to J
    return (energy_per_electron_J * current_A) / 1000  # W to kW

def gas_density(temp_C):
    """Calculate gas density using ideal gas law"""
    return 101325 / (R_air * (temp_C + 273.15))

def calculate_dose(beam_power_kW, mass_flow_kg_s, efficiency=0.6):
    """Calculate energy dose in kGy (kJ/kg)"""
    return (beam_power_kW * efficiency) / mass_flow_kg_s

def removal_efficiency(k, dose, concentration):
    """Exponential removal model"""
    return 1 - np.exp(-k * dose * concentration)

# ========================
# 2D REACTOR MODEL 
# ========================
def run_2D_simulation(params):
    """2D simulation with automatic beam power calculation"""
    # Calculate derived parameters
    beam_power = calculate_beam_power(params['voltage_MeV'], params['current_mA'])
    gas_dens = gas_density(params['exhaust_temp_C'])
    mass_flow = params['exhaust_flow_m3_per_s'] * gas_dens  # kg/s
    
    # Create spatial grid
    X, Y = np.meshgrid(
        np.linspace(0, params['reactor_length'], 50),
        np.linspace(0, params['reactor_width'], 50)
    )
    
    # Electron flux (Gaussian distribution)
    flux = beam_power * np.exp(-((X - params['reactor_length']/2)**2 + 
                               (Y - params['reactor_width']/2)**2) / 
                              (2 * params['beam_sigma']**2))
    flux /= (2 * np.pi * params['beam_sigma']**2)  # Normalize
    
    # Residence time and dose
    residence_time = X / params['gas_velocity']  # Time in seconds
    dose = calculate_dose(flux, mass_flow) * residence_time
    
    # Pollutant removal efficiencies
    eta_SOx = removal_efficiency(k_SOx, dose, params['humidity'])
    eta_NOx = removal_efficiency(k_NOx, dose, params['NH3_ppm']/1e6)
    
    return X, Y, flux, eta_SOx, eta_NOx

# ========================
# SENSITIVITY ANALYSIS 
# ========================
def sensitivity_analysis(param_ranges):
    """Robust parameter analysis with physical constraints"""
    results = {'params': [], 'SOx': [], 'NOx': []}
    
    for param_name, values in param_ranges.items():
        for val in values:
            params = BASE_PARAMS.copy()
            params[param_name] = val
            
            try:
                # Run simulation
                _, _, _, eta_SOx, eta_NOx = run_2D_simulation(params)
                
                # Store area-averaged results
                results['params'].append(val)
                results['SOx'].append(np.mean(eta_SOx))
                results['NOx'].append(np.mean(eta_NOx))
                
            except (ValueError, ZeroDivisionError) as e:
                print(f"Skipping invalid {param_name}={val}: {str(e)}")
    
    return results

# ========================
# TRANSIENT DYNAMICS (VALIDATED)
# ========================
def transient_model(t, y, params):
    """Dynamic system for time-varying engine loads"""
    SOx, NOx = y
    load_factor = 1 + 0.2 * np.sin(2 * np.pi * t/60)  # 60-second cycle
    
    modified_params = params.copy()
    modified_params['exhaust_flow_m3_per_s'] *= load_factor
    modified_params['current_mA'] *= load_factor  # Scale beam power
    
    _, _, _, eta_SOx, eta_NOx = run_2D_simulation(modified_params)
    
    dSOx_dt = -np.mean(eta_SOx) * SOx
    dNOx_dt = -np.mean(eta_NOx) * NOx
    
    return [dSOx_dt, dNOx_dt]

# ========================
# OPTIMIZATION (SAFE IMPLEMENTATION)
# ========================
def optimization_objective(x):
    """Multi-objective: Maximize efficiency while minimizing power/NH3"""
    voltage, current, NH3 = x
    
    # Physical constraints
    if voltage < 0.5 or voltage > 1.5:
        return np.inf
    if current < 100 or current > 300:
        return np.inf
    
    params = BASE_PARAMS.copy()
    params.update({
        'voltage_MeV': voltage,
        'current_mA': current,
        'NH3_ppm': NH3
    })
    
    try:
        _, _, _, eta_SOx, eta_NOx = run_2D_simulation(params)
        power = calculate_beam_power(voltage, current)
        
        # Trade-off: Efficiency vs. energy/NH3 use
        return (0.3*(1 - eta_SOx.mean()) + 
                0.3*(1 - eta_NOx.mean()) + 
                0.2*(power/200) + 
                0.2*(NH3/2500))
    
    except:
        return np.inf

# ========================
# VISUALIZATION (ERROR-CHECKED)
# ========================
def plot_results(X, Y, flux, eta_SOx, eta_NOx):
    """2D visualization of reactor performance"""
    plt.figure(figsize=(15, 5))
    
    cmap = LinearSegmentedColormap.from_list('custom', ['#2A00FF', '#FF0000'])
    
    plt.subplot(131)
    plt.pcolormesh(X, Y, flux, cmap='viridis')
    plt.title('Electron Flux (kW/m²)')
    plt.colorbar()
    
    plt.subplot(132)
    plt.pcolormesh(X, Y, eta_SOx, cmap=cmap, vmin=0, vmax=1)
    plt.title('SOx Removal Efficiency')
    plt.colorbar()
    
    plt.subplot(133)
    plt.pcolormesh(X, Y, eta_NOx, cmap=cmap, vmin=0, vmax=1)
    plt.title('NOx Removal Efficiency')
    plt.colorbar()
    
    plt.tight_layout()

def plot_sensitivity(results):
    """Parameter sensitivity visualization"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.plot(results['params'], results['SOx'], 'b-o')
    plt.xlabel('Parameter Value')
    plt.ylabel('SOx Removal Efficiency')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(results['params'], results['NOx'], 'r--s')
    plt.xlabel('Parameter Value')
    plt.ylabel('NOx Removal Efficiency')
    plt.grid(True)
    
    plt.tight_layout()

# ========================
# MAIN EXECUTION (SAFE)
# ========================
if __name__ == "__main__":
    # 1. Base 2D Simulation
    print("Running base 2D simulation...")
    X, Y, flux, eta_SOx, eta_NOx = run_2D_simulation(BASE_PARAMS)
    plot_results(X, Y, flux, eta_SOx, eta_NOx)
    
    # 2. Sensitivity Analysis
    print("\nRunning sensitivity analysis...")
    param_ranges = {
        'current_mA': np.linspace(100, 300, 10),
        'exhaust_flow_m3_per_s': np.linspace(200, 400, 10),
        'gas_velocity': np.linspace(3, 7, 10)
    }
    sens_results = sensitivity_analysis(param_ranges)
    plot_sensitivity(sens_results)
    
    # 3. Transient Simulation
    print("\nRunning transient simulation...")
    sol = solve_ivp(transient_model, [0, 300], [2000, 1500],
                    args=(BASE_PARAMS,), t_eval=np.linspace(0, 300, 50))
    
    plt.figure()
    plt.plot(sol.t, sol.y[0], label='SOx')
    plt.plot(sol.t, sol.y[1], label='NOx')
    plt.title('Transient Pollutant Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (ppm)')
    plt.legend()
    
    # 4. Optimization
    print("\nRunning optimization...")
    bounds = [(0.5, 1.5), (100, 300), (1000, 3000)]
    result = differential_evolution(optimization_objective, bounds,
                                    popsize=10, tol=0.01, seed=42)
    
    print(f"\nOptimal Parameters:")
    print(f"Voltage: {result.x[0]:.2f} MeV")
    print(f"Current: {result.x[1]:.0f} mA")
    print(f"NH3: {result.x[2]:.0f} ppm")
    
    plt.show()