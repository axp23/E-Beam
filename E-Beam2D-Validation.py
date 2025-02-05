import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Constants
e_charge = 1.602e-19  # Coulombs
R_air = 287.0         # J/kg·K
k_SOx = 8.0           # kGy⁻¹
k_NOx = 3.0           # kGy⁻¹

# --------------------------
# 2D Reactor Model Parameters
# --------------------------
reactor_length = 5.0    # Reactor length (m)
reactor_width = 2.0     # Reactor width (m)
grid_resolution = 50    # Grid points per dimension (50x50)
gas_velocity = 5.0      # Gas flow velocity (m/s) - plug flow assumption

# Electron beam parameters (2D Gaussian flux distribution)
beam_center = (reactor_length/2, reactor_width/2)  # Beam irradiates center
beam_sigma = 0.5       # Gaussian spread (m)

# --------------------------
# Core 2D Functions
# --------------------------
def create_2D_grid():
    """Create 2D spatial grid for the reactor."""
    x = np.linspace(0, reactor_length, grid_resolution)
    y = np.linspace(0, reactor_width, grid_resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y

def electron_flux(X, Y, beam_power_kW):
    """Calculate 2D electron flux (kW/m²) with Gaussian distribution."""
    # Gaussian beam profile
    flux = beam_power_kW * np.exp(-((X - beam_center[0])**2 + (Y - beam_center[1])**2) / (2 * beam_sigma**2))
    flux /= (2 * np.pi * beam_sigma**2)  # Normalize
    return flux

def gas_residence_time(X):
    """Calculate residence time (s) based on distance from inlet."""
    return X / gas_velocity

def calculate_2D_dose(flux, residence_time, efficiency=0.6):
    """Calculate 2D dose (kGy) from flux and residence time."""
    return flux * residence_time * efficiency  # (kW/m²) * s = kJ/m² → convert to kGy (kJ/kg)
                                              # Assume gas density ~1 kg/m³ for simplification

def update_pollutants(SOx_ppm, NOx_ppm, dose, humidity, NH3_ppm):
    """Update 2D pollutant concentrations after e-beam treatment."""
    eta_SOx = 1 - np.exp(-k_SOx * dose * humidity)
    eta_NOx = 1 - np.exp(-k_NOx * dose * (NH3_ppm / 1e6))
    SOx_new = SOx_ppm * (1 - eta_SOx)
    NOx_new = NOx_ppm * (1 - eta_NOx)
    return SOx_new, NOx_new

# --------------------------
# Simulation Workflow
# --------------------------
def run_2D_simulation():
    # Initialize grid and parameters
    X, Y = create_2D_grid()
    SOx_ppm_initial = 2000 * np.ones_like(X)  # Uniform initial SOx
    NOx_ppm_initial = 1500 * np.ones_like(X)  # Uniform initial NOx
    humidity_2D = 0.15 * np.ones_like(X)      # Uniform humidity
    NH3_ppm_2D = 2000 * np.ones_like(X)       # Uniform NH3
    
    # Calculate electron flux (beam power = 160 kW)
    flux = electron_flux(X, Y, beam_power_kW=160)
    
    # Calculate residence time and dose
    residence_time = gas_residence_time(X)
    dose = calculate_2D_dose(flux, residence_time)
    
    # Update pollutant concentrations
    SOx_ppm_final, NOx_ppm_final = update_pollutants(
        SOx_ppm_initial, NOx_ppm_initial, dose, humidity_2D, NH3_ppm_2D
    )
    
    # Byproduct formation (kg/m³)
    m_SOx_removed = (SOx_ppm_initial - SOx_ppm_final) * 1e-6 * 1.2  # 1.2 kg/m³ gas density
    m_ammonium_sulfate = m_SOx_removed * (132.14 / 64.06)
    
    return X, Y, flux, SOx_ppm_final, m_ammonium_sulfate

# --------------------------
# Visualization
# --------------------------
def plot_2D_results(X, Y, flux, SOx_final, sulfate):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Custom colormaps
    cmap_flux = LinearSegmentedColormap.from_list('flux', ['black', 'yellow'])
    cmap_pollutant = LinearSegmentedColormap.from_list('sox', ['green', 'red'])
    cmap_byproduct = LinearSegmentedColormap.from_list('sulfate', ['blue', 'cyan'])
    
    # Plot electron flux
    flux_plot = axes[0].pcolormesh(X, Y, flux, cmap=cmap_flux, shading='auto')
    axes[0].set_title('Electron Flux (kW/m²)')
    axes[0].set_xlabel('Reactor Length (m)')
    axes[0].set_ylabel('Reactor Width (m)')
    plt.colorbar(flux_plot, ax=axes[0])
    
    # Plot SOx concentration
    sox_plot = axes[1].pcolormesh(X, Y, SOx_final, cmap=cmap_pollutant, vmin=0, vmax=2000)
    axes[1].set_title('SOx Concentration (ppm) Post-Treatment')
    axes[1].set_xlabel('Reactor Length (m)')
    plt.colorbar(sox_plot, ax=axes[1])
    
    # Plot ammonium sulfate
    sulfate_plot = axes[2].pcolormesh(X, Y, sulfate, cmap=cmap_byproduct, vmin=0, vmax=2)
    axes[2].set_title('Ammonium Sulfate Yield (kg/m³)')
    axes[2].set_xlabel('Reactor Length (m)')
    plt.colorbar(sulfate_plot, ax=axes[2])
    
    plt.suptitle('2D Electron Beam Scrubbing Process', y=1.05, fontsize=14)
    plt.tight_layout()
    plt.show()

# Run simulation and plot
X, Y, flux, SOx_final, sulfate = run_2D_simulation()
plot_2D_results(X, Y, flux, SOx_final, sulfate)