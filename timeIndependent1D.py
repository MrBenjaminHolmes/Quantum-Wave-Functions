import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants as sci

# Parameters
L = 1.0                # Width of the well in nm
L_m = L * 1e-9         # Convert to meters
x = np.linspace(0, L, 1000)
ns = [1, 2, 3]         # Quantum numbers to plot

# Wavefunction
def psi(x, n):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# Probability density
def psi_sq(x, n):
    return (2/L) * np.sin(n * np.pi * x / L)**2

# Energy of state n (in Joules)
def Energy(n):
    return (n**2 * np.pi**2 * sci.hbar**2) / (2 * sci.electron_mass * L_m**2)

# Energy for n=1
n_value = 1
E1_J = Energy(n_value)
E1_eV = E1_J / sci.e
print(f"Energy for n={n_value}: {E1_J:.3e} J ({E1_eV:.3f} eV)")

# Check Area
area, error = quad(lambda x: psi_sq(x, n_value), 0, L)
print("Total area under |ψ|^2 for n=1:", np.round(area, 2))

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax_psi, ax_psiDist = axes

# Wavefunctions and probability densities
for n in ns:
    ax_psi.plot(x, psi(x, n), label=f'n={n}')
    ax_psiDist.plot(x, psi_sq(x, n), label=f'n={n}')

# Format wavefunction plot
ax_psi.set_title('Wavefunctions ψ_n(x)')
ax_psi.set_xlabel('x (nm)')
ax_psi.set_ylabel('ψ_n(x)')
ax_psi.grid(True)
ax_psi.legend()

# Probability density plot
ax_psiDist.set_title('Probability Densities |ψ_n(x)|²')
ax_psiDist.set_xlabel('x (nm)')
ax_psiDist.set_ylabel('|ψ_n(x)|²')
ax_psiDist.grid(True)
ax_psiDist.legend()
plt.tight_layout()
plt.show()