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

n_value = 1
E_J = Energy(n_value)
E_eV = E_J / sci.e
area, error = quad(lambda x: psi_sq(x, n_value), 0, L)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax_psi, ax_psiDist = axes

info_text = (
    "\n"rf"$n = {n_value}$" "\n"
    rf"$E = {E_J:.2e}\ \mathrm{{J}}$" "\n"
    rf"$E = {E_eV:.3f}\ \mathrm{{eV}}$" "\n"
    rf"$\int_0^L |\psi|^2 dx = {area:.2f}$"
)

# Styled annotation box
bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9)

# Wavefunctions and probability densities
for n in ns:
    ax_psi.plot(x, psi(x, n), label=f'n={n}')
    ax_psiDist.plot(x, psi_sq(x, n), label=f'n={n}')

# Format wavefunction plot
ax_psi.set_title('Wavefunctions ψ_n(x)')
ax_psi.set_xlabel('x (nm)')
ax_psi.set_ylabel('ψ_n(x)')
ax_psi.grid(True)
ax_psi.text(0.03, 0.45, info_text,
            transform=ax_psi.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=bbox_props)
ax_psi.legend()

# Probability density plot
ax_psiDist.set_title('Probability Densities |ψ_n(x)|²')
ax_psiDist.set_xlabel('x (nm)')
ax_psiDist.set_ylabel('|ψ_n(x)|²')
ax_psiDist.grid(True)
plt.tight_layout()
plt.show()