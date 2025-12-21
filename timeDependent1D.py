import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants as sci

# Parameters
L = 1.0                # Width of the well in nm
L_m = L * 1e-9         # Convert to meters
x = np.linspace(0, L, 1000)
n = 1        # Quantum number
t=0.5e-14
# Wavefunction
def psi(x, n):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def psiTime(x, t, n):
    E = Energy(n)
    return psi(x, n) * np.exp(-1j * E * t / sci.hbar)

# Probability density
def psi_sq(x, n):
    return (2/L) * np.sin(n * np.pi * x / L)**2

# Energy of state n
def Energy(n):
    return (n**2 * np.pi**2 * sci.hbar**2) / (2 * sci.electron_mass * L_m**2)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax_psi, ax_psiDist = axes

ax_psi.plot(x, np.real(psiTime(x,t,n)),  label=f"Re[Ψ(x,t)], n={n}")
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