import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants as sci
from matplotlib.animation import FuncAnimation


# Parameters
L = 1.0                # Width of the well in nm
L_m = L * 1e-9         # Convert to meters
x = np.linspace(0, L, 1000)
n = 3        # Quantum number
t=0.0

# Wavefunction
def psi(x, n):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def psiTime(x_nm, t, n):
    x_m = x_nm * 1e-9
    E = Energy(n)
    return psi(x_nm, n) * np.exp(-1j * E * t / sci.hbar)

def psiSuper(x,t):
    return 1/np.sqrt(4)* (psiTime(x, t, 1) + psiTime(x, t, 2) + (psiTime(x, t, 4) + psiTime(x, t, 3)))

# Probability density
def psi_sq(x, t,n):
    return np.abs(psiSuper(x, t))**2

# Energy of state n
def Energy(n):
    return (n**2 * np.pi**2 * sci.hbar**2) / (2 * sci.electron_mass * L_m**2)

E_J = Energy(n)
E_eV = E_J / sci.e
area, error = quad(lambda x: psi_sq(x, t,n), 0, L)

#---------------PLOTTING---------------#
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
ax_psi, ax_psiIm ,ax_psiDist = axes

line_psi, = ax_psi.plot(x, np.real(psiSuper(x, 0)), label=f"Re[Ψ(x,t)], n={n}")
line_psiIm, = ax_psiIm.plot(x, np.imag(psiSuper(x, 0)), label=f"Im[Ψ(x,t)], n={n}",color='orange')
line_prob, = ax_psiDist.plot(x, psi_sq(x, 0, n), label=f'n={n}',color='green')

# Formatting
ax_psi.set_title('Wavefunctions ψ_n(x,t)')
ax_psi.set_xlabel('x (nm)')
ax_psi.set_ylabel('Reψ_n(x,t)')
ax_psi.grid(True)
ax_psi.set_ylim(-1.45)

ax_psiIm.set_title('Wavefunctions ψ_n(x,t)')
ax_psiIm.set_xlabel('x (nm)')
ax_psiIm.set_ylabel('Imψ_n(x,t)')
ax_psiIm.grid(True)
ax_psiIm.set_ylim(-1.45,1.45)

ax_psiDist.set_title('Probability Distribution |ψ_n(x)|²')
ax_psiDist.set_xlabel('x (nm)')
ax_psiDist.set_ylabel('|ψ_n(x)|²')
ax_psiDist.grid(True)

# Animation function
def animate(frame):
    t = frame * 1e-16  
    line_psi.set_ydata(np.real(psiSuper(x, t)))
    line_psiIm.set_ydata(np.imag(psiSuper(x, t)))
    line_prob.set_ydata(psi_sq(x, t, n))
    area, error = quad(lambda x: psi_sq(x, t,n), 0, L)
    print(f"∫₀ᴸ |ψ(x,t)|² dx = {area:.6f} (± {error:.2e})")
    return line_psi,line_psiIm ,line_prob

anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
fig.suptitle(f"Time Dependent 1D Well Wave Functions ψ(x,t) \n n={n}")
plt.tight_layout()
plt.show()