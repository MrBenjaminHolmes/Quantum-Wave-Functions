
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants as sci
from matplotlib.animation import FuncAnimation

# Parameters
res = 5000
L = 1e-9              
x = np.linspace(0, L, res)              
dx = x[1] - x[0]
N = len(x) 
t=0.0
tvalues = np.linspace(0,2e-14,res)
n_max = 3
momentumBounds = n_max * np.pi * sci.hbar / L
p = np.linspace(-12*momentumBounds, 12*momentumBounds, res)
#--------------POSITION SPACE-------------#
# Wavefunction
def psi(x, n):
    return np.sqrt(2/L) * np.sin(n*np.pi*x/L)

def psiTime(x, t, n):
    return psi(x, n) * np.exp(-1j * Energy(n) * t / sci.hbar)

def psiSuper(x, t):
    return (1/np.sqrt(n_max)) * sum(
        psi(x, n) * np.exp(-1j * Energy(n) * t / sci.hbar)
        for n in range(1, n_max + 1)
    )

# Probability density
def psi_sq(x, t):
    return np.abs(psiSuper(x, t))**2

# Energy of state n
def Energy(n):
    return (n**2 * np.pi**2 * sci.hbar**2) / (2 * sci.electron_mass * L**2)

#Expected X Values
def expectedX(t):
    expectedXValue, _ = quad(lambda x: x*np.abs(psiSuper(x,t))**2 , 0, L)
    return expectedXValue

def expectedXSquared(t):
    expectedXSquaredValue, _ = quad(lambda x: (x**2)*np.abs(psiSuper(x,t))**2 , 0, L)
    return expectedXSquaredValue

#--------------MOMENTUM-------------#
def phi_n(p, n):
    numerator = n * np.sqrt(np.pi * L) * (1 - (-1)**n * np.exp(-1j * p * L / sci.hbar))
    denominator = n**2 * np.pi**2 - (p * L / sci.hbar)**2
    return numerator / denominator

# Superposition of states
def phiSuper(p, t):
    return (1/np.sqrt(n_max)) * sum(
        phi_n(p, n) * np.exp(-1j * Energy(n) * t / sci.hbar)
        for n in range(1, n_max + 1)
    )

# Probability density
def phi_sq(p, t):
    return np.abs(phiSuper(p, t))**2

#--------------EXPECTATION VALUES-------------#
#Calculated Expeted Values
expectedXValues = [expectedX(time) for time in tvalues]
expectedXSquaredValues = [expectedXSquared(time) for time in tvalues]
deltaXValues = [np.sqrt(expectedXSquared(time)-(expectedX(time)**2)) for time in tvalues]

#---------------PLOTTING---------------#
x_nm = x * 1e9
fig, axes = plt.subplots(3, 4, figsize=(12, 7))

(ax_psi, ax_psiIm,ax_psiDist,ax_ExpX) , (ax_phi, ax_phiIm,ax_phiDist,ax_ExpP), (ax_DeltaX , ax_DeltaP, _,_) = axes

line_psi, = ax_psi.plot(x_nm, np.real(psiSuper(x, 0)), label=f"Re[Ψ(x,t)]")
line_psiIm, = ax_psiIm.plot(x_nm, np.imag(psiSuper(x, 0)), label=f"Im[Ψ(x,t)]",color='orange')
line_prob, = ax_psiDist.plot(x_nm, psi_sq(x, 0),color='green')

line_phi, = ax_phi.plot(p, np.real(phiSuper(p, 0)), label=f"Re[Ψ(x,t)]")
line_phiIm, = ax_phiIm.plot(p, np.imag(phiSuper(p, 0)), label=f"Im[Ψ(x,t)]",color='orange')
line_phiprob, = ax_phiDist.plot(p, phi_sq(p, 0),color='green')

# Formatting
ax_psi.set_title('Wavefunctions ψₙ(x,t)')
ax_psi.set_xlabel('x (nm)')
ax_psi.set_ylabel('Reψₙ(x,t)')
ax_psi.grid(True)
ax_psi.set_ylim(-6e4, 6e4)

ax_psiIm.set_title('Wavefunctions ψₙ(x,t)')
ax_psiIm.set_xlabel('x (nm)')
ax_psiIm.set_ylabel('Imψₙ(x,t)')
ax_psiIm.grid(True)
ax_psiIm.set_ylim(-6e4, 6e4)

ax_psiDist.set_title('Probability Distribution |ψₙ(x)|²')
ax_psiDist.set_xlabel('x (nm)')
ax_psiDist.set_ylabel('|ψₙ(x)|²')
ax_psiDist.grid(True)

ax_ExpX.plot(tvalues,expectedXValues ,color='lightcoral')
ax_ExpX.set_title('<X> Over Time')
ax_ExpX.set_xlabel('Time (s)')
ax_ExpX.set_ylabel('<X>')
ax_ExpX.grid(True)

ax_DeltaX.plot(tvalues,deltaXValues ,color='mediumpurple')
ax_DeltaX.set_title('ΔX Over Time')
ax_DeltaX.set_xlabel('Time (s)')
ax_DeltaX.set_ylabel('ΔX')
ax_DeltaX.grid(True)

ax_phi.set_title('Wavefunction ϕ(p,t)')
ax_phi.set_xlabel('p(kgms)')
ax_phi.set_ylabel('Reϕ(p,t)')
ax_phi.grid(True)
ax_phi.set_xlim(-3.5e-24,3.5e-24)

ax_phiIm.set_title('Wavefunction ϕ(p,t)')
ax_phiIm.set_xlabel('p(kgms)')
ax_phiIm.set_ylabel('Imϕ(p,t)')
ax_phiIm.grid(True)
ax_phiIm.set_xlim(-3.5e-24,3.5e-24)

ax_phiDist.set_title('Probability Distribution |ϕ(x)|²')
ax_phiDist.set_xlabel('p(kgms)')
ax_phiDist.set_ylabel('|ϕ(x)|²')
ax_phiDist.grid(True)
ax_phiDist.set_xlim(-3.5e-24,3.5e-24)
def phi_sq_scalar(p, t):
    # p is a scalar
    return np.abs(phiSuper(np.array([p]), t)[0])**2
# Animation function
def animate(frame):
    t = frame * 1e-16  
    line_psi.set_ydata(np.real(psiSuper(x, t)))
    line_psiIm.set_ydata(np.imag(psiSuper(x, t)))
    line_prob.set_ydata(psi_sq(x, t))
    psiArea, psiError = quad(lambda x: psi_sq(x, t), 0, L)
    line_phi.set_ydata(np.real(phiSuper(p, t)))
    line_phiIm.set_ydata(np.imag(phiSuper(p, t)))
    phiArea, phiError = quad(lambda p: phi_sq(p, t), p[0], p[-1])
    print(f"-------------------------------")
    print(f"Time: {t}")
    print(f"∫₀ᴸ |ψₙ(x,t)|² dx = {psiArea:.6f} (± {psiError:.2e})")
    print(f"∫ |ϕ(p,t)|² dp = {phiArea:.6f}")
    #print(f"∫ |ϕ(p,t)|² dx = {phiArea:.6f} (± {psiError:.2e})")
    print(f"<X> = {expectedX(t)*1e9} nm")
    print(f"<X²> = {expectedXSquared(t)}")
    print(f"-------------------------------")
    return line_psi, line_psiIm, line_prob, line_phi, line_phiIm

anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
fig.suptitle(f"Time Dependent 1D Well Wave Functions")
plt.tight_layout()
plt.show()
