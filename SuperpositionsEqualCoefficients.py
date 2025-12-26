import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants as sci
from matplotlib.animation import FuncAnimation

# Parameters
L = 1e-9              
x = np.linspace(0, L, 1000)                   

t=0.0
tvalues = np.linspace(0,2e-14,1000)

n_max = 3
momentumBounds = n_max * np.pi * sci.hbar / L
pValues = np.linspace(-12*momentumBounds, 12*momentumBounds, 1000)
# Wavefunction
def psi(x, n):
    return np.sqrt(2/L) * np.sin(n*np.pi*x/L)

def psiTime(x, t, n):
    return psi(x, n) * np.exp(-1j * Energy(n) * t / sci.hbar)

def psiSuper(x, t):
    return (1/np.sqrt(n_max)) * (
        psiTime(x, t, 1)+
        psiTime(x, t, 2)+
        psiTime(x, t, 3)
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

def momentumFunc(p, t):
    real, _ = quad(lambda x: np.real(np.exp(-1j*x*p/sci.hbar) * psiSuper(x,t)), 0, L)
    imag, _ = quad(lambda x: np.imag(np.exp(-1j*x*p/sci.hbar) * psiSuper(x,t)), 0, L)
    return (real + 1j*imag) / np.sqrt(2*np.pi*sci.hbar)


#Calculated Expeted Values
expectedXValues = [expectedX(time) for time in tvalues]
expectedXSquaredValues = [expectedXSquared(time) for time in tvalues]
deltaXValues = [np.sqrt(expectedXSquared(time)-(expectedX(time)**2)) for time in tvalues]
phiValues = np.array([momentumFunc(p, 0) for p in pValues])
phiArea, phiError = quad(lambda p: np.abs(momentumFunc(p, 0))**2, pValues[0], pValues[-1])
#---------------PLOTTING---------------#

x_nm = x * 1e9
fig, axes = plt.subplots(3, 4, figsize=(12, 7))

(ax_psi, ax_psiIm,ax_psiDist,ax_ExpX) , (ax_phi, ax_phiIm,ax_phiDist,ax_DeltaP), (ax_DeltaX , ax_DeltaP, _,_) = axes

line_psi, = ax_psi.plot(x_nm, np.real(psiSuper(x, 0)), label=f"Re[Ψ(x,t)]")
line_psiIm, = ax_psiIm.plot(x_nm, np.imag(psiSuper(x, 0)), label=f"Im[Ψ(x,t)]",color='orange')
line_prob, = ax_psiDist.plot(x_nm, psi_sq(x, 0),color='green')

line_phi, = ax_phi.plot(pValues, np.real(phiValues), label=f"Re[Ψ(x,t)]")
line_phiIm, = ax_phiIm.plot(pValues, np.imag(phiValues), label=f"Im[Ψ(x,t)]",color='orange')
line_phiProb, = ax_phiDist.plot(pValues, np.abs(np.square(phiValues)),color='green')

# Formatting
ax_psi.set_title('Wavefunctions ψ_n(x,t)')
ax_psi.set_xlabel('x (nm)')
ax_psi.set_ylabel('Reψ_n(x,t)')
ax_psi.grid(True)
ax_psi.set_ylim(-6e4, 6e4)

ax_psiIm.set_title('Wavefunctions ψ_n(x,t)')
ax_psiIm.set_xlabel('x (nm)')
ax_psiIm.set_ylabel('Imψ_n(x,t)')
ax_psiIm.grid(True)
ax_psiIm.set_ylim(-6e4, 6e4)

ax_psiDist.set_title('Probability Distribution |ψ_n(x)|²')
ax_psiDist.set_xlabel('x (nm)')
ax_psiDist.set_ylabel('|ψ_n(x)|²')
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

ax_phiIm.set_title('Wavefunction ϕ(p,t)')
ax_phiIm.set_xlabel('p(kgms)')
ax_phiIm.set_ylabel('Imϕ(p,t)')
ax_phiIm.grid(True)

ax_phiDist.set_title('Probability Distribution |ϕ_n(x)|²')
ax_phiDist.set_xlabel('p(kgms)')
ax_phiDist.set_ylabel('|ϕ_n(x)|²')
ax_phiDist.grid(True)

# Animation function
def animate(frame):
    t = frame * 1e-16  
    line_psi.set_ydata(np.real(psiSuper(x, t)))
    line_psiIm.set_ydata(np.imag(psiSuper(x, t)))
    line_prob.set_ydata(psi_sq(x, t))
    psiArea, psiError = quad(lambda x: psi_sq(x, t), 0, L)
    print(f"-------------------------------")
    print(f"Time: {t}")
    print(f"∫₀ᴸ |ψ(x,t)|² dx = {psiArea:.6f} (± {psiError:.2e})")
    print(f"∫₀ᴸ |ϕ(p,t)|² dx = {phiArea:.6f} (± {phiError:.2e})")
    print(f"<X> = {expectedX(t)*1e9} nm")
    print(f"<X²> = {expectedXSquared(t)}")
    print(f"-------------------------------")
    return line_psi,line_psiIm ,line_prob

anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
fig.suptitle(f"Time Dependent 1D Well Wave Functions ψ(x,t)")
plt.tight_layout()
plt.show()
