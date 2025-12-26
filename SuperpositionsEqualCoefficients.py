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
pValues = np.linspace(-6.6261e-25,+6.6261e-25,1000)
n_max=2
print(-6.6261e-25)

# Wavefunction
def psi(x, n):
    return np.sqrt(2/L) * np.sin(n*np.pi*x/L)

def psiTime(x, t, n):
    return psi(x, n) * np.exp(-1j * Energy(n) * t / sci.hbar)

def psiSuper(x, t):
    return (1/np.sqrt(1)) * (
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
    expectedXValue, error = quad(lambda x: x*np.abs(psiSuper(x,t))**2 , 0, L)
    return expectedXValue

def expectedXSquared(t):
    expectedXSquaredValue, error = quad(lambda x: (x**2)*np.abs(psiSuper(x,t))**2 , 0, L)
    return expectedXSquaredValue

def momentumFunc(p, t):
    real, _ = quad(lambda x: np.real(np.exp(-1j*x*p/sci.hbar) * psiSuper(x,t)), 0, L)
    imag, _ = quad(lambda x: np.imag(np.exp(-1j*x*p/sci.hbar) * psiSuper(x,t)), 0, L)
    return (real + 1j*imag) / np.sqrt(2*np.pi*sci.hbar)


#Calculated Expeted Values
expectedXValues = []
i=0
for time in tvalues:
    expectedXValues.append(expectedX(time))

expectedXSquaredValues = []
i=0
for time in tvalues:
    expectedXSquaredValues.append(expectedXSquared(time))

deltaXValues = []
i=0
for time in tvalues:
    deltaXValues.append(np.sqrt(expectedXSquared(time)-(expectedX(time)**2)))

#---------------PLOTTING---------------#

x_nm = x * 1e9
fig, axes = plt.subplots(3, 4, figsize=(12, 7))

(ax_psi, ax_psiIm,ax_psiDist,ax_ExpX) , (ax_phi, ax_phiIm,ax_phiDist,ax_DeltaP), (ax_DeltaX , ax_DeltaP, _,_) = axes

line_psi, = ax_psi.plot(x_nm, np.real(psiSuper(x, 0)), label=f"Re[Ψ(x,t)]")
line_psiIm, = ax_psiIm.plot(x_nm, np.imag(psiSuper(x, 0)), label=f"Im[Ψ(x,t)]",color='orange')
line_prob, = ax_psiDist.plot(x_nm, psi_sq(x, 0),color='green')

#line_phi, = ax_phi.plot(x, np.real(momentumFunc(pValues,0)), label=f"Re[Ψ(x,t)]")
#line_phiIm, = ax_phiIm.plot(x, np.imag(psiSuper(x, 0)), label=f"Im[Ψ(x,t)]",color='orange')
#line_phiProb, = ax_phiDist.plot(x, psi_sq(x, 0),color='green')

# Formatting
ax_psi.set_title('Wavefunctions ψ_n(x,t)')
ax_psi.set_xlabel('x (nm)')
ax_psi.set_ylabel('Reψ_n(x,t)')
ax_psi.grid(True)
ax_psi.set_ylim(-5e4, 5e4)

ax_psiIm.set_title('Wavefunctions ψ_n(x,t)')
ax_psiIm.set_xlabel('x (nm)')
ax_psiIm.set_ylabel('Imψ_n(x,t)')
ax_psiIm.grid(True)
ax_psiIm.set_ylim(-5e4, 5e4)

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

# Animation function
def animate(frame):
    t = frame * 1e-16  
    line_psi.set_ydata(np.real(psiSuper(x, t)))
    line_psiIm.set_ydata(np.imag(psiSuper(x, t)))
    line_prob.set_ydata(psi_sq(x, t))
    #line_phi.set_ydata(momentumFunc(p,t))
    area, error = quad(lambda x: psi_sq(x, t), 0, L)
    print(f"-------------------------------")
    print(f"Time: {t}")
    print(f"∫₀ᴸ |ψ(x,t)|² dx = {area:.6f} (± {error:.2e})")
    print(f"<X> = {expectedX(t)}")
    print(f"<X²> = {expectedXSquared(t)}")
    print(f"-------------------------------")
    return line_psi,line_psiIm ,line_prob

anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
fig.suptitle(f"Time Dependent 1D Well Wave Functions ψ(x,t)")
plt.tight_layout()
plt.show()
