import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants as sci
from matplotlib.animation import FuncAnimation

# Parameters
res = 10000
L = 1e-9              
x = np.linspace(0, L, res)              
dx = x[1] - x[0]
N = len(x) 
t=0.0
tvalues = np.linspace(0,2e-14,res)
n_max = 2
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
    k = n * np.pi / L
    prefactor = np.sqrt(2 / L) / np.sqrt(2 * np.pi * sci.hbar)
    numerator = k * (1 - (-1)**n * np.exp(-1j * p * L / sci.hbar))
    denominator = k**2 - (p / sci.hbar)**2
    return prefactor * numerator / denominator

# Superposition of states
def phiSuper(p, t):
    return (1/np.sqrt(n_max)) * sum(
        phi_n(p, n) * np.exp(-1j * Energy(n) * t / sci.hbar)
        for n in range(1, n_max + 1)
    )

# Probability density
def phi_sq(p, t):
    return np.abs(phiSuper(p, t))**2

def expectedP(t):
    expectedPValue, _ = quad(lambda p: p*np.abs(phiSuper(p,t))**2 , -12*momentumBounds, 12*momentumBounds)
    return expectedPValue

def expectedPSquared(t):
    expectedPSquaredValue, _ = quad(lambda p: (p**2)*np.abs(phiSuper(p,t))**2 , -12*momentumBounds, 12*momentumBounds)
    return expectedPSquaredValue
    
#--------------EXPECTATION VALUES-------------#
#Calculated Expeted Values
expectedXValues = [expectedX(time) for time in tvalues]
expectedXSquaredValues = [expectedXSquared(time) for time in tvalues]
deltaXValues = [np.sqrt(expectedXSquared(time)-(expectedX(time)**2)) for time in tvalues]

expectedPValues = [expectedP(time) for time in tvalues]
expectedPSquaredValues = [expectedPSquared(time) for time in tvalues]
deltaPValues = [np.sqrt(expectedPSquared(time)-(expectedP(time)**2)) for time in tvalues]

heisenbergUncValue = ([a * b for a, b in zip(deltaXValues, deltaPValues)])
dXdtValues = np.gradient(expectedXValues, tvalues)
dXdt_analytic = np.array(expectedPValues) / sci.electron_mass

dPdtValues = np.gradient(expectedPValues, tvalues)

#---------------PLOTTING---------------#
x_nm = x * 1e9
fig, axes = plt.subplots(3, 4, figsize=(12, 7))

(ax_psi, ax_psiIm,ax_psiDist,ax_ExpX) , (ax_phi, ax_phiIm,ax_phiDist,ax_ExpP), (ax_DeltaX , ax_DeltaP, ax_heisenbergUnc, ax_ehren) = axes

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
ax_phi.set_ylim(-10e11,10e11)

ax_phiIm.set_title('Wavefunction ϕ(p,t)')
ax_phiIm.set_xlabel('p(kgms)')
ax_phiIm.set_ylabel('Imϕ(p,t)')
ax_phiIm.grid(True)
ax_phiIm.set_xlim(-3.5e-24,3.5e-24)
ax_phiIm.set_ylim(-10e11,10e11)

ax_phiDist.set_title('Probability Distribution |ϕ(x)|²')
ax_phiDist.set_xlabel('p(kgms)')
ax_phiDist.set_ylabel('|ϕ(x)|²')
ax_phiDist.grid(True)
ax_phiDist.set_xlim(-3.5e-24,3.5e-24)


ax_ExpP.plot(tvalues,expectedPValues ,color='lightcoral')
ax_ExpP.set_title('<P> Over Time')
ax_ExpP.set_xlabel('Time (s)')
ax_ExpP.set_ylabel('<P>')
ax_ExpP.grid(True)

ax_DeltaP.plot(tvalues,deltaPValues ,color='mediumpurple')
ax_DeltaP.set_title('ΔP Over Time')
ax_DeltaP.set_xlabel('Time (s)')
ax_DeltaP.set_ylabel('ΔP')
ax_DeltaP.grid(True)

ax_heisenbergUnc.plot(tvalues,heisenbergUncValue ,color='mediumpurple')
ax_heisenbergUnc.set_title('ΔX ΔP Over Time')
ax_heisenbergUnc.set_xlabel('Time (s)')

ax_heisenbergUnc.set_ylabel('ΔX ΔP')
ax_heisenbergUnc.axhline(y=sci.hbar/2, color='red', linestyle='--', label='ℏ/2')
ax_heisenbergUnc.grid(True)
ax_heisenbergUnc.legend()


ax_ehren.set_title('d<X>/dt vs <P>/m (Ehrenfest Check)')
ax_ehren.plot(tvalues,dXdtValues ,color='hotpink')
ax_ehren.plot(tvalues,dXdt_analytic  ,color='steelblue')
ax_ehren.set_xlabel('Time (s)')

# Animation function
def animate(frame):
    t = frame * 1e-16  
    line_psi.set_ydata(np.real(psiSuper(x, t)))
    line_psiIm.set_ydata(np.imag(psiSuper(x, t)))
    line_prob.set_ydata(psi_sq(x, t))
    psiArea, psiError = quad(lambda x: psi_sq(x, t), 0, L)
    line_phi.set_ydata(np.real(phiSuper(p, t)))
    line_phiIm.set_ydata(np.imag(phiSuper(p, t)))
    line_phiprob.set_ydata(phi_sq(p, t))
    phiArea, phiError = quad(lambda p: phi_sq(p, t), p[0], p[-1])
    diff = np.abs(dXdtValues[frame]-dXdt_analytic[frame])
    print(f"-----------{n_max}------------")
    print(f"Time: {t}")
    print(f"∫₀ᴸ |ψₙ(x,t)|² dx = {psiArea:.6f} (± {psiError:.2e})")
    print(f"∫ |ϕ(p,t)|² dx = {phiArea:.6f} (± {phiError:.2e})")
    print(f"<X> = {expectedX(t)*1e9} nm")
    print(f"<X²> = {expectedXSquared(t)}")
    print(f"<P> = {expectedP(t)} kgms")
    print(f"<P²> = {expectedPSquared(t)}")
    print(f"ΔXΔP = {heisenbergUncValue[frame]}")
    print(f"Difference(t) = {diff}")
    print(f"% Error = {np.abs(diff) / np.abs(dXdt_analytic[frame]) * 100}%")
    print(f"<F> = {dPdtValues[frame]}")
    print(f"-------------------------------")
    if heisenbergUncValue[frame] < (sci.hbar/2):
        print("Heisenberg Uncertainty Principle Broken")

    return line_psi, line_psiIm, line_prob, line_phi, line_phiIm

anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
fig.suptitle(f"Time Dependent 1D Well Wave Functions")
plt.tight_layout()
plt.show()
