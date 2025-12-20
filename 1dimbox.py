import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

L=1.0
n=2
x= np.linspace(0,L,1000)

def psi(x):
    return np.sqrt(2/L) * np.sin( ( n*np.pi*x) / L ) 

def psi_sq(x):
    return (2/L) * np.sin(n*np.pi*x/L)**2

area, error = quad(psi_sq, 0, L)
print(np.round(area,2))
# Plot
fig, axes = plt.subplots(1,2, figsize=(9, 5))
(ax_psi,ax_psiDist) = axes
#Psi Wave Function
ax_psi.plot(x,psi(x), color='dodgerblue')
ax_psi.set_title('')
ax_psi.set_xlabel('x')
ax_psi.set_ylabel('ψ_n(x)')
ax_psi.grid(True)

#Psi Position Probability Distribution
ax_psiDist.plot(x,psi_sq(x), color='dodgerblue')
ax_psiDist.set_title('')
ax_psiDist.set_xlabel('x')
ax_psiDist.set_ylabel('|ψ_n(x)|²')
ax_psiDist.grid(True)
plt.show()