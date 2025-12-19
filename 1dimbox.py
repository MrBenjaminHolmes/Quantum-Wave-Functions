import numpy as np
import matplotlib.pyplot as plt

L=1.0
n=1
x= np.linspace(0,L,1000)

psi = np.sqrt(2/L) * np.sin( ( n*np.pi*x) / L ) 

# Plot
plt.plot(x, psi, label=f'n={n}')
plt.title("Wave Function for 1D Particle in a Box")
plt.xlabel("x")
plt.ylabel("ψ_n(x)")
plt.grid(True)
plt.legend()
plt.show()