import numpy as np
import matplotlib.pyplot as plt

# The SIR model without vital dynamics.

# Initial conditions:
S0 = 1.457e7 # Susceptible (pop. of Ontario)
I0 = 6 # Infected. Guess base value.
R0 = 0 # Recovered.

Tc = 1.50 # time between contacts in days (a guess)
Tr = 23 # time to recovery in days (time to symptoms 4-14 days + 14 days of illness)

beta = 1/Tc
gamma = 1/Tr

Tfinal = 120
dt = 0.05

numsteps = int(np.ceil(Tfinal / dt))

# Allocate time-series arrays.
R = np.zeros((numsteps)) # recovered
I = np.zeros((numsteps)) # infected
S = np.zeros((numsteps)) # susceptible

S[0] = S0
I[0] = I0
R[0] = R0

# total population (assumed conserved, i.e., no deaths or births).
N = S0 + I0 + R0

t = np.linspace(0, Tfinal, numsteps)
# Euler stepping, because lazy.
for j in range(1, numsteps):
    R[j] = R[j-1] + dt*gamma*I[j-1]
    S[j] = S[j-1] + dt*(-beta*I[j-1]*S[j-1]/N)
    I[j] = I[j-1] + dt*(beta*I[j-1]*S[j-1]/N - gamma*I[j-1])

plt.figure()
plt.plot(t, S/1e6, '-g', t, I/1e6, '-r', t, R/1e6, '-b' )
plt.xlabel("time (days)")
plt.ylabel("Number of People (millions)")
plt.grid()
plt.legend(["Susceptible", "Infected", "Recovered"])
plt.draw()
plt.show()