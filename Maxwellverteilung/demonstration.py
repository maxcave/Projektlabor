import matplotlib.pyplot as plt
import numpy as np

def maxwell(x, A):
    return x * A * np.exp(-1 * (A*(x)**2 / 2))
xspace = np.linspace(0,25,1000)

plt.subplot(1,2,1)
plt.plot(xspace, maxwell(xspace, 0.04), label=("m1"))
plt.plot(xspace, maxwell(xspace, 0.01), label=("m2"))
plt.xlim(0, 25)
plt.xlabel("v")
plt.ylabel("Häufigkeit")
plt.legend()
plt.subplot(1,2,2)
plt.plot(xspace, maxwell(xspace, 0.04), color="green", label=("T2"))
plt.plot(xspace, maxwell(xspace, 0.01), color="red", label=("T1"))
plt.xlim(0, 25)
plt.xlabel("v")
plt.ylabel("Häufigkeit")
plt.legend()
plt.show()
