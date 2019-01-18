import matplotlib.pyplot as plt
import numpy as np

phio=np.logspace(-2, 2., num=50, endpoint=True, base=10.0)

fig = plt.figure(figsize=(8,5))
y =np.arange(0,len(phio), 1)
ax = plt.subplot(111)
ax.set_xscale("log")
plt.plot(phio,y, 'o')
plt.show()
