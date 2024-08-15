import matplotlib
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 11])
ypoints = np.array([0, 1000])

plt.title("Training Accuracy per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(xpoints, ypoints)
plt.savefig('foo.png', bbox_inches='tight')
plt.show()