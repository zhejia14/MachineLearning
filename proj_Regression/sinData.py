import numpy as np

def sine(x):
	return np.sin(2*np.pi*x) + np.random.normal(0,0.04)

x = np.linspace(0,1,15)
y = [0]*len(x)

for i in range(len(x)):
	y[i] = sine(x[i])

with open('sinDataset.txt', 'w') as file:

	for i in range(len(x)):
		file.write("{:.2f} {:.2f}\n".format(x[i], y[i]))
		print("{:.2f} {:.2f}\n".format(x[i], y[i]))