import numpy as np

def line(x):
	return 2*x + np.random.normal(0,1)

x = np.linspace(-3,3,15)
y = [0]*len(x)
for i in range(len(x)):
	y[i]=line(x[i])

with open('lineDataset.txt', 'w') as file:

	for i in range(len(x)):
		file.write("{:.2f} {:.2f}\n".format(x[i], y[i]))
		print("{:.2f} {:.2f}\n".format(x[i], y[i]))
