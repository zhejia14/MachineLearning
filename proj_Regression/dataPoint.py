import numpy as np
import sys
import os


if len(sys.argv)<2:
    print("Error : need data points number !")
    exit()
num = int(sys.argv[1])
print(num)
def sine(x):
	return np.sin(2*np.pi*x) + np.random.normal(0,0.04)

x = np.linspace(0,1,num)
y = [0]*len(x)

for i in range(len(x)):
	y[i] = sine(x[i])

with open('points.txt', 'w') as file:
    for i in range(len(x)):
	    file.write("{:.2f} {:.2f}\n".format(x[i], y[i]))