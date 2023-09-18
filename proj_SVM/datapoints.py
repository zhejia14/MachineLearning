import numpy as np
import random
def line(x):
	return 2*x + np.random.normal(0,1)

x = np.linspace(-100,100,500)
y = [0]*len(x)
s = [0]*len(x)

for i in range(len(x)):
	y[i]=line(x[i])
	if( (2*x[i]+0.5) >= y[i]):
		s[i]=1
	else:
		s[i]=0
    

with open('dataset.txt', 'w') as file:

	for i in range(len(x)):
		file.write("{} 1:{:.2f} 2:{:.2f}\n".format(s[i], x[i], y[i]))