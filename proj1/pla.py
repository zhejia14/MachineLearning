import matplotlib.pyplot as plt
import numpy as np
import operator
import fileinput
import time
dataset=[]
for line in fileinput.input():
	input=line.strip().split()
	x,y,label=float(input[0]), float(input[1]), float(input[2])
	dataset.append(((1, x, y),label))

def dot(*v):
	return sum(map(operator.mul, *v))

def sign(v):
	if v>0:
		return 1
	elif v==0:
		return 0
	else:
		return -1
	
def check_error(w, x, y):
	if sign(dot(w,x)) != y:
		return True
	else:
		return False

def err_sum(w, dataset):
	error = 0
	for x,y in dataset:
		if check_error(w, x, y):
			error +=1
	return error


def update(w,x,y):
	u = map(operator.mul, [y]*len(x), x)
	w = map(operator.add, w, u)
	return list(w)

def pla(dataset):
	w = np.zeros(3)
	iter=0
	while True:
		print("{}: {}".format(iter, tuple(w)))
		err = True
		for x,y in dataset:
			if check_error(w, x, y):
				w = update(w, x, y)
				err = False
				break
		iter +=1
		
		if err or iter>100000:
			break
	print("Total iteration : {}".format(iter))
	errors = err_sum(w, dataset) 
	accuracy = ((len(dataset) - errors) / len(dataset)) * 100
	print("Accuracy: {}%".format(accuracy))
	return w
start_t = time.time()
w = pla(dataset)
end_t = time.time()
exe_t = end_t - start_t
print("ExecutionTime {:.3f}".format(exe_t))


fig = plt.figure()
ax1 = fig.add_subplot(111)
xx = list(filter(lambda d: d[1] == -1, dataset))
ax1.scatter([x[0][1] for x in xx], [x[0][2] for x in xx],s=10, c='b',marker="x",label='-1')
oo = list(filter(lambda d: d[1] == 1, dataset))
ax1.scatter([x[0][1] for x in oo], [x[0][2] for x in oo],s=10, c='r', marker="o", label='1')
l = np.linspace(-100, 100)
a = -w[1] / w[2]
b = -w[0] / w[2]
ax1.plot(l, a*l+b, 'b-')
plt.legend(loc='upper left', scatterpoints=1);
plt.show()





