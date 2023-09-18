import sys
import os
from subprocess import *
if len(sys.argv) <=1:
    print("Error! You should input file!")
    exit()

svmtrain = "./svm-train"
svmpredict = "./svm-predict"
grid = "./grid.py"
subset = "./subset.py"
assert os.path.exists(svmtrain),"svm-train executable not found"
assert os.path.exists(svmpredict),"svm-predict executable not found"
assert os.path.exists(grid),"grid.py not found"
assert os.path.exists(subset),"subset.py not found"
dataset = sys.argv[1]
train = "./"+dataset+".train"
test = "./"+dataset+".test"
cmd ='python "{0}" -s 1 "{1}" 350 "{2}" "{3}"'.format(subset, dataset, train, test)
print('Train dataset 350...Test dataset 150...')
Popen(cmd, shell = True).communicate()
cmd ='python "{0}" -v 5 "{1}"'.format(grid, train) #5-fold cross-validation
print('5-fold cross-validation find best c & gamma ...')
Popen(cmd, shell = True).communicate()
s=0
t=2
print("Parameter set : c")
c=input()
print("Parameter set : gamma")
g=input()

cmd='{0} -s {1} -t {2} -c {3} -g {4} "{5}"'.format(svmtrain, s, t, c, g, train)
print('Training...')
Popen(cmd, shell = True).communicate()

model = "./"+train+".model"
out = dataset+".result"

cmd='{0} "{1}" "{2}" "{3}"'.format(svmpredict, test, model, out)
print('Test..')
Popen(cmd, shell = True).communicate()