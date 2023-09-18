import random

# y = mx + b
m = 3
b = 7

points = []
count_positive=0
count_negative=0
label=1
while count_positive+count_negative<30 :
    x = random.randint(-50, 50)
    y = random.randint(-50, 50)
    if abs(y - (m*x + b)) > 10:
        if y> (m*x+b) and count_positive<15:
            label=1
            count_positive+=1
        elif y< (m*x+b) and count_negative<15:
            label=-1
            count_negative+=1
        else:
            continue
        point = f"{x:>10.2f}   {y:>10.2f}     {label}\n"
        points.append(point)
# 將座標寫入檔案
with open('points30.txt', 'w') as f:
    f.writelines(points)

