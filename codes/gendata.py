import random


with open('data.txt', 'w') as f:
    for i in range(0, 2000):
        x = random.uniform(1, 100)
        y = random.uniform(1, 100)
        if y // 50 == 1:
            c = 1
        else:
            c = 0
        f.write(str(x)+','+str(y)+','+str(c)+'\n')
