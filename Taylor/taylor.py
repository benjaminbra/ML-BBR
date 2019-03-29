import matplotlib.pyplot as plt



def factorial(n):
    if n >= 1:
        return n * factorial(n - 1)
    else:
        return 1

plt.close()
sum = 0
p_x = []
p_y = []
n = 30
x = 1

for i in range(0, n):
    sum += ((x ** i) / factorial(i))
    p_x.append(i)
    p_y.append(sum)
    print('Sum : ', sum, ' i : ', i)

plt.plot(p_x,p_y)
plt.show()