import numpy as np

a = 1.

for step in range(1, 100000):
    i = 1 / step
    if (np.log(a) > 1):
        a -= i
    else:
        a += i

    print(step)
    print(a)