import numpy as np
import matplotlib.pyplot as plt

def h_teta(t0, t1, x):
    return t0 + t1 * x


def somme_teta(t_0, t_1, pts, pts_size, isTeta1, isCost):
    sum_points = 0
    for i in range(pts_size):
        point_value = h_teta(t_0, t_1, pts[i][0]) - pts[i][1]

        if isCost:
            point_value = point_value ** 2

        if isTeta1:
            point_value *= pts[i][0]

        sum_points += point_value
    return sum_points


def calctetacost(t_0, t_1, pts):
    teta_0 = next_0 = t_0
    teta_1 = next_1 = t_1
    points = pts
    points_size = points.__len__()
    min_cost = None
    min_teta_0 = None
    min_teta_1 = None

    for step in range(1, 2000000):
        alpha = 1 / 10000000 / step
        next_0 = teta_0 - (alpha / points_size) * (somme_teta(teta_0, teta_1, points, points_size, 0, 0))
        next_1 = teta_1 - (alpha / points_size) * (somme_teta(teta_0, teta_1, points, points_size, 1, 0))
        teta_0 = next_0
        teta_1 = next_1

        cost = (1 / (points_size * 2)) * (somme_teta(teta_0, teta_1, points, points_size, 0, 1))

        if min_cost is None or cost < min_cost:
            min_teta_0 = teta_0
            min_teta_1 = teta_1
            min_cost = cost

    return {'teta0': min_teta_0, 'teta1': min_teta_1, 'cost': min_cost}

# Jeux de données
# Dans un soucis de simplicité, on va transformer les dates en nombres simples
# La date du 16-03-2019 sera 0
# La date du 03-04-2019 sera 18
# On cherche donc la valeur pour 19

dataset = [
    [0.0, 81682.0],
    [2.0, 81720.0],
    [4.0, 81760.0],
    [8.0, 81826.0],
    [9.0, 81844.0],
    [10.0, 81864.0],
    [11.0, 818881.0],
    [12.0, 81900.0],
    [14.0, 81933.0],
    [18.0, 82003.0]
]

tetas = calctetacost(1,1,dataset)

print(tetas)

# Je dois encore estimer quel sera le prochain point