import numpy as np
import matplotlib.pyplot as plt


def h_teta(t0,t1,x):
	return t0 + t1 * x

def somme_teta(t_0,t_1,pts,pts_size,isTeta1,isCost):
	sum_points = 0
	for i in range(pts_size):
		point_value = h_teta(t_0,t_1,pts[i][0]) - pts[i][1]

		if isCost:
			point_value = point_value ** 2

		if isTeta1:
			point_value *= pts[i][0]
			
		sum_points += point_value
	return sum_points


def calctetacost(t_0,t_1,pts):
	teta_0 = next_0 = t_0
	teta_1 = next_1 = t_1
	points = pts
	points_size = points.__len__()
	min_cost = None

	for step in range(1,300000):
		alpha = 1/step
		next_0 = teta_0 - (alpha / points_size) * (somme_teta(teta_0,teta_1,points, points_size, 0, 0))
		next_1 = teta_1 - (alpha / points_size) * (somme_teta(teta_0,teta_1,points, points_size, 1, 0))
		teta_0 = next_0
		teta_1 = next_1

		cost = (1 / (points_size * 2)) * (somme_teta(teta_0,teta_1,points, points_size, 0, 1))

		if min_cost == None or cost < min_cost:
			min_cost = cost


	print(teta_0)
	print(teta_1)
	print(min_cost)

	plt.close()
	plt1 = plt.subplot(211)
	x,y = [elem[0] for elem in points],[elem[1] for elem in points]
	plt1.scatter(x,y)
	plot = np.arange(1,4,1)
	y_chapeau = [h_teta(teta_0,teta_1,elem[0]) for elem in points]
	plt1.plot(plot, y_chapeau)		

	plt.show()

#calctetacost(1,1,[[10,4],[1,5],[2,3]])
calctetacost(1,1,[[13,13],[2,2],[5,18]])
