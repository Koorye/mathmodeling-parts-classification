import numpy as np
import matplotlib.pyplot as plt
from sko.SA import SA_TSP
from scipy import spatial

FREQUENCY = 5000  # Times of saving a processing image
TITLE = 'TSP'
X_LABEL = 'Y'
Y_LABEL = 'X'

count = 0  # Timer of counting times the calculating does


def get_point_position(points, coordinate):
    points_position = np.concatenate([points, [points[0]]])
    return coordinate[points_position, :]


def draw_coordinate(points, coordinate):
    pos_x = []
    pos_y = []
    points_coordinate = get_point_position(points, coordinate)
    for each in points_coordinate:
        pos_x.append(each[0])
        pos_y.append(each[1])

    plt.figure(figsize=(4,4))
    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.scatter(pos_x, pos_y)
    plt.plot(pos_x, pos_y, c='coral')
    plt.savefig('result/TSP_' + str(int(count / FREQUENCY) + 1) + '.png')
    plt.show()


def cal_total_distance(routine):
    global count
    if count % FREQUENCY == 0:
        temp_points_coordinate = get_point_position(routine, points_coordinate)
        draw_coordinate(routine, temp_points_coordinate)
    count += 1

    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


points_coordinate = np.loadtxt('data/tsp_data.csv', delimiter=',')
num_points = points_coordinate.shape[0]
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)

best_points, best_distance = sa_tsp.run()

print('best points: ')
print(best_points)
print('best distance: ')
print(best_distance)
print('total distance: ')
print(cal_total_distance(best_points))

print(points_coordinate)
draw_coordinate(best_points, points_coordinate)
