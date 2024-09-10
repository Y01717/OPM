import math
import numpy as np

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def build_adjacency_matrix(data, threshold):
    num_points = len(data)
    adjacency_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        count = 0
        for j in range(i+1, num_points):
            coord1 = (data[i, 0], data[i, 1])  # Longitude, Latitude of point i
            coord2 = (data[j, 0], data[j, 1])  # Longitude, Latitude of point j
            distance = euclidean_distance(coord1, coord2)  # Calculate the distance between coordinates.  Euclidean Distance!!!!!!!!

            if distance <= threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                count += 1
                if count >= 8:
                    break

        # print(f"Point {i}: {count} neighbors")
    return adjacency_matrix




















