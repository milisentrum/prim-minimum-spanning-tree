import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import heapq

def prim_algorithm(vertices, edges, start_vertex, weight_method):
    connected = set([start_vertex])
    mst = []

    # Приоритетная очередь для рёбер, исходящих из start_vertex
    eligible_edges = [(weight_method(*edge[2:]), edge[:2]) for edge in edges if start_vertex in edge[:2]]
    heapq.heapify(eligible_edges)

    while len(connected) < len(vertices) and eligible_edges:
        weight, edge = heapq.heappop(eligible_edges)
        from_vertex, to_vertex = edge
        if to_vertex not in connected or from_vertex not in connected:
            connected.add(to_vertex)
            connected.add(from_vertex)
            mst.append((from_vertex, to_vertex, weight))

            # Добавление новых рёбер в приоритетную очередь
            for edge in edges:
                if edge[0] == to_vertex and edge[1] not in connected:
                    heapq.heappush(eligible_edges, (weight_method(*edge[2:]), edge[:2]))
                elif edge[1] == to_vertex and edge[0] not in connected:
                    heapq.heappush(eligible_edges, (weight_method(*edge[2:]), edge[:2]))

    return mst

# Функция для визуализации MST
def visualize_mst(points, mst):
    plt.figure(figsize=(10, 6))
    # Рисуем точки
    plt.scatter(points[:, 0], points[:, 1], color='blue')

    # Рисуем рёбра MST
    for from_vertex, to_vertex, _ in mst:
        point1 = points[from_vertex]
        point2 = points[to_vertex]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-')

    plt.title("Минимальное покрывающее дерево (MST)")
    plt.show()

# Генерация искусственного набора данных
np.random.seed(0)
points = np.random.rand(10, 2)  # 10 точек в 2D

# Вычисление матрицы расстояний и создание списка рёбер
dist_matrix = distance_matrix(points, points)
edges = [(i, j, dist_matrix[i, j]) for i in range(len(points)) for j in range(i+1, len(points))]

# Параметры для экспериментов
start_vertex = 0  # Начальная точка
weight_method = lambda weight: weight  # Метод расчета веса ребра

# Вызов алгоритма Прима
mst = prim_algorithm(range(len(points)), edges, start_vertex, weight_method)

# Визуализация
visualize_mst(points, mst)
