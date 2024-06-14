# -*- coding: utf-8 -*-
"""
SER501 Assignment 3 scaffolding code
created by: Xiangyu Guo
"""
import sys
import heapq
# =============================================================================


class Graph(object):
    """docstring for Graph"""
    # user_defined_vertices = []
    dfs_timer = 0

    def __init__(self, vertices, edges):
        super(Graph, self).__init__()
        n = len(vertices)
        self.matrix = [[0 for x in range(n)] for y in range(n)]
        # self.matrixT = [[0 for x in range(n)] for y in range(n)]
        self.vertices = vertices
        self.edges = edges
        # self.T = False
        self.outdegree = {vertex: 0 for vertex in self.vertices}
        self.indegree = {vertex: 0 for vertex in self.vertices}
        self.discover = [0] * n
        self.finish = [0] * n
        # Graph.user_defined_vertices = [False] * n
        self.parent = [None] * n
        self.weight = [sys.maxsize] * n
        self.in_mst = [False] * n
        for edge in edges:
            x = vertices.index(edge[0])
            y = vertices.index(edge[1])
            self.matrix[x][y] = edge[2]

    def display(self):
        print(self.vertices)
        for i, v in enumerate(self.vertices):
            print(v, self.matrix[i])

    def transpose(self):
        # Method 1:
        self.matrix = [list(row) for row in zip(*self.matrix)]

        # # Method 2:
        # size = len(self.matrix)
        # for i in range(size):
        #     for j in range(size):
        #         self.matrixT[j][i] = self.matrix[i][j]
        # self.matrix = [row[:] for row in self.matrixT]
        # self.T = True

    def in_degree(self):
        # Method 1
        print("Method 1 for in degree")
        print("========================")
        for edge in self.edges:
            if edge[1] in self.indegree:
                self.indegree[edge[1]] += 1

        print("In degree of the graph:")
        for key, value in self.indegree.items():
            print(f"Vertex : {key} Degree : {value}")

        # # Method 2:
        # print("Method 2 for in degree")
        # print("========================")
        # in_degrees = [0] * len(self.vertices)
        # for i in range(len(self.vertices)):
        #     for j in range(len(self.vertices)):
        #         if self.matrix[j][i] != 0:
        #             in_degrees[i] += 1

        # print("In degree of the graph:")
        # for i, degree in enumerate(in_degrees):
        #     print(f"Vertex: {self.vertices[i]} Degree: {degree}")

    def out_degree(self):
        # Method 1:
        print("Method 1 for out degree")
        print("========================")
        for edge in self.edges:
            if edge[0] in self.outdegree:
                self.outdegree[edge[0]] += 1

        print("Out degree of the graph:")
        for key, value in self.outdegree.items():
            print(f"Vertex : {key} Degree : {value}")

        # # Method 2:
        # print("Method 2 for out degree")
        # print("========================")
        # out_degrees = [sum(row) for row in self.matrix]

        # print("Out degree of the graph:")
        # for i, degree in enumerate(out_degrees):
        #     print(f"Vertex: {self.vertices[i]} Degree: {degree}")

    def dfs_visit(self, u):
        Graph.dfs_timer += 1
        self.discover[u] = Graph.dfs_timer
        # Graph.user_defined_vertices[u] = True
        for v in range(len(self.vertices)):
            # if self.matrix[u][v] != 0 and not Graph.user_defined_vertices[v]:
            if self.matrix[u][v] != 0 and self.discover[v] == 0:
                self.dfs_visit(v)
        Graph.dfs_timer += 1
        self.finish[u] = Graph.dfs_timer

    def dfs_on_graph(self):
        for u in range(len(self.vertices)):
            # if not Graph.user_defined_vertices[u]:
            if self.discover[u] == 0:
                self.dfs_visit(u)
        self.print_discover_and_finish_time(self.discover, self.finish)

    def prim(self, root):
        n = len(self.vertices)
        root_index = self.vertices.index(root)
        self.weight[root_index] = 0

        self.print_d_and_pi("Initial", self.weight, self.parent)

        for iteration in range(n):
            min_index = -1
            min_value = sys.maxsize
            for v in range(n):
                if self.weight[v] < min_value and self.in_mst[v] is False:
                    min_value = self.weight[v]
                    min_index = v

            self.in_mst[min_index] = True
            for v in range(n):
                if self.matrix[min_index][v] > 0 and self.in_mst[v] is False and self.weight[v] > self.matrix[min_index][v]:
                    self.weight[v] = self.matrix[min_index][v]
                    self.parent[v] = self.vertices[min_index]

            self.print_d_and_pi(iteration, self.weight, self.parent)

    def bellman_ford(self, source):
        d = [sys.maxsize] * len(self.vertices)
        pi = [None] * len(self.vertices)
        source_index = self.vertices.index(source)
        d[source_index] = 0

        self.print_d_and_pi("Initial", d, pi)

        for iteration in range(len(self.vertices) - 1):
            for u, v, w in self.edges:
                u_index = self.vertices.index(u)
                v_index = self.vertices.index(v)
                if d[u_index] != sys.maxsize and d[u_index] + w < d[v_index]:
                    d[v_index] = d[u_index] + w
                    pi[v_index] = u

            self.print_d_and_pi(iteration, d, pi)

        for u, v, w in self.edges:
            u_index = self.vertices.index(u)
            v_index = self.vertices.index(v)
            if d[u_index] != sys.maxsize and d[u_index] + w < d[v_index]:
                print("No Solution")
                return

    def dijkstra(self, source):
        n = len(self.vertices)
        d = [sys.maxsize] * n
        pi = [None] * n
        source_index = self.vertices.index(source)
        d[source_index] = 0
        queue = [(0, source_index)]
        visited = set()

        self.print_d_and_pi("Initial", d, pi)

        while queue:
            (dist, u_index) = heapq.heappop(queue)
            if u_index in visited:
                continue

            visited.add(u_index)
            u = self.vertices[u_index]

            for v_index, v in enumerate(self.vertices):
                if self.matrix[u_index][v_index] != 0:
                    if d[u_index] + self.matrix[u_index][v_index] < d[v_index]:
                        d[v_index] = d[u_index] + self.matrix[u_index][v_index]
                        pi[v_index] = u
                        heapq.heappush(queue, (d[v_index], v_index))

            self.print_d_and_pi(len(visited) - 1, d, pi)

    def print_d_and_pi(self, iteration, d, pi):
        assert((len(d) == len(self.vertices)) and
               (len(pi) == len(self.vertices)))

        print("Iteration: {0}".format(iteration))
        for i, v in enumerate(self.vertices):
            val = 'inf' if d[i] == sys.maxsize else d[i]
            print("Vertex: {0}\td: {1}\tpi: {2}".format(v, val, pi[i]))

    def print_discover_and_finish_time(self, discover, finish):
        assert((len(discover) == len(self.vertices)) and
               (len(finish) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDiscovered: {1}\tFinished: {2}".format(
                    v, discover[i], finish[i]))

    def print_degree(self, degree):
        assert((len(degree) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDegree: {1}".format(v, degree[i]))


def main():
    # Thoroughly test your program and produce useful output.
    # Q1 and Q2
    graph = Graph(['1', '2'], [('1', '2', 1)])
    graph.display()
    graph.transpose()
    graph.display()
    graph.transpose()
    graph.display()
    graph.in_degree()
    graph.out_degree()
    graph.print_d_and_pi(1, [1, sys.maxsize], [2, None])
    graph.print_degree([1, 0])
    graph.print_discover_and_finish_time([0, 2], [1, 3])

    # Q3
    graph = Graph(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
                  [('q', 's', 1),
                   ('s', 'v', 1),
                   ('v', 'w', 1),
                   ('w', 's', 1),
                   ('q', 'w', 1),
                   ('q', 't', 1),
                   ('t', 'x', 1),
                   ('x', 'z', 1),
                   ('z', 'x', 1),
                   ('t', 'y', 1),
                   ('y', 'q', 1),
                   ('r', 'y', 1),
                   ('r', 'u', 1),
                   ('u', 'y', 1)])
    graph.display()
    graph.dfs_on_graph()

    # Q4 - Prim
    graph = Graph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                  [('A', 'H', 6),
                   ('H', 'A', 6),
                   ('A', 'B', 4),
                   ('B', 'A', 4),
                   ('B', 'H', 5),
                   ('H', 'B', 5),
                   ('B', 'C', 9),
                   ('C', 'B', 9),
                   ('G', 'H', 14),
                   ('H', 'G', 14),
                   ('F', 'H', 10),
                   ('H', 'F', 10),
                   ('B', 'E', 2),
                   ('E', 'B', 2),
                   ('G', 'F', 3),
                   ('F', 'G', 3),
                   ('E', 'F', 8),
                   ('F', 'E', 8),
                   ('D', 'E', 15),
                   ('E', 'D', 15)])
    graph.prim('G')

    # Q5
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 7),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('z')

    # Q5 alternate
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 4),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('s')

    # Q6
    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('s', 't', 3),
                   ('s', 'y', 5),
                   ('t', 'x', 6),
                   ('t', 'y', 2),
                   ('x', 'z', 2),
                   ('y', 't', 1),
                   ('y', 'x', 4),
                   ('y', 'z', 6),
                   ('z', 's', 3),
                   ('z', 'x', 7)])
    graph.dijkstra('s')


if __name__ == '__main__':
    main()
