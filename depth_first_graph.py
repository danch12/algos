# dfs
# marked = [False for n in range(graph.size())]
# def dfs_preorder(G,v):
#     visit(v)
#     marked[v]= True
#     for neighbour in G.neighbours(v):
#         if not marked[neighbour]:
#             dfs_preorder(G,w)

# marked = [False for n in range(graph.size())]
# def dfs_postorder(G,v):
#     marked[v]= True
#     for neighbour in G.neighbours(v):
#         if not marked[neighbour]:
#             dfs_postorder(G,w)
#     visit(v)

class Graph:
    def __init__(self, nodes: int):
        self.nodes = nodes
        self.neighbours = {}
        for i in range(nodes):
            self.neighbours[i] = []

    def add_neighbour(self, a, b):
        if b not in self.neighbours[a]:
            self.neighbours[a].append(b)
        if a not in self.neighbours[b]:
            self.neighbours[b].append(a)

    def get_neighbours(self, node):
        return self.neighbours[node]

    def size(self):
        return self.nodes


def visit(node):
    print(f"{node}")

def dfs_postorder(graph, v, marked):
    marked[v]= True
    for neighbour in graph.get_neighbours(v):
        if not marked[neighbour]:
            dfs_postorder(graph, neighbour, marked)
    visit(v)

def dfs_preorder(graph, v, marked):
    visit(v)
    marked[v]= True
    for neighbour in graph.get_neighbours(v):
        if not marked[neighbour]:
            dfs_preorder(graph, neighbour, marked)

def dfs_traverse(order,graph, start_node):
    marked = [False for _ in range(graph.size())]
    order(graph, start_node,marked)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    graph = Graph(6)
    graph.add_neighbour(0,1)
    graph.add_neighbour(1,3)
    graph.add_neighbour(1, 2)
    graph.add_neighbour(2, 3)
    graph.add_neighbour(3, 4)
    graph.add_neighbour(5, 4)
    dfs_traverse(dfs_postorder,graph,0)

