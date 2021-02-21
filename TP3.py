import numpy as np
import sys
import os, psutil
import heapq

def mem():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1000000, "Mb", file=sys.stderr)

class Graph:

    def neighbors(self, node):
        # thanks to great implementation of numpy, this is a view and not a copy
        return self.neighbors_array[self.index[node]:self.index[node+1]]

    def __init__(self, left, right, number_nodes):
        self.nb_nodes = number_nodes
        self.nb_edges = len(left)
        self.deg = np.zeros(self.nb_nodes, dtype=np.int32)
        uniques, counts = np.unique( #counting
            np.concatenate((l1, l2)), return_counts=True)
        self.deg = np.zeros(maxIdx+1, dtype=np.int32)
        for unique, count in zip(uniques, counts):
            self.deg[unique] = count
        self.index = np.zeros(self.nb_nodes+1, dtype=np.int32) # create index for O(1) neighbors access
        for i in range(1, self.nb_nodes):
            self.index[i] = self.index[i-1]+self.deg[i-1]
        self.index[self.nb_nodes] = self.index[self.nb_nodes-1]+self.deg[self.nb_nodes-1]
        mutable_index = np.copy(self.index)
        # memory of size sum number of degrees
        self.neighbors_array = np.zeros( #adjacency list in one unique array
            self.index[self.nb_nodes-1]+self.deg[self.nb_nodes-1], dtype=np.int32)
        for a, b in zip(left, right):
            self.neighbors_array[mutable_index[a]] = b
            self.neighbors_array[mutable_index[b]] = a
            mutable_index[a] += 1
            mutable_index[b] += 1

def triangles(graph, u):
    global is_neighbor_of_u
    if "is_neighbor_of_u" not in globals(): # faster than try catch
        is_neighbor_of_u = np.full(graph.nb_nodes, False)
    nb_triangles = 0
    for v in graph.neighbors(u):
      is_neighbor_of_u[v] = True
    for v in graph.neighbors(u):
      for w in graph.neighbors(v):
        if is_neighbor_of_u[w]: #O(1)
          nb_triangles += 1
    for v in graph.neighbors(u):
      is_neighbor_of_u[v] = False
    return nb_triangles//2 # remove duplicates


def clust(graph):
  CluL = 0
  CluG = 0
  triple_nb_triangles = 0
  nb_v = 0
  for u in range(graph.nb_nodes):
      d = graph.deg[u]
      if d>1: #only if deg above 1 otherwise = 0
          triple_nb_triangles += triangles(graph,u)
          nb_v += d*(d-1)/2
          CluL += 2*triangles(graph, u)/(d*(d-1))
  CluL = CluL/graph.nb_nodes
  CluG = triple_nb_triangles/nb_v
  return (CluL, CluG)

class MarkedGraph(Graph):

    def __init__(self, left, right, number_nodes):
        super().__init__(left, right, number_nodes)
        self.mark = np.full(graph.nb_nodes, True)
    
    def __init__(self, graph):
        self.nb_nodes = graph.nb_nodes
        self.nb_edges = graph.nb_edges
        self.deg = graph.deg
        self.index = graph.index
        self.neighbors_array = graph.neighbors_array
        self.mark = np.full(graph.nb_nodes, True)




def kcoeur(graph):
    graph = MarkedGraph(G)
    k = 0
    previous_size = 0 # save size of k-core before it degenerates
    size = np.sum(graph.mark)
    while size:
        heap = [(a,i) for i, a in enumerate(graph.deg) if graph.mark[i]]
        heapq.heapify(heap) #heapq algorithm
        k+=1
        toDelete = set()
        for elem in heap:
            if(elem[0] < k):
                toDelete.add(elem[1]) # below k gets deleted
        while toDelete:
            elem = toDelete.pop()
            if graph.mark[elem]:
                for neighbor in graph.neighbors(elem):
                    graph.deg[neighbor] -= 1
                    if graph.deg[neighbor] < k:
                        toDelete.add(neighbor) # delete also element getting below k during the removing process
                graph.mark[elem] = False
        previous_size = size
        size = np.sum(graph.mark)
    return max(0,k-1), previous_size



if __name__ == "__main__":
    argv = sys.argv[1:]
    estimNbAretes = int(argv[2])
# lecture du fichier et constitution du tableau des arêtes
    l1 = np.zeros(estimNbAretes, dtype=np.int32)
    l2 = np.zeros(estimNbAretes, dtype=np.int32)
    with open(argv[1], 'r') as f:
        count = 0
        for line in f:
            if line[0] != '#':
                newline = line.split()
                a = int(newline[0], 10)
                b = int(newline[1], 10)
                l1[count] = a
                l2[count] = b
                count += 1
                if count == estimNbAretes:
                    break
    maxIdx = max(np.max(l1), np.max(l2))
    l1 = l1[:count]
    l2 = l2[:count]
    G = Graph(l1, l2, maxIdx+1)
    del l1
    del l2

    if argv[0] == "triangles":
      u = int(argv[3])
      print(triangles(G, u))
    elif argv[0] == "clust":
        (clustering_local, clustering_global) = clust(G)
        print(round(clustering_local, 5))
        print(round(clustering_global, 5))
    elif argv[0] == "k-coeur":
        k, size = kcoeur(G)
        print(k)
        print(size)



