import numpy as np
import sys
import os, psutil 

def mem() :
	process = psutil.Process(os.getpid()) 
	print(process.memory_info().rss / 1000000, "Mb", file=sys.stderr)

class Graph:

    def adjacency(self, node):
        return self.adjacency_array[self.index[node]:self.index[node]+self.deg[node]]
    def __init__(self, edges, number_nodes):
        self.nb_nodes = number_nodes
        self.nb_edges = len(edges)

        self.deg = np.zeros(self.nb_nodes, dtype = np.int32)
        for a, b in edges:
            self.deg[a]+=1
            self.deg[b]+=1

        self.index = np.zeros(self.nb_nodes, dtype = np.int32)
        for i in range(1, self.nb_nodes):
            self.index[i] = self.index[i-1]+self.deg[i-1]

        mutable_index = np.copy(self.index) # memory O(nb_nodes) not a problem

        self.adjacency_array = np.zeros(sum(self.deg), dtype = np.int32) # memory of size sum number of degrees

        for a, b in edges:
            self.adjacency_array[mutable_index[a]] = b
            self.adjacency_array[mutable_index[b]] = a
            mutable_index[a]+=1
            mutable_index[b]+=1
        




        

if __name__ == "__main__":
    argv = sys.argv[1:]
    assert(len(argv) < 2)
    estimNbAretes = int((os.popen('wc -l web-BerkStan.txt').read()).split()[0])
    edges = np.zeros((estimNbAretes,2), dtype=np.int32)
    maxIdx = 0
    with open(argv[0], 'r') as f:
        count=0
        for line in f:
            if not line[0]=='#':
                newline=line.split()
                a = int(newline[0])
                b = int(newline[1])
                edges[count][0]=a
                edges[count][1]=b
                maxIdx = max(maxIdx, a, b)
                count+=1
    G = Graph(edges, count)