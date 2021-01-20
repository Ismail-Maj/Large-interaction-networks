import numpy as np
import sys
import os, psutil
from collections import deque 

def mem() :
	process = psutil.Process(os.getpid()) 
	print(process.memory_info().rss / 1000000, "Mb", file=sys.stderr)

class Graph:

    def neighbors(self, node):
        return self.neighbors_array[self.index[node]:self.index[node]+self.deg[node]] # thanks to great implementation of numpy, this is a view and not a copy
    def __init__(self, left, right, number_nodes):
        self.nb_nodes = number_nodes
        self.nb_edges = len(left)
        self.deg = np.zeros(self.nb_nodes, dtype = np.int32)
        uniques,counts = np.unique(np.concatenate((l1,l2)), return_counts=True)
        self.deg = np.zeros(maxIdx+1, dtype = np.int32)
        for unique, count in zip(uniques,counts):
            self.deg[unique] = count
        self.index = np.zeros(self.nb_nodes, dtype = np.int32)
        for i in range(1, self.nb_nodes):
            self.index[i] = self.index[i-1]+self.deg[i-1]
        mutable_index = np.copy(self.index)
        self.neighbors_array = np.zeros(self.index[self.nb_nodes-1]+self.deg[self.nb_nodes-1], dtype = np.int32) # memory of size sum number of degrees
        for a, b in zip(left, right):
            self.neighbors_array[mutable_index[a]] = b
            self.neighbors_array[mutable_index[b]] = a
            mutable_index[a]+=1
            mutable_index[b]+=1      

if __name__ == "__main__":
    argv = sys.argv[1:]
    estimNbAretes = int(argv[1])
#lecture du fichier et constitution du tableau des arêtes    
    l1 = np.zeros(estimNbAretes, dtype=np.int32)
    l2 = np.zeros(estimNbAretes, dtype=np.int32)
    with open(argv[0], 'r') as f:
        count=0
        for line in f:
            if line[0]!='#':
                newline=line.split()
                a = int(newline[0],10)
                b = int(newline[1],10)
                l1[count]=a
                l2[count]=b
                count+=1
    maxIdx = max(np.max(l1),np.max(l2))
    l1 = l1[:count]
    l2 = l2[:count]
    G = Graph(l1, l2, maxIdx+1)
    del l1
    del l2
    mem()


#on peut retourner le nombre de sommets et d'arêtes
    print("n="+str(G.nb_nodes))
    print("m="+str(G.nb_edges))

#calcul et retour du degré max    
    degMax=np.max(G.deg)

    print("degmax="+str(degMax))
    
#calcul et retour de las distance entre u et v
    u=int(argv[2])
    v=int(argv[3])
    res = -1
  #on procède à un BFS partant de u en utilisant une file pour la visite (to_visit) et en retenant les noeuds vus (seen) et leur distance
    if u==v:
      res = 0
    else:
      seen=np.zeros(maxIdx+1, dtype=np.int32)
      dist=np.zeros(maxIdx+1, dtype=np.int32) 
      seen[u]=1
      to_visit=deque([]) 
      for w in G.neighbors(u):
        seen[w]=1
        dist[w]=1
        to_visit.append(w)
      while to_visit:
        w=to_visit.popleft()
        if w==v:
            res = dist[w]
            break
        else:
          for z in G.neighbors(w):
            if not seen[z]:
              to_visit.append(z)
              seen[z]=1
              dist[z]=dist[w]+1
    mem()
    if res == -1:                 
      print("dist="+str(float('inf')))
    else:
        print("dist="+str(res))
            