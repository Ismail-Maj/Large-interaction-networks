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


#BFS ne renvoyant que le noeud le plus éloigné
def BFS_without_dist(Graph, u, NbVert):
    seen=np.zeros(NbVert, dtype=np.int32)
    to_visit=deque([])
    seen[u]=1
    for w in Graph.neighbors(u):
        seen[w]=1
        to_visit.append(w)
    while to_visit:
        w=to_visit.popleft()
        for z in G.neighbors(w):
            if not seen[z]:
              to_visit.append(z)
              seen[z]=1
    furthest = w
    return furthest

#BFS renvoyant le noeud le plus éloigné de u et la distance de tous les noeuds à u
def BFS_with_dist(Graph, u, NbVert):
    dist=np.zeros(NbVert, dtype=np.int32)
    seen=np.zeros(NbVert, dtype=np.int32)
    to_visit=deque([])
    seen[u]=1
    for w in Graph.neighbors(u):
        seen[w]=1
        dist[w]=1
        to_visit.append(w)
    w=0
    while to_visit:
        w=to_visit.popleft()
        for z in G.neighbors(w):
            if not seen[z]:
              to_visit.append(z)
              seen[z]=1
              dist[z]=dist[w]+1
    furthest = w
    return (furthest,dist,seen)

#heuristique du double-BFS
def doubleBFS_heuristic(Graph, u, NbVert):
    v=BFS_without_dist(Graph, u, NbVert)
    (w,dist,seen) = BFS_with_dist(Graph, v, NbVert)
    return(v,w,dist[w])

#BFS renvoyant le noeud le plus éloigné, sa distance, et le noeud au milieu du chemin
def BFS_with_dist_and_middle(Graph, u, NbVert):
    dist=np.zeros(NbVert, dtype=np.int32)
    seen=np.zeros(NbVert, dtype=np.int32)
    pred=np.zeros(NbVert, dtype=np.int32)
    to_visit=deque([])
    seen[u]=1
    pred[u]=-1
    for w in Graph.neighbors(u):
        seen[w]=1
        dist[w]=1
        pred[w]=u
        to_visit.append(w)
    while to_visit:
        w=to_visit.popleft()
        for z in G.neighbors(w):
            if not seen[z]:
              to_visit.append(z)
              seen[z]=1
              dist[z]=dist[w]+1
              pred[z]=w
    furthest = w
    distance=dist[furthest]
    middle=w
    for i in range(distance//2):
        middle=pred[middle]
    return (furthest,distance,middle)

#heuristique du double-double-BFS
def double_doubleBFS_heuristic(Graph, u, NbVert):
    v=BFS_without_dist(Graph, u ,NbVert)
    (w,dist1,middle)=BFS_with_dist_and_middle(Graph,v,NbVert)
    return doubleBFS_heuristic(Graph, middle ,NbVert)[2]

#sum-sweep
def sum_sweep(Graph, u, NbVert):
    sumdist = np.zeros(NbVert, dtype=np.int32)
    ecc = np.zeros(4, dtype=np.int32)
    next_chosen = u
    for i in range (4):
        (next_furthest, dist, seen) = BFS_with_dist(Graph, next_chosen, NbVert)
        ecc[i] = dist[next_furthest]
        sumdist = np.add(sumdist, dist)
        next_chosen = np.argmax(sumdist)
    return np.amax(ecc)

#Takes et Koster
def diametre(Graph, u, NbVert):
    eccsup = np.full(NbVert,NbVert-1)   #une composante connexe est de diametre strictement inférieur au nombre de sommets (cas critique: la chaîne)
    (f,d,seen)= BFS_with_dist(Graph, u, NbVert)
    for v in range(NbVert):     #on attribue une eccsup de -1 aux sommets qui ne sont pas dans la composante connexe pour ne jamais les considérer
        if not seen[v]:
            eccsup[v]=-1
    ecc = d[f]
    eccsup[u]=ecc
    maxeccsup = NbVert-1
    diamlow=ecc
    while maxeccsup > diamlow:
        a = np.argmax(eccsup)
        maxeccsup = eccsup[a]
        (f,d,s) = BFS_with_dist(Graph, a, NbVert)
        ecc = d[f]
        eccsup[a]=ecc
        if ecc>diamlow:
            diamlow = ecc
    return diamlow

#trouver un sommet de la plus grande composante connexe
def sommet_plus_grande_comp_connexe(Graph, u, NbVert):
    sommet = u
    (f,d,seen)=BFS_with_dist(Graph, u, NbVert)
    size_comp=np.shape(np.nonzero(seen))[1]
    while np.shape(np.nonzero(seen))[1]!=NbVert:
        new_sommet=np.argmin(seen)
        (newf,newd,new_seen)=BFS_with_dist(Graph,new_sommet,NbVert)
        seen = np.add(seen,new_seen)
        new_size_comp = np.shape(np.nonzero(new_seen))[1]
        if new_size_comp > size_comp:
            sommet=new_sommet
            size_comp = new_size_comp
    return sommet
        
        
#===============================================================================================================
if __name__ == "__main__":
    argv = sys.argv[1:]
    estimNbAretes = int(argv[2])
#lecture du fichier et constitution du tableau des arêtes    
    l1 = np.zeros(estimNbAretes, dtype=np.int32)
    l2 = np.zeros(estimNbAretes, dtype=np.int32)
    with open(argv[1], 'r') as f:
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
    
    u = int(argv[3])
    
#2-sweep    
    if argv[0]=="2-sweep":
        (v,w,diam) = doubleBFS_heuristic(G, u, maxIdx+1)
        print("v="+str(v))
        print("w="+str(w))
        print("diam>="+str(diam))
        print("\n")

#4-sweep
    if argv[0]=="4-sweep":
        print("diam>="+str(double_doubleBFS_heuristic(G, u , maxIdx+1)))
        print("\n")
#sum-sweep
    if argv[0]=="sum-sweep":
        print("diam>="+str(sum_sweep(G, u, maxIdx+1)))
        print("\n")
#Takes et Koster
    if argv[0]=="diametre":
        print("diam="+str(diametre(G, u, maxIdx+1)))

#plus grande composante connexe
    if argv[0]=="vertex":
        print("un bon sommet est "+str(sommet_plus_grande_comp_connexe(G,u,maxIdx+1)))