import numpy as np
import sys
import os, psutil 

def mem() :
	process = psutil.Process(os.getpid()) 
	print(process.memory_info().rss / 1000000, "Mb", file=sys.stderr)

class Graph:
    def __init__(self, links, number_nodes):
        self.deg = np.zeros(number_nodes, dtype = np.int32)
        for a, b in links:
            self.deg[a]+=1
            self.deg[b]+=1
        self.index = np.zeros(number_nodes, dtype = np.int32)
        self.array = np.zeros(sum(self.deg), dtype = np.int32)

        

if __name__ == "__main__":
    argv = sys.argv[1:]
    assert(len(argv) < 2)
    estimNbAretes = int((os.popen('wc -l web-BerkStan.txt').read()).split()[0])
    links = np.zeros((estimNbAretes,2), dtype=np.int32)
    maxIdx = 0
    with open(argv[0], 'r') as f:
        count=0
        for line in f:
            if not line[0]=='#':
                newline=line.split()
                a = int(newline[0])
                b = int(newline[1])
                links[count][0]=a
                links[count][1]=b
                maxIdx = max(maxIdx, a, b)
                count+=1
    G = Graph(links, count)