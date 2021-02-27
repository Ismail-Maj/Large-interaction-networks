import numpy as np
import sys
import os
import psutil
import math
import random


def mem():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1000000, "Mb", file=sys.stderr)


def given_deg(seq):
    E = np.zeros(np.sum(seq), dtype=np.int32)
    idx = 0
    for i, deg in enumerate(seq):
        for j in range(deg):
            E[idx] = i
            idx += 1
    random.shuffle(E)
    return E


def sqrt_deg(n):
    seq = np.zeros(n, dtype=np.int32)
    for i in range(n):
        seq[i] = int(math.sqrt(i+1))
    if sum(seq) % 2 != 0:
        seq[n-1] += 1
    return given_deg(seq)


def print_graph(seq):  # redirect to file, so it can be used with tp3
    for i in range(0, len(seq), 2):
        print(seq[i], seq[i+1])


def power_deg(n, gamma):
    rep = np.zeros(n, dtype=np.float64)
    for k in range(n):
        rep[k] = (k**(-gamma) if k else 0)
    rep = np.rint(rep / (np.sum(rep)/n)).astype(np.int32)
    # Vérification de Parité de la somme des degrés
    sum_deg = 0
    for k in range(n):
  	    sum_deg += k*rep[k]
    if sum_deg % 2:
  	    rep[1] += 1
  # Création de Seq
    seq = np.zeros(sum(rep), dtype=np.int32)
    j = 0
    idx = 0
    for i in range(sum(rep)):
        c = rep[i]
        while c > 0:
            seq[j]=i
            j+=1
            c-=1
    return given_deg(seq)

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[0] == "exemple":
        seq = given_deg([1, 2, 1, 4])
        print_graph(seq)
    elif argv[0] == "racine":
        n = int(argv[1])
        print_graph(sqrt_deg(n))
        mem()
    elif argv[0] == "puissance":
        n = int(argv[1])
        gamma = float(argv[2])
        print_graph(power_deg(n, gamma))
        

