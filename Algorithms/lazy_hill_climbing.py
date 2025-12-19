import networkx as nx
import numpy as np
from collections import deque
from heapq import heappush, heapreplace, nsmallest, heappop
from typing import List, Set, Callable, Any

class lazyHillClimbing:
    '''
    This algorithm is efficient than classical hill climbing in a notary sense, i.e we use sketch instead of mc estimation of influence for a candidate set
    '''

    def __init__(self, G: nx.DiGraph,  budget: float, delta_func: Any):
        self.G = G

        self.budget = budget
        self.spent = 0
        self.delta_func = delta_func

        self.heap = [] # -score, stamp, u, gain, cost

        self.nodes = list(self.G.nodes())
    
        self.seed_set = set()

        
        self.init_heap()
    
    def init_heap(self):
        for node in self.nodes:
            gain, cost = self.delta_func(node, set())
            heappush(self.heap, (-gain, node))
        

    def solve(self):
        while True:
            u, gain, cost = self.iterate()
            if u is None:
                break
            self.seed_set.add(u)
            self.spent += cost
        

        return self.seed_set

    
    def iterate(self):

        S = self.seed_set

        while self.heap:

            neg_ub, u = heappop(self.heap)

            if u in S:
                continue

            gain, cost = self.delta_func(u, S)

            if gain <= 0 or cost <= 0 or self.spent + cost > self.budget:
                continue

            best_ub = 0.0 if not self.heap else -self.heap[0][0]

            if gain >= best_ub:
                return u, gain, cost
            
            heappush(self.heap, (-gain, u))

        
        return None, 0.0, 0.0

