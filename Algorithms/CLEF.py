from lazy_hill_climbing import lazyHillClimbing
from collections import deque

class CLEF:

    def __init__(self, G, budget):
        self.G = G
        self.budget = budget

        self.node_to_idx = {node:idx for idx, node in enumerate(self.G.nodes())}
        self.idx_to_nodes = {idx:node for node, idx in self.node_to_idx.items()}

        self.node_to_cost = {u: self.G.nodes[u].get("cost", 1.0) for u in self.G.nodes()}

    def covereage(self, u):

        seen = {u}
        q = deque([u])

        while q:
            curr = q.popleft()

            for neighbor in self.G.successors(curr):
                if neighbor not in seen:
                    seen.add(neighbor)
                    q.append(neighbor)

        return seen
            


    def base_delta(self, u, S):
        u_coverage = self.covereage(u)

        union_set = set()
        for s in S:
            union_set = union_set.union(self.covereage(s))

        gain = len(u_coverage - union_set)
        cost = self.node_to_cost[u]

        return gain, cost
    
    def base_value_func(self, S):
        union_set = set()
        cost = 0
        for s in S:
            union_set = union_set.union(self.covereage(s))
            cost += self.node_to_cost[s]

        gain = len(union_set)

        return gain
    

    def delta_unit(self, u, S):
        gain,cost = self.base_delta(u, S)
        return gain, cost

    def delta_ratio(self, u, S):
        gain,cost = self.base_delta(u, S)
        if cost == 0:
            return 0, cost
        return gain/cost, cost

    def solve(self):

        lhc_1  = lazyHillClimbing(self.G, self.budget, self.delta_unit)
        lhc_2 = lazyHillClimbing(self.G, self.budget, self.delta_ratio)

        sol_set_1 = lhc_1.solve()
        sol_set_2 = lhc_2.solve()

        v1 = self.base_value_func(sol_set_1)
        v2 = self.base_value_func(sol_set_2)

        if v1 >= v2:
            return sol_set_1

        else:
            return sol_set_2



#### LLM GENERATED CODE
if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt


    # ------------------------------------------------------------
    # Example graph where NO single node can cover everything
    # (three directed "regions" that are not reachable from each other)
    # ------------------------------------------------------------
    G = nx.DiGraph()

    # Region A: 0 -> 1 -> 2 -> 3 -> 4
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    # Region B: 5 -> 6 -> 7 -> 8 -> 9
    G.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9)])

    # Region C: 10 -> 11 -> 12
    G.add_edges_from([(10, 11), (11, 12)])

    # Add a couple “local” extra edges (still no global super-root)
    G.add_edges_from([(0, 2), (5, 7), (10, 12)])

    # Node costs
    costs = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
        5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0,
        10: 1.0, 11: 1.0, 12: 1.0,
    }
    nx.set_node_attributes(G, costs, "cost")

    budget = 3.0  # should typically pick ~3 seeds here

    # ------------------------------------------------------------
    # Run your implementation
    # ------------------------------------------------------------
    solver = CLEF(G, budget=budget)
    S = solver.solve()  # your selected "influential" nodes (seeds)

    covered = set()
    for s in S:
        covered |= solver.covereage(s)

    fS = solver.base_value_func(S)
    total_cost = sum(costs.get(u, 1.0) for u in S)

    print("Selected seeds S:", sorted(S))
    print(f"Total cost: {total_cost:.2f} / {budget:.2f}")
    print("f(S) =", fS)
    print("Covered nodes:", sorted(covered))

    # ------------------------------------------------------------
    # Visualization: highlight seeds in RED
    # ------------------------------------------------------------
    pos = nx.spring_layout(G, seed=7)

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.5)

    # Non-seed nodes
    non_seeds = [u for u in G.nodes() if u not in S]
    nx.draw_networkx_nodes(G, pos, nodelist=non_seeds, node_size=750, node_color="#8FB3D9")

    # Seed nodes (RED)
    nx.draw_networkx_nodes(G, pos, nodelist=list(S), node_size=900, node_color="red")

    # Labels show node id + cost; add * for seeds
    labels = {}
    for u in G.nodes():
        prefix = "*" if u in S else ""
        labels[u] = f"{prefix}{u}\nc={G.nodes[u].get('cost', 1.0)}"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(f"Your CELF picked seeds (red): S={sorted(S)} | cost={total_cost:.2f}/{budget} | f(S)={fS}")
    plt.axis("off")
    plt.show()
