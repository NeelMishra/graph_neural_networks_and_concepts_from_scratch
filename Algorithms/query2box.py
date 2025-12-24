import torch
import networkx as nx

def softplus(z):
    return torch.log1p(torch.exp(-torch.abs(z))) + torch.maximum(z, torch.zeros_like(z))

class Query2Box(torch.nn.Module):

    def __init__(self, G, emb_dim, alpha=0.2, gamma=12.0, default_relation="connected_to"):

        super().__init__()

        self.G = G
        self.emb_dim = emb_dim
        self.alpha = alpha
        self.gamma = gamma
        self.default_relation = default_relation

        self.entities = sorted(list(self.G.nodes()))
        
        self.idx_to_entity = {idx : entity for idx, entity in enumerate(self.entities)}
        self.entity_to_idx = {entity : idx for idx, entity in enumerate(self.entities)}

        self.relations = set()
        for u, v, attr in self.G.edges(data=True):
            r = attr.get("relation", self.default_relation)
            self.relations.add(r)
        self.relations = sorted(self.relations)

        self.idx_to_relation = {idx : relation for idx, relation in enumerate(self.relations)}
        self.relation_to_idx = {relation : idx for idx, relation in enumerate(self.relations)}

        self.m = len(self.entities) # m => entites
        self.n = len(self.relations) # n => relations

        self.entity_emb = torch.nn.Parameter(torch.rand(self.m, self.emb_dim))
        self.rel_cen = torch.nn.Parameter(torch.rand(self.n, self.emb_dim))
        self.rel_off_raw = torch.nn.Parameter(torch.rand(self.n, self.emb_dim))

    def rel_offset(self, relation_idx):
        return softplus(self.rel_off_raw[relation_idx])

    def anchor_box(self, entity_idx):

        center = self.entity_emb[entity_idx]
        offset = torch.zeros(self.emb_dim, device=center.device, dtype=center.dtype) # Entitys are rectangle with area 0

        return center, offset
    
    def project(self, box, relation_idx):
        center, offset = box
        new_center, new_offset = center + self.rel_cen[relation_idx], offset + self.rel_offset(relation_idx)

        return new_center, new_offset
    
    def dist_box_point(self, box, entity_idx):
        c, o = box
        v, _ = self.anchor_box(entity_idx)

        delta = torch.abs(v-c)
        out_vec = torch.maximum(delta-o, torch.zeros_like(delta))
        in_vec = torch.minimum(delta, o)

        d_box = torch.sum(out_vec) + self.alpha * torch.sum(in_vec)
        return d_box
    
    def score(self, box, entity_id):

        return self.gamma - self.dist_box_point(box, entity_id)
    
    def loss(self, box, pos_entity_idx, neg_entity_idx):
        
        term1 = softplus(-self.score(box, pos_entity_idx))#- np.log(sigmoid(self.score(box, pos_entity_idx)))
        term2 = softplus(self.score(box, neg_entity_idx))#- np.log(sigmoid(-self.score(box, neg_entity_idx)))

        return term1 + term2 
    
    def query_1hop(self, head_entity_idx, relation_idx):
        box = self.anchor_box(head_entity_idx)
        box = self.project(box, relation_idx)

        return box
    
    def path_query(self, head_entity_idx, relation_path):
        box = self.anchor_box(head_entity_idx)

        for relation_idx in relation_path:
            box = self.project(box, relation_idx)

        return box
    
    def intersect(self, boxes):
        c_intersect, o_intersect = boxes[0]
        l_intersect, u_intersect = c_intersect - o_intersect, c_intersect + o_intersect

        for box in boxes[1:]:
            c, o = box
            l, u = c-o, c+o
            
            l_intersect = torch.maximum(l_intersect, l)
            u_intersect = torch.minimum(u_intersect, u)

        c_intersect = (l_intersect + u_intersect) / 2
        o_intersect = torch.maximum(torch.zeros_like(u_intersect), (u_intersect - l_intersect)/2)

        return c_intersect, o_intersect
    
    def conj_query(self, head_entity_idx, relation_paths):

        path_boxes = []
        for path in relation_paths:
            path_boxes.append(self.path_query(head_entity_idx, path))
        
        return self.intersect(path_boxes)



#### LLM Generated driver code

import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Assumes Query2Box and softplus are already defined exactly as in your implementation.

def make_toy_kg():
    G = nx.DiGraph()

    people = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Heidi","Ivan","Judy"]
    cities = ["Paris","Berlin","Tokyo","Mumbai"]
    companies = ["Acme","Globex"]
    interests = ["RL","Graphs","Vision","NLP"]

    for n in people + cities + companies + interests:
        G.add_node(n)

    def add(u, v, r):
        G.add_edge(u, v, relation=r)

    # lives_in
    add("Alice","Paris","lives_in")
    add("Bob","Berlin","lives_in")
    add("Carol","Tokyo","lives_in")
    add("Dave","Mumbai","lives_in")
    add("Eve","Paris","lives_in")
    add("Frank","Berlin","lives_in")
    add("Grace","Tokyo","lives_in")
    add("Heidi","Mumbai","lives_in")
    add("Ivan","Paris","lives_in")
    add("Judy","Berlin","lives_in")

    # works_at
    for p in ["Alice","Carol","Eve","Grace","Ivan"]:
        add(p,"Acme","works_at")
    for p in ["Bob","Dave","Frank","Heidi","Judy"]:
        add(p,"Globex","works_at")

    # likes
    add("Alice","RL","likes")
    add("Alice","Graphs","likes")
    add("Bob","Vision","likes")
    add("Carol","NLP","likes")
    add("Dave","RL","likes")
    add("Eve","Graphs","likes")
    add("Frank","Vision","likes")
    add("Grace","NLP","likes")
    add("Heidi","RL","likes")
    add("Ivan","Graphs","likes")
    add("Judy","NLP","likes")

    # friend_of (bidirectional)
    friend_pairs = [("Alice","Bob"),("Alice","Eve"),("Bob","Frank"),("Carol","Grace"),("Dave","Heidi"),("Ivan","Judy")]
    for a,b in friend_pairs:
        add(a,b,"friend_of"); add(b,a,"friend_of")

    return G

def extract_triples(G, entity_to_idx, relation_to_idx, default_relation):
    triples = []
    for h, t, attr in G.edges(data=True):
        r = attr.get("relation", default_relation)
        triples.append((entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]))
    return triples

def build_true_tails(triples):
    true = {}
    for h, r, t in triples:
        true.setdefault((h, r), set()).add(t)
    return true

def sample_negative(num_entities, forbidden):
    while True:
        neg = random.randrange(num_entities)
        if neg not in forbidden:
            return neg

@torch.no_grad()
def eval_filtered_linkpred(model, triples, true_tails, ks=(1,3,10)):
    hits = {k: 0 for k in ks}
    mrr = 0.0

    for h, r, t in triples:
        box = model.query_1hop(h, r)
        scores = torch.stack([model.score(box, eid) for eid in range(model.m)])  # (m,)

        filt = true_tails.get((h, r), set())
        if len(filt) > 1:
            scores = scores.clone()
            for tt in filt:
                if tt != t:
                    scores[tt] = -1e9

        rank = int((scores > scores[t]).sum().item()) + 1
        mrr += 1.0 / rank
        for k in ks:
            hits[k] += 1 if rank <= k else 0

    n = len(triples)
    for k in ks:
        hits[k] /= n
    mrr /= n
    return hits, mrr

@torch.no_grad()
def plot_query_box_2d(model, box, title, annotate_topk=10):
    assert model.emb_dim == 2, "Use emb_dim=2 for box visualization."

    pts = model.entity_emb.detach().cpu()
    c, o = box
    c = c.detach().cpu()
    o = o.detach().cpu()
    lower = c - o
    upper = c + o

    plt.figure()
    plt.scatter(pts[:, 0], pts[:, 1], marker='o')

    rect = Rectangle(
        (float(lower[0]), float(lower[1])),
        float(upper[0] - lower[0]),
        float(upper[1] - lower[1]),
        fill=False
    )
    plt.gca().add_patch(rect)

    scores = torch.stack([model.score((c.to(model.entity_emb.device), o.to(model.entity_emb.device)), i).detach().cpu()
                          for i in range(model.m)])
    topk = torch.topk(scores, k=min(annotate_topk, model.m)).indices.tolist()
    for i in topk:
        x, y = float(pts[i, 0]), float(pts[i, 1])
        plt.text(x, y, f" {model.idx_to_entity[i]}", fontsize=9)

    plt.title(title)
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.axis("equal")
    plt.show()

def main():
    random.seed(0)
    torch.manual_seed(0)

    G = make_toy_kg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Query2Box(G, emb_dim=2, alpha=0.2, gamma=12.0).to(device)

    triples = extract_triples(G, model.entity_to_idx, model.relation_to_idx, model.default_relation)
    true_tails = build_true_tails(triples)

    h = model.entity_to_idx["Alice"]
    r = model.relation_to_idx["lives_in"]
    t = model.entity_to_idx["Paris"]

    model.eval()
    with torch.no_grad():
        box0 = model.query_1hop(h, r)
    plot_query_box_2d(model, box0, "Before training: (Alice, lives_in, ?)")

    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 200
    losses, hits1, mrrs = [], [], []

    for ep in range(epochs):
        model.train()
        random.shuffle(triples)

        total = 0.0
        for h_idx, r_idx, t_idx in triples:
            neg = sample_negative(model.m, true_tails[(h_idx, r_idx)])

            box = model.query_1hop(h_idx, r_idx)
            loss = model.loss(box, t_idx, neg)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item())

        losses.append(total / len(triples))

        model.eval()
        with torch.no_grad():
            hits, mrr = eval_filtered_linkpred(model, triples, true_tails, ks=(1,3,10))
        hits1.append(hits[1])
        mrrs.append(mrr)

    plt.figure()
    plt.plot(losses)
    plt.title("Training loss (avg per triple)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    plt.figure()
    plt.plot(hits1, label="Hits@1")
    plt.plot(mrrs, label="MRR")
    plt.title("Effectiveness over training")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend()
    plt.show()

    model.eval()
    with torch.no_grad():
        box1 = model.query_1hop(h, r)
    plot_query_box_2d(model, box1, "After training: (Alice, lives_in, ?)")

    likes = model.relation_to_idx["likes"]
    friend = model.relation_to_idx["friend_of"]
    with torch.no_grad():
        conj_box = model.conj_query(model.entity_to_idx["Alice"], [[likes], [friend, likes]])
    plot_query_box_2d(model, conj_box, "Conj query: likes AND (friend -> likes)")

if __name__ == "__main__":
    main()
