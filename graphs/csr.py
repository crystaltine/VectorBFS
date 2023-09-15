import torch

class CSRGraph:
    def __init__(self, num_nodes, directed=False, weighted=False):
        self.num_nodes = num_nodes
        self.directed = directed
        self.weighted = weighted
        
        self.row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
        self.col_idx = []
        self.weights = [] if weighted else None
    
    def add_edge(self, source, target, weight=None):        
        self.col_idx.append(target)
        if self.weighted:
            self.weights.append(weight)
        
        if self.directed:
            self.row_ptr[source + 1] += 1
        else:
            self.row_ptr[source + 1] += 1
            self.row_ptr[target + 1] += 1
    
    def build(self):
        # for i in range(1, self.num_nodes + 1):
        #     self.row_ptr[i] += self.row_ptr[i - 1]
        self.row_ptr = torch.cumsum(self.row_ptr, dim=0)
        
    def get_neighbors(self, node):
        if node < 0 or node >= self.num_nodes:
            raise ValueError("Node index out of range")
        
        start = self.row_ptr[node]
        end = self.row_ptr[node + 1]
        
        neighbors = []
        for idx in range(start, end):
            neighbor = self.col_idx[idx]
            weight = self.weights[idx] if self.weighted else None
            neighbors.append((neighbor, weight))
        
        return neighbors
    
    def get_edge(self, source, target) -> any:
        start = self.row_ptr[source]
        end = self.row_ptr[source + 1]
        
        for idx in range(start, end):
            if self.col_idx[idx] == target:
                return self.weights[idx] if self.weighted else 1
        
        return None