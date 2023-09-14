class CSRPPGraph:
    def __init__(self, num_nodes, directed=False, weighted=False):
        self.num_nodes = num_nodes
        self.directed = directed
        self.weighted = weighted
        
        self.row_ptr = [0] * (num_nodes + 1)
        self.col_idx = []
        self.weights = [] if weighted else None
    
    def add_edge(self, source, target, weight=None):
        if source < 0 or source >= self.num_nodes or target < 0 or target >= self.num_nodes:
            raise ValueError("Node index out of range")

        if self.weighted and weight is None:
            raise ValueError("Weight not provided for a weighted graph")
        
        self.col_idx.append(target)
        if self.weighted:
            self.weights.append(weight)
        
        if self.directed:
            self.row_ptr[source + 1] += 1
        else:
            self.row_ptr[source + 1] += 1
            self.row_ptr[target + 1] += 1
    
    def build(self):
        for i in range(1, self.num_nodes + 1):
            self.row_ptr[i] += self.row_ptr[i - 1]
        
        if not self.directed:
            self.symmetric_reorder()
    
    def symmetric_reorder(self):
        new_col_idx = [-1] * len(self.col_idx)
        new_weights = [] if self.weighted else None
        new_row_ptr = [0] * (self.num_nodes + 1)
        
        for node in range(self.num_nodes):
            start = self.row_ptr[node]
            end = self.row_ptr[node + 1]
            for idx in range(start, end):
                neighbor = self.col_idx[idx]
                if new_col_idx[neighbor] == -1:
                    new_col_idx[neighbor] = new_row_ptr[neighbor]
                    new_row_ptr[neighbor + 1] += 1
                    if self.weighted:
                        new_weights.append(self.weights[idx])
        
        for i in range(1, self.num_nodes + 1):
            new_row_ptr[i] += new_row_ptr[i - 1]
        
        self.col_idx = new_col_idx
        self.row_ptr = new_row_ptr
        self.weights = new_weights
    
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


# Example usage
num_nodes = 5
graph = CSRPPGraph(num_nodes, directed=False, weighted=True)
graph.add_edge(0, 1, 2)
graph.add_edge(0, 2, 3)
graph.add_edge(1, 2, 1)
graph.add_edge(2, 3, 4)
graph.add_edge(3, 4, 5)
graph.build()

print("Neighbors of node 0:", graph.get_neighbors(0))
print("Neighbors of node 1:", graph.get_neighbors(1))
print("Neighbors of node 2:", graph.get_neighbors(2))
print("Neighbors of node 3:", graph.get_neighbors(3))
print("Neighbors of node 4:", graph.get_neighbors(4))
