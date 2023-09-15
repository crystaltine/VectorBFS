"""
Implementation of the CSR++ representation for graphs.
"""

from GBC import GraphBaseClass

class CSRPPGraph(GraphBaseClass):
    def __init__(self, is_directed: bool = True) -> None:
        self.nodes = []
        self.edges = []
        self.starts = []
        self.lengths = []
        self.weights = []
        super().__init__(is_directed)
        
    def add_node(self, name: any) -> None:
        self.nodes.append(name)
        self.starts.append(len(self.edges))
        self.lengths.append(0)
        
    def remove_node(self, name: any) -> None:
        index = self.nodes.index(name)
        del self.nodes[index]
        del self.starts[index]
        del self.lengths[index]
        
        for i in range(index, len(self.starts)):
            self.starts[i] -= 1
        
        for i in range(self.starts[index], self.starts[index] + self.lengths[index]):
            del self.edges[i]
            del self.weights[i]
        
        del self.lengths[index]
        
    # Gets a set of the names of all nodes in the graph
    def get_nodes(self) -> set:
        return set(self.nodes)
    
    # Returns True if the node name1 is connected to the node name2 and False otherwise
    def is_connected(self, name1: any, name2: any) -> bool:
        if not self.is_directed:
            return (name1, name2) in self.edges or (name2, name1) in self.edges
        return (name1, name2) in self.edges
    
    # Adds an edge from node start to node finish with the weight specified
    def add_edge(self, start: any, finish: any, weight: int) -> None:
        self.edges.append((start, finish))
        self.weights.append(weight)
        self.lengths[self.nodes.index(start)] += 1
        if not self.is_directed:
            self.edges.append((finish, start))
            self.weights.append(weight)
            self.lengths[self.nodes.index(finish)] += 1
            
    # Returns a set of the names of nodes adjacent to the given node (i.e. there's an arc from the node to the neighbor)
    def get_neighbors(self, name: str) -> set:
        neighbors = set()
        index = self.nodes.index(name)
        for i in range(self.starts[index], self.starts[index] + self.lengths[index]):
            neighbors.add(self.edges[i][1])
        return neighbors
    
    # Gets a set of edges leading out of the given node
    def get_edges(self, name: str) -> set:
        edges = set()
        index = self.nodes.index(name)
        for i in range(self.starts[index], self.starts[index] + self.lengths[index]):
            edges.add(self.edges[i])
        return edges
    
    # Gets the weight of the edge from start to finish
    def get_weight(self, start: any, finish: any) -> int:
        index = self.nodes.index(start)
        for i in range(self.starts[index], self.starts[index] + self.lengths[index]):
            if self.edges[i][1] == finish:
                return self.weights[i]
        return None
    
    def remove_edge(self, start: any, finish: any) -> None:
        pass