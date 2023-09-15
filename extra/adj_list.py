from typing import Set
from edge import Edge
from GBC import GraphBaseClass

class AdjacencyListGraph(GraphBaseClass):
    def __init__(self, is_directed:bool=True) -> None:
        self.nodes: dict[str, set[Edge]] = dict() # dict matching name : set of connections
        super().__init__(is_directed)
    
    def add_node(self, name:any) -> None:
        self.nodes.setdefault(name, set())

    def remove_node(self, name:any) -> None:
        self.nodes.pop(name)
        
        for node in self.nodes:
            deleted_edges = set()
            for edge in self.nodes[node]:
                if edge.finish == name:
                    deleted_edges.add(edge)
            self.nodes[node].difference_update(deleted_edges)
    
    # Gets a set of the names of all nodes in the graph
    def get_nodes(self) -> Set[str]:
        return set(self.nodes.keys())

    # Returns True if the node name1 is connected to the node name2 and False otherwise
    def is_connected(self, name1:any, name2:any) -> None:
        if not self.is_directed:
            return Edge(name1, name2, None) in self.nodes[name1] or Edge(name2, name1, None) in self.nodes[name2]
        return Edge(name1, name2, None) in self.nodes[name1]

    # Adds an edge from node start to node finish with the weight specified
    def add_edge(self, start:any, finish:any, weight:int) -> None:
        self.nodes[start].add(Edge(start, finish, weight))
        if not self.is_directed:
            self.nodes[finish].add(Edge(finish, start, weight))

    # Returns a set of the names of nodes adjacent to the given node (i.e. there's an arc from the node to the neighbor)
    def get_neighbors(self, name:str) -> Set[Edge]:
        return self.nodes[name]    
    
    # Gets a set of edges leading out of the given node
    def get_edges(self, name:str) -> Set[Edge]:
        all_edges = set()
        for edge_set in self.nodes.values():
            all_edges.update(edge_set)
        return all_edges