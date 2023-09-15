from abc import ABC, abstractmethod
from typing import Set, List
from edge import Edge

class GraphBaseClass(ABC):
    def __init__(self, is_directed: bool) -> None:
        self.is_directed = is_directed
        super().__init__()

    @abstractmethod
    def add_node(self, name:any) -> None:
        pass

    @abstractmethod
    def remove_node(self, name:any) -> None:
        pass

    @abstractmethod
    # return True if the node name1 is connected to the node name2 and False otherwise
    def is_connected(self, name1:any, name2:any) -> bool:
        pass

    @abstractmethod
    def add_edge(self, start:any, finish:any, weight:int) -> None:
        pass

    @abstractmethod
    # Returns a set of node names adjacent to the given node (i.e. there's an arc from the node to the neighbor)
    def get_neighbors(self, name:any) -> Set[any]:
        pass
    
    @abstractmethod
    # Gets a set of arcs leading out of the given node
    def get_edges(self, name:any) -> Set[Edge]:
        pass
    
    @abstractmethod
    # Gets a set of arcs leading out of the given node
    def get_nodes(self) -> Set[str]:
        pass
    
    def get_edges_sorted(self, name: str) -> List[Edge]:
        edges = list(self.get_edges(name))
        edges.sort(key=lambda edge: edge.weight)
        return edges
    
    def get_neighbors_sorted(self, name: str) -> List[Edge]:
        edges = list(self.get_edges(name))
        edges.sort(key=lambda edge: edge.weight)
        return [edge.finish for edge in edges]