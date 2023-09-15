from typing import Set
from edge import Edge
from map import Map

class AdjacencyMatrixGraph():
    # if is_directed is true, this should be a directed graph.  If false, it's an undirected graph
    # Make sure your implementation accounts for node order on an edge for directed and is neutral for undirected
    # super sets a property self.is_directed to the parameter value
    def __init__(self, is_directed:bool) -> None:
        self.adjacency_matrix: list[list[int]] = []
        self.nodes = Map(str, int) # name -> index
        self.is_directed = is_directed
    
    # Removes the node named "name" and all arcs connected to it
    def add_node(self, name:str) -> None:
        self.nodes.add_pair(name, len(self.adjacency_matrix))
        for row in self.adjacency_matrix: row.append(None)
        self.adjacency_matrix.append([None] * (len(self.adjacency_matrix) + 1))

    # Gets a set of all nodes in the graph
    def remove_node(self, name:any) -> None:
        
        shift_index = self.nodes[name]

        for row in self.adjacency_matrix:
            row[shift_index] = row[-1] # Shift last column to the deleted column
            row.pop() # Delete last column since it has been shifted forward
        # delete the last row since it has been shifted forward
        self.adjacency_matrix[shift_index] = self.adjacency_matrix[-1] # Shift last row to the deleted row
        self.adjacency_matrix.pop()
        
        self.nodes.remove_pair(name, shift_index)
        
        # update the index for the node that was shifted to the deleted node's spot
        shifted_node = self.nodes[len(self.adjacency_matrix) - 1]
        self.nodes.__setitem__(shifted_node, shift_index)

    # Returns True if the node name1 is connected to the node name2 and False otherwise
    def is_connected(self, name1:any, name2:any)->None:
        
        index_1 = self.nodes[name1]
        index_2 = self.nodes[name2]
        
        return (
            self.adjacency_matrix[index_1][index_2] is not None if self.is_directed 
            else (
            self.adjacency_matrix[index_1][index_2] is not None or
            self.adjacency_matrix[index_2][index_1] is not None
            ))
    
    # Adds an edge from node start to node finish with the weight specified
    def add_edge(self, start:any, finish:any, weight:int)->None:
        self.adjacency_matrix[self.nodes[start]][self.nodes[finish]] = weight

    # Returns a set of the names of nodes adjacent to the given node (i.e. there's an arc from the node to the neighbor)
    def get_neighbors(self, name:str)->Set[any]:
        neighbors = set()
        node_index = self.nodes[name]
        
        for i in range(len(self.adjacency_matrix)):
            if self.adjacency_matrix[node_index][i] is not None:
                neighbors.add(self.nodes[i])
            if self.is_directed and self.adjacency_matrix[i][node_index] is not None:
                neighbors.add(self.nodes[i])
        return neighbors
    
    # Gets a set of edges leading out of the given node
    def get_edges(self, name:str)->Set[Edge]:
        edges = set()
        node_index = self.nodes[name]
        
        for i in range(len(self.adjacency_matrix)):
            if self.adjacency_matrix[node_index][i] is not None:
                start_name = name
                end_name = self.nodes[i]
                edges.add(Edge(start=start_name, finish=end_name, weight=self.adjacency_matrix[node_index][i]))
            
            if self.is_directed and self.adjacency_matrix[i][node_index] is not None:
                start_name = self.nodes[i]
                end_name = name
                edges.add(Edge(start=start_name, finish=end_name, weight=self.adjacency_matrix[i][node_index]))
        return edges
    
    # Gets a set of the names of all nodes in the graph
    def get_nodes(self):
        return self.nodes.keys()