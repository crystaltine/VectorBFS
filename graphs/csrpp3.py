from math import ceil, sqrt
import torch
from time import perf_counter
from bitarray import bitarray

from stack_queue import Queue

class Vertex:
    """
    Nameless, has idx.
    Vertices - 8 bytes:
        - 4 bytes: segment id
        - 4 bytes: edge count
        
    Initialized with 0 degree and empty edge list.
    """
    def __init__(self, segment_idx, vertex_idx):
        self.loc = (segment_idx, vertex_idx)
        self.degree: int = 0
        self.neighbors: list[tuple[int, int]] = [] # list of segment_idx, vertex_idx
        self.edge_props: list[int] = [] # must correspond to neighbors

# Classless Vertex schema (for tensors):
# ((segment_idx, vertex_idx), degree, [neighbors], [edge_props]

class CSRPP3:
    def __init__(self, init_vertices: int = 0, directed = False, weighted = True, segment_len: int = 1024):
        """
        Creates a graph using the CSR++ format.
        """
        self.directed = directed
        self.weighted = weighted
        self.segment_len = segment_len
        self.segments = []
        if init_vertices > 0:
            for i in range(init_vertices):
                self.add_vertex()
        
        # The segments are stored in a list. Each segment is a list of vertices (tuple)
        # Vertices contain: (vertex_name, vertex_idx, segment_idx, edge[])
        # Edges contain: (dest_vertex_idx, dest_segment_idx, weight)
        
    def add_vertex(self) -> int:
        """
        Adds a vertex to the graph. Returns the absolute index of the added vertex.
        """
        last_segment = self.segments[-1] if len(self.segments) > 0 else None
        
        # If the last segment already has max vertices, create a new segment
        if last_segment is None or len(last_segment) >= self.segment_len:
            # Create a new segment with the vertex
            # v_idx 0 since its the first in the segment
            self.segments.append([Vertex(0, len(self.segments))])
        else: self.segments[-1].append(Vertex(len(last_segment), len(self.segments) - 1))
        
        return len(self.segments) * self.segment_len - 1
        
    def add_edge(self, start_idx: int, end_idx: int, weight: any = None):
        """
        Adds an edge between the two vertices. Must provide absolute indices.
        If graph is unweighted, weight is ignored.
        """
        start_seg_idx, start_v_idx = start_idx // self.segment_len, start_idx % self.segment_len
        end_seg_idx, end_v_idx = end_idx // self.segment_len, end_idx % self.segment_len
        
        self.segments[start_seg_idx][start_v_idx].loc = (start_seg_idx, start_v_idx)
        self.segments[start_seg_idx][start_v_idx].degree += 1
        
        self.segments[end_seg_idx][end_v_idx].loc = (end_seg_idx, end_v_idx)
        if not self.directed: self.segments[end_seg_idx][end_v_idx].degree += 1
        
        self.segments[start_seg_idx][start_v_idx].neighbors.append((end_seg_idx, end_v_idx))
        if self.weighted: self.segments[start_seg_idx][start_v_idx].edge_props.append(weight)
        
        # If the graph is undirected, add the reverse edge
        if not self.directed:
            self.segments[end_seg_idx][end_v_idx].neighbors.append((start_seg_idx, start_v_idx))
            if self.weighted: self.segments[end_seg_idx][end_v_idx].edge_props.append(weight)
            
    def get_neighbors(self, vertex_idx: int, output_abs_ids: bool = False) -> list[int] | list[tuple[int, int]]:
        """
        vertex_idx should be the absolute index of the vertex, equal to self.segment_len * segment_idx + v_idx
        if output_abs_ids is True, returns a list of absolute vertex indices
        if output_abs_ids is False, returns a list of (segment_idx, v_idx) tuples
        #### using false should be faster - then calc abs indicies separately when needed
        """
        
        v_id, seg_id = vertex_idx % self.segment_len, vertex_idx // self.segment_len
        
        if output_abs_ids:
            return [self.segment_len * neighbor[0] + neighbor[1] for neighbor in self.segments[seg_id][v_id].neighbors]
        return self.segments[seg_id][v_id].neighbors
    
    def get_edge_data(self, start_idx: int, end_idx: int) -> any:
        """
        Returns the data associated with the edge between the two vertices, given by absolute indices.
        None if no edge exists.
        """
        start_seg_idx, start_v_idx = start_idx // self.segment_len, start_idx % self.segment_len
        end_seg_idx, end_v_idx = end_idx // self.segment_len, end_idx % self.segment_len
        
        # Find the index of the edge in the start vertex
        try:
            index = self.segments[start_seg_idx][start_v_idx].neighbors.index((end_seg_idx, end_v_idx))
            return self.segments[start_seg_idx][start_v_idx].edge_props[index]
        except ValueError:
            return None
    
    def get_vertex_abs(self, abs_idx: int) -> Vertex:
        """Returns the vertex at the absolute index."""
        v_id, seg_id = abs_idx % self.segment_len, abs_idx // self.segment_len
        return self.segments[seg_id][v_id]
    
    def get_vertex_coord(self, vertex_loc: tuple[int, int]) -> Vertex:
        """Returns the vertex at the location given by (segment_idx, vertex_idx)"""
        return self.segments[vertex_loc[0]][vertex_loc[1]]
    
    def get_num_vertices(self) -> int:
        """Returns the number of vertices in the graph."""
        return self.segment_len * (len(self.segments) - 1) + len(self.segments[-1])
    
    def coord_to_abs(self, vertex_loc: tuple[int, int]) -> int:
        """Returns the absolute index of the vertex at the location given by (segment_idx, vertex_idx)"""
        return self.segment_len * vertex_loc[0] + vertex_loc[1]
    
    def bfs(self, start_idx: int, end_idx: int) -> int:
        """
        Returns the number of edges that must be traversed to get from start to end.
        """
        q = Queue()
        visited = set()
        q.enqueue((start_idx, 0))
        while not q.is_empty():
            node, depth = q.dequeue()
            
            # If we are at destination, the depth is stored inside the queue
            if node == end_idx: return depth
            
            # Otherwise add to visited
            visited.add(node)
            
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    q.enqueue((self.coord_to_abs(neighbor), depth + 1))
        return -1

class CSRPP3_Torch:
    def __init__(self, init_vertices = 64, max_vertices_per_node = 16, large=False, directed = False, weighted = True):
        """
        Creates a graph using the CSR++ format, using torch tensors.
        The graph is represented by a 3d tensor. 
        Each "drawer" (line of 1d cells) represents the neighbor list of a vertex. Vertex pos is given by the position
        of the cabinet's first cell on the main "face" of the 3D tensor.
        Edge/Neighbor data is stored in an int32 or int64 object, depending on the `large` parameter.
        First half of bits represent the location (index) of the neighbor vertex, second half represents the edge data or weight.
        """
        self.directed = directed
        self.weighted = weighted
        self.segment_len = ceil(sqrt(init_vertices))
        self.size = 0
        self.large = 1 if not large else 2 # For efficient bit shifting
        self.segments: torch.Tensor = torch.zeros(
            (self.segment_len, self.segment_len, max_vertices_per_node), 
            dtype=(torch.int32 if not large else torch.int64)
        )
        
    def add_edge(self, start_idx: int, end_idx: int, weight = 0):
        """
        Adds an edge between the two vertices. Must provide absolute indices.
        If graph is unweighted, weight is ignored.
        Weight must be a positive value
        """
        start_seg_idx, start_v_idx = start_idx // self.segment_len, start_idx % self.segment_len
        
        # Add 1 to the degree of the start vertex
        self.segments[start_seg_idx][start_v_idx][0] += 1
        
        # Add the neighbor to the start vertex, encoded as an int
        self.segments[start_seg_idx][start_v_idx][self.segments[start_seg_idx][start_v_idx][0]] = \
            (end_idx << 16 * self.large) + (weight * self.weighted)
        
        # If the graph is undirected, add the reverse edge
        if not self.directed:
            end_seg_idx, end_v_idx = end_idx // self.segment_len, end_idx % self.segment_len
            self.segments[end_seg_idx][end_v_idx][0] += 1
            self.segments[end_seg_idx][end_v_idx][self.segments[end_seg_idx][end_v_idx][0]] = \
                (start_idx << 16 * self.large) + (weight * self.weighted)
            
    def get_edge_data(self, vertex_idx: int) -> torch.Tensor:
        """
        Returns a list of the neighbors and edge weights of the vertex, given by absolute index.
        Encoded data: first half bits represent the location (index) of the neighbor vertex, second half represents the edge data or weight.
        """
        
        return self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len] \
            [1:self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len][0]+1]
            
    def get_edge_data_raw(self, vertex_idx: int) -> torch.Tensor:
        """Returns the whole 'cabinet'"""
        return self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len]
    
    def bfs(self, start_idx: int, end_idx: int) -> int:
        """
        Returns the number of edges that must be traversed to get from start to end.
        """
        
        q = Queue()
        visited = set()
        q.enqueue((start_idx, 0))
        while not q.is_empty():
            node, depth = q.dequeue()
            
            # If we are at destination, the depth is stored inside the queue
            if node == end_idx: return depth
            
            # Otherwise add to visited
            visited.add(node)

            for neighbor_data in self.get_edge_data(node):
                neighbor_idx = neighbor_data.item() >> (16 * self.large)
                
                if neighbor_idx not in visited:
                    q.enqueue((neighbor_idx, depth + 1))
        return -1

    def bfs2(self, start_idx: int, end_idx: int) -> int:
        """
        Returns the number of edges that must be traversed to get from start to end.
        """
        
        q = Queue()
        visited = set()
        q.enqueue((start_idx, 0))
        while not q.is_empty():
            node, depth = q.dequeue()
            
            # If we are at destination, the depth is stored inside the queue
            if node == end_idx: return depth
            
            # Otherwise add to visited
            visited.add(node)

            raw_edge_data = self.get_edge_data_raw(node)
            
            for i in range(self.segments[node // self.segment_len][node % self.segment_len][0]):
                neighbor = raw_edge_data[i + 1].item() >> (16 * self.large)
                if neighbor not in visited:
                    q.enqueue((neighbor, depth + 1))
        return -1
    
class CSRPP3_Torch_2:
    def __init__(self, init_vertices = 64, max_vertices_per_node = 16, directed = False, weighted = True):
        """
        Creates a graph using the CSR++ format, using torch tensors.
        The graph is represented by a 4d tensor. 
        Each "drawer" (two lines of 1d cells) represents the neighbor list of a vertex. Vertex pos is given by the position
        of the cabinet's first cell on the main "face" of the 3D tensor.
        Edge/Neighbor data is stored in two ints which are contained in the two lines
        Less mem efficient, but faster than encoding data in a single int
        """
        self.directed = directed
        self.weighted = weighted
        self.segment_len = ceil(sqrt(init_vertices))
        self.size = 0
        self.segments: torch.Tensor = torch.zeros(
            (self.segment_len, self.segment_len, 2, max_vertices_per_node), 
            dtype=torch.int32
        )
        
    def add_edge(self, start_idx: int, end_idx: int, weight = 0):
        """
        Adds an edge between the two vertices. Must provide absolute indices.
        If graph is unweighted, weight is ignored.
        Weight must be a positive value
        """
        start_seg_idx, start_v_idx = start_idx // self.segment_len, start_idx % self.segment_len
        
        # Add 1 to the degree of the start vertex
        self.segments[start_seg_idx][start_v_idx][0][0] += 1
        degree = self.segments[start_seg_idx][start_v_idx][0][0]
        
        # Add the neighbor to the start vertex, encoded as an int
        self.segments[start_seg_idx][start_v_idx][0][degree] = end_idx
        self.segments[start_seg_idx][start_v_idx][1][degree] = weight * self.weighted
        
        # If the graph is undirected, add the reverse edge
        if not self.directed:
            end_seg_idx, end_v_idx = end_idx // self.segment_len, end_idx % self.segment_len
            self.segments[end_seg_idx][end_v_idx][0][0] += 1
            degree = self.segments[end_seg_idx][end_v_idx][0][0]
            
            self.segments[end_seg_idx][end_v_idx][0][degree] = start_idx
            self.segments[end_seg_idx][end_v_idx][1][degree] = weight * self.weighted
            
    def get_neighbors(self, vertex_idx: int) -> torch.Tensor:
        """
        vertex_idx should be the absolute index of the vertex, equal to self.segment_len * segment_idx + v_idx
        returns a tensor of absolute vertex indices
        """
        
        v_id, seg_id = int(vertex_idx % self.segment_len), int(vertex_idx // self.segment_len)
        
        return self.segments[seg_id][v_id][0][1:self.segments[seg_id][v_id][0][0]+1]
    
    def get_neighbors_raw(self, vertex_idx: int) -> torch.Tensor:
        """Returns the whole 'cabinet', only first line (neighbor indices)"""
        return self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len][0]          
            
    def get_edge_data(self, vertex_idx: int) -> torch.Tensor:
        """
        Returns a list of the neighbors and edge weights of the vertex, given by absolute index.
        Encoded data: first half bits represent the location (index) of the neighbor vertex, second half represents the edge data or weight.
        """
        
        return self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len] \
            [1:self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len][0]+1]
            
    def get_edge_data_raw(self, vertex_idx: int) -> torch.Tensor:
        """Returns the whole 'cabinet'"""
        return self.segments[vertex_idx // self.segment_len][vertex_idx % self.segment_len]

    def bfs(self, start_idx: int, end_idx: int) -> int:
        """
        Returns the number of edges that must be traversed to get from start to end.
        """
        
        q = Queue()
        visited = set()
        q.enqueue((start_idx, 0))
        while not q.is_empty():
            node, depth = q.dequeue()
            
            # If we are at destination, the depth is stored inside the queue
            if node == end_idx: return depth
            
            # Otherwise add to visited
            visited.add(node)

            raw_neighbor_data = self.get_neighbors_raw(node)
            
            # neighbors_not_in_visited = torch.tensor([neighbor for neighbor in raw_neighbor_data if neighbor not in visited])
            
            for i in range(raw_neighbor_data[0].item()):
                neighbor = raw_neighbor_data[i + 1].item()
                if neighbor not in visited:
                    q.enqueue((neighbor, depth + 1))
                    

        return -1
    
    def bfs_parallel_top_down(self, start_idx: int, end_idx: int) -> int:
        v_in = torch.tensor([start_idx])
        v_out = torch.tensor([])
        visited = bitarray(self.segment_len * self.segment_len)
        visited.setall(0)
        visited[start_idx] = 1
        
        #parents = torch.empty(self.segment_len * self.segment_len, dtype=torch.int32)
        
        depth = 0
        
        while len(v_in) > 0:
            v_out = torch.tensor([])
            for vertex in v_in:
                vertex = vertex.item()
                for neighbor in self.get_neighbors(vertex):
                    neighbor = neighbor.item()
                    if not visited[neighbor]:
                        #parents[neighbor] = vertex
                        if neighbor == end_idx:
                            return depth + 1
                        v_out = torch.cat((v_out, torch.tensor([neighbor])))
                        visited[neighbor] = 1
            v_in = v_out
            depth += 1
        return -1
            
class TorchGraph3:
    """
    Optimized for BFS - unweighted, always directed
    Implemented with an adjacency-list-esque structure
    """
    def __init__(self, init_vertices = 64, max_vertices_per_node = 256):
        self.degrees = torch.zeros(init_vertices, dtype=torch.int32)
        self.vertices = torch.full((init_vertices, max_vertices_per_node), -1, dtype=torch.int32)
        self.size = init_vertices
        
    def add_vertex(self) -> int:
        """
        Adds a vertex to the graph. Returns the absolute index of the added vertex.
        """
        if self.size >= len(self.degrees):
            self.degrees = torch.cat((self.degrees, torch.zeros(self.size, dtype=torch.int32)))
            self.vertices = torch.cat((self.vertices, torch.full((self.size, self.vertices.shape[1]), -1, dtype=torch.int32)))
        
        self.size += 1
        return self.size - 1
        
    def add_edge(self, start_idx: int, end_idx: int):
        """
        Adds an edge between the two vertices. Must provide absolute indices.
        If graph is unweighted, weight is ignored.
        Weight must be a positive value
        """
        self.degrees[start_idx] += 1
        self.vertices[start_idx][self.degrees[start_idx]-1] = end_idx
    
    def get_neighbors(self, vertex_idx: int) -> torch.Tensor:
        """
        vertex_idx should be the absolute index of the vertex, equal to self.segment_len * segment_idx + v_idx
        returns a tensor of absolute vertex indices
        """
        
        return self.vertices[vertex_idx][1:self.vertices[vertex_idx][0]+1]
    
    def bfs2(self, start_idx: int, end_idx: int) -> int:
        curr = torch.tensor([start_idx])
        visited = bitarray(len(self.degrees))
        
        curr_depth = 0
        
        def not_in_visited(vertex):
            return 0 if visited[vertex] else vertex
        
        while len(curr) > 0:
            flattened = self.vertices[curr].flatten()
            next = flattened[flattened != -1]
            next = next.apply_(not_in_visited)[next != 0] # List of neighbors not in visited
            curr_depth += 1
            
            if end_idx in next: return curr_depth
            
            curr = next
            print("next is ", next)
            # Set all visited vertices to 1
            for vertex in curr:
                visited[vertex] = 1
        
        return -1
    
    # 1944 -> 585 failing
    def bfs(self, start_idx: int, end_idx: int) -> int:        
        curr = torch.tensor([start_idx])

        visited = bitarray(len(self.degrees))
        visited.setall(0)

        curr_depth = 0
        
        def not_in_visited(vertex):
            #return 0 if visited[vertex] else vertex
            return vertex * (1 - visited[vertex])
        
        while len(curr) > 0:
            flattened = self.vertices[curr]
            next = flattened[flattened != -1]
            next = next.apply_(not_in_visited)[next != 0] # List of neighbors not in visited
            curr_depth += 1
            
            curr = next

            # Set all visited vertices to 1
            #for vertex in curr:
            #    visited[vertex] = 1
            visited[curr.tolist()] = 1
            if visited[end_idx]: return curr_depth
        
        return -1

class TorchGraph4:
    """
    Another version which is optimized for BFS - unweighted, default directed
    Implemented this time with tensor of bool tensors representing connections (like adjacency matrix)
    """
    def __init__(self, init_vertices = 64, directed = True):
        #adj matrix
        self.vertices = torch.zeros((init_vertices, init_vertices), dtype=torch.bool)
        self.directed = directed
    
    def add_edge(self, start_idx: int, end_idx: int):
        """
        Adds an edge between the two vertices. Must provide absolute indices.
        If graph is unweighted, weight is ignored.
        Weight must be a positive value
        """
        self.vertices[start_idx][end_idx] = 1
        # ignore undirected for now
        
    def bfs(self, start_idx: int, end_idx: int):
        
        visited = torch.zeros(len(self.vertices), dtype=torch.bool)
        visited[start_idx] = 1
        
        curr = visited.clone()
        curr_depth = 0
        
        #total_curr_update_time = [0, 0, 0]
        
        while True: # assume connected graph           
            curr_depth += 1
            
            #start_time = perf_counter()
            curr = self.vertices[curr]
            #total_curr_update_time[0] += perf_counter() - start_time
            #start_time = perf_counter()
            curr = curr.any(dim=0)
            #total_curr_update_time[1] += perf_counter() - start_time
            #start_time = perf_counter()
            curr = curr & ~visited
            #total_curr_update_time[2] += perf_counter() - start_time
            
            # if end_idx is in curr, return curr_depth
            if curr[end_idx]: 
                #print(f"\ntimes: {[f'{1000*t:.2f}ms' for t in total_curr_update_time]}")
                return curr_depth
            
            # Set all visited vertices to 1
            
            visited[curr] = 1
            
    
    def bfs4(self, start_idx: int, end_idx: int):
        
        visited = torch.zeros(len(self.vertices), dtype=torch.bool)
        
        curr = visited.clone()
        curr[start_idx] = 1
        
        curr_depth = 0
        
        while curr.any(): # assume connected graph
            
            curr_depth += 1
            # filter out visited vertices
            curr = torch.any(self.vertices[~visited][curr], dim=0)
            
            # return if end_idx is in curr
            if curr[end_idx]: return curr_depth
            
            # Set all visited vertices to 1
            visited[curr] = 1
        return -1
    
    def bfs_bitarray(self, start_idx: int, end_idx: int):
        
        visited = bitarray(len(self.vertices))
        visited.setall(0)
        visited[start_idx] = 1
        
        curr = visited.copy()
        curr_depth = 0
        
        while curr.any():
            if visited[end_idx]: return curr_depth
            
            curr_depth += 1
            
            # XXX - issue with indexing the list
            curr = self.vertices[curr] & ~visited
            
            visited[curr] = 1        

    def bfs2(self, start_idx: int, end_idx: int):
        
        #start_time = perf_counter()
        vertices_copy = self.vertices.clone()

        curr = torch.zeros(len(self.vertices), dtype=torch.bool)
        curr[start_idx] = 1
        
        curr_depth = 0
        #print("Torch Graph 4 BFS Setup Time: ", 1000*(perf_counter() - start_time), "ms")
        # ^ about 3-5ms
        
        while True: # assume connected graph
            if (curr == 1)[end_idx]: return curr_depth
            
            curr_depth += 1
            
            vertices_copy[curr] = 0
            vertices_copy[:, curr] = 0
            
            curr = torch.any(self.vertices[curr], dim=0)
    
    def bfs3(self, start_idx: int, end_idx: int):
        visited = bitarray(len(self.vertices))
        visited[start_idx] = 1
        
        curr = bitarray(len(self.vertices))
        curr[start_idx] = 1
        
        curr_depth = 0
        
        while True: # assume connected graph
            # print(curr.count())
            if visited[end_idx]: return curr_depth
            
            curr_depth += 1
            
            # print("Stuff is " + str(torch.any(self.vertices[curr], dim=0)))
            curr = bitarray(torch.any(self.vertices[curr], dim=0).tolist()) & ~visited
            
            # Set all visited vertices to 1
            visited |= curr
            
class BitArrayGraph:
    """
    Another version using bitarray - unweighted, default directed
    Implemented this time with adj matrix of bitarrays representing connections (0 or 1)
    """
    def __init__(self, init_vertices = 64, directed = True):
        #adj matrix

        self.vertices = []
        for _ in range(init_vertices):
            self.vertices.append(bitarray(init_vertices))
            self.vertices[-1].setall(0)
    
        self.directed = directed
    
    def add_edge(self, start_idx: int, end_idx: int):
        """
        Adds an edge between the two vertices. Must provide absolute indices.
        If graph is unweighted, weight is ignored. (for now, always)
        """
        self.vertices[start_idx][end_idx] = 1
        # ignore undirected for now
        
    def bfs(self, start_idx: int, end_idx: int):
        
        visited = bitarray(len(self.vertices))
        visited.setall(0)
        visited[start_idx] = 1
        
        curr = visited.copy()
        curr_depth = 0
        
        while curr.any():
            if visited[end_idx]: return curr_depth
            
            curr_depth += 1
            
            # XXX - issue with indexing the list
            curr = self.vertices[curr] & ~visited
            
            visited[curr] = 1
            
    def bfs_bad(self, start_idx: int, end_idx: int):
        
        visited = bitarray(len(self.vertices))
        visited.setall(0)
        visited[start_idx] = 1
        
        curr = visited.copy()
        curr_depth = 0
        
        while curr.any():
            if visited[end_idx]: return curr_depth
            
            curr_depth += 1
            
            # XXX - issue with indexing the list
            # Temp fix: use torch.any
            curr = bitarray(torch.any(torch.tensor(self.vertices)[curr], dim=0).tolist()) & ~visited
            
            visited |= curr
