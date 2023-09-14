def bfs(self, start_idx: int, end_idx: int) -> int:
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