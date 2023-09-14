from stack_queue import Queue

def bfs(
    graph: any, 
    starting_node: any, 
    dest_node: any,
    get_neighbors: callable
    ) -> int:
    """
    `get_neighbors` should be a wrapper function that takes the graph and a node id/name and returns a list of neighbors.
    Returns the number of edges that must be traveled to reach dest_node from starting_node. -1 if no path.
    """
    q = Queue()
    visited = set()
    q.enqueue((starting_node, 0))
    while not q.is_empty():
        node, depth = q.dequeue()
        
        # If we are at destination, the depth is stored inside the queue
        if node == dest_node: return depth
        
        # Otherwise add to visited
        visited.add(node)
        
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                q.enqueue((neighbor, depth + 1))
    return -1