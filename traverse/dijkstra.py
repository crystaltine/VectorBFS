from prio_queue import PriorityQueue
from GBC import GraphBaseClass

def dijkstra(graph: GraphBaseClass, start_node: str, end_node: str) -> int:
    """
    Returns the shortest path from start_node to end_node. Graph must be weighted with non-negative weights.
    -1 if no path.
    """
    pq = PriorityQueue()
    visited = set()
    pq.enqueue((start_node, 0))
    while not pq.is_empty():
        node, cost = pq.dequeue()
        
        # If we are at destination, the cost is stored inside the queue
        if node == end_node: return cost
        
        # Otherwise add to visited
        visited.add(node)
        
        for edge in graph.get_edges(node):
            if edge.finish not in visited:
                pq.enqueue((edge.finish, cost + edge.weight))
    return -1