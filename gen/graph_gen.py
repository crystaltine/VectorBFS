import random

def create_graph(
    num_vertices: int,
    num_edges: int,
    out_filename: str,
    weighted: bool = True,
    weight_range: tuple[int, int] = (1, 100),
    connected: bool = True,
    directed: bool = False,
    ) -> None:
    """
    Randomly generates a graph with the given parameters.
    Writes to file in the format:
    
    ```
    V E
    start end weight
    ...
    ```
    
    Node names and edge weights are integers.
    """
    
    if connected and num_edges < num_vertices - 1:
        raise ValueError("Cannot create a connected graph with fewer edges than vertices - 1")
    
    # Assume always connected, weighted, and undirected for now
    
    graph_data = [[] for _ in range(num_vertices)]
    
    # if connected, each node has to have at least one edge
    with open(f"./data/generated/{out_filename}", 'w') as f:
        f.write(f"{num_vertices} {num_edges}\n")
        for i in range(num_edges):
            start = i % num_vertices
            
            finish = random.randint(0, num_vertices - 1)
            if finish == i % num_vertices: # don't connect to self
                finish = (finish + 1) % num_vertices
                
            weight = random.randint(*weight_range) if weighted else None
            graph_data[start].append((finish, weight))
            
        for start_node in range(len(graph_data)):
            for finish_node, weight in graph_data[start_node]:
                f.write(f"{start_node} {finish_node} {weight}\n")

create_graph(5000, 12000, "test.txt")    