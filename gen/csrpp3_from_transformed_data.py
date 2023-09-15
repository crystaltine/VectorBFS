from graphs.csrpp3 import CSRPP3, CSRPP3_Torch, CSRPP3_Torch_2, TorchGraph3, TorchGraph4
from time import perf_counter

def get_csrpp3_from_transformed_data(filename: str) -> CSRPP3:
    """
    IMPORTANT! filename should be the name of a text file in the **./data folder** (NO EXTENSION).
    
    Using data of the format of data/graph4096_transformed.txt:
    1. N E
    2-N+1. start_idx end_idx weight
    
    generates a CSRPP3 Graph object.
    """
    lines = open(f"./data/{filename}.txt", "r").readlines()
    N, E = int(lines[0].split()[0].strip()), int(lines[0].split()[1].strip())
    
    #start_time = perf_counter()
    graph = CSRPP3(init_vertices=N, directed=True, weighted=True, segment_len=256)
    #print("\x1b[2mInitialized CSRPP3 Graph (%.4fs)\x1b[0m" % (perf_counter() - start_time))
    
    #total_time = 0
    
    for i in range(1, E+1):
        line = lines[i].split()
        start_idx, end_idx, weight = int(line[0].strip()), int(line[1].strip()), int(line[2].strip())
        # print("\x1b[34m[Graph Gen] Adding edge from %d to %d with weight %d\x1b[0m" % (start_idx, end_idx, weight))
        #start_time = perf_counter()
        graph.add_edge(start_idx, end_idx, weight)
        #total_time += perf_counter() - start_time
    
    #print(f"Total time for regular add edge: {total_time:.4f}s")
        
    return graph

def get_csrpp3_torch_from_transformed_data(filename: str) -> CSRPP3_Torch:
    """
    IMPORTANT! filename should be the name of a text file in the **./data folder** (NO EXTENSION).
    
    Using data of the format of data/graph4096_transformed.txt:
    1. N E
    2-N+1. start_idx end_idx weight
    
    generates a CSRPP3 Pytorch Version Graph object.
    """
    lines = open(f"./data/{filename}.txt", "r").readlines()
    N, E = int(lines[0].split()[0].strip()), int(lines[0].split()[1].strip())
    
    start_time = perf_counter()
    graph = CSRPP3_Torch(init_vertices=N, max_vertices_per_node=64, directed=True, weighted=True)
    print("\x1b[2mInitialized CSRPP3 PyTorch Graph (%.4fs)\x1b[0m" % (perf_counter() - start_time))
    
    total_time = 0
    
    for i in range(1, E+1):
        line = lines[i].split()
        start_idx, end_idx, weight = int(line[0].strip()), int(line[1].strip()), int(line[2].strip())
        # print("\x1b[34m[Graph Gen] Adding edge from %d to %d with weight %d\x1b[0m" % (start_idx, end_idx, weight))
        start_time = perf_counter()
        graph.add_edge(start_idx, end_idx, weight)
        total_time += perf_counter() - start_time
    
    print(f"Total time for PyTorch add edge: {total_time:.4f}s")
        
    return graph

def get_csrpp3_torch_2_from_transformed_data(filename: str) -> CSRPP3_Torch:
    """
    IMPORTANT! filename should be the name of a text file in the **./data folder** (NO EXTENSION).
    
    Using data of the format of data/graph4096_transformed.txt:
    1. N E
    2-N+1. start_idx end_idx weight
    
    generates a CSRPP3_2 Pytorch Version Graph object.
    """
    lines = open(f"./data/{filename}.txt", "r").readlines()
    N, E = int(lines[0].split()[0].strip()), int(lines[0].split()[1].strip())
    
    start_time = perf_counter()
    graph = CSRPP3_Torch_2(init_vertices=N, max_vertices_per_node=64, directed=True, weighted=True)
    print("\x1b[2mInitialized CSRPP3 PyTorch Graph 2 (%.4fs)\x1b[0m" % (perf_counter() - start_time))
    
    total_time = 0
    
    for i in range(1, E+1):
        line = lines[i].split()
        start_idx, end_idx, weight = int(line[0].strip()), int(line[1].strip()), int(line[2].strip())
        # print("\x1b[34m[Graph Gen] Adding edge from %d to %d with weight %d\x1b[0m" % (start_idx, end_idx, weight))
        start_time = perf_counter()
        graph.add_edge(start_idx, end_idx, weight)
        total_time += perf_counter() - start_time
    
    print(f"Total time for PyTorch 2 add edge: {total_time:.4f}s")
        
    return graph

def get_torch_graph_3(filename: str) -> TorchGraph3:
    """
    IMPORTANT! filename should be the name of a text file in the **./data folder** (NO EXTENSION).
    
    Using data of the format of data/graph4096_transformed.txt:
    1. N E
    2-N+1. start_idx end_idx weight
    
    generates a CSRPP3_2 Pytorch Version Graph object.
    """
    lines = open(f"./data/{filename}.txt", "r").readlines()
    N, E = int(lines[0].split()[0].strip()), int(lines[0].split()[1].strip())
    
    #start_time = perf_counter()
    graph = TorchGraph3(init_vertices=N, max_vertices_per_node=64)
    #print("\x1b[2mInitialized Torch Graph 3 (%.4fs)\x1b[0m" % (perf_counter() - start_time))
    
    #total_time = 0
    
    for i in range(1, E+1):
        line = lines[i].split()
        start_idx, end_idx = int(line[0].strip()), int(line[1].strip())
        # print("\x1b[34m[Graph Gen] Adding edge from %d to %d with weight %d\x1b[0m" % (start_idx, end_idx, weight))
        #start_time = perf_counter()
        graph.add_edge(start_idx, end_idx)
        #total_time += perf_counter() - start_time
    
    #print(f"Total time for Torch graph 3 add edge: {total_time:.4f}s")
        
    return graph

def get_torch_graph_4(filename: str) -> TorchGraph4:
    """
    IMPORTANT! filename should be the name of a text file in the **./data folder** (NO EXTENSION).
    
    Using data of the format of data/graph4096_transformed.txt:
    1. N E
    2-N+1. start_idx end_idx weight
    
    generates a CSRPP3_2 Pytorch Version Graph object.
    """
    lines = open(f"./data/{filename}.txt", "r").readlines()
    N, E = int(lines[0].split()[0].strip()), int(lines[0].split()[1].strip())
    
    #start_time = perf_counter()
    graph = TorchGraph4(init_vertices=N)
    #print("\x1b[2mInitialized Torch Graph 4 (%.4fs)\x1b[0m" % (perf_counter() - start_time))
    
    #total_time = 0
    
    for i in range(1, E+1):
        line = lines[i].split()
        start_idx, end_idx = int(line[0].strip()), int(line[1].strip())
        # print("\x1b[34m[Graph Gen] Adding edge from %d to %d with weight %d\x1b[0m" % (start_idx, end_idx, weight))
        #start_time = perf_counter()
        graph.add_edge(start_idx, end_idx)
        #total_time += perf_counter() - start_time
    
    #print(f"Total time for Torch graph 4 add edge: {total_time:.4f}s")
        
    return graph

def get_graph_generic(filename: str, graph_class):
    """
    IMPORTANT! filename should be the name of a text file in the **./data folder** (NO EXTENSION).
    
    Using data of the format of data/graph4096_transformed.txt:
    1. N E
    2-N+1. start_idx end_idx weight
    
    generates the requested Graph object and returns it.
    """
    lines = open(f"./data/{filename}.txt", "r").readlines()
    N, E = int(lines[0].split()[0].strip()), int(lines[0].split()[1].strip())
    
    #start_time = perf_counter()
    graph = graph_class(init_vertices=N)
    #print("\x1b[2mInitialized Torch Graph 4 (%.4fs)\x1b[0m" % (perf_counter() - start_time))
    
    #total_time = 0
    
    for i in range(1, E+1):
        line = lines[i].split()
        start_idx, end_idx = int(line[0].strip()), int(line[1].strip())
        # print("\x1b[34m[Graph Gen] Adding edge from %d to %d with weight %d\x1b[0m" % (start_idx, end_idx, weight))
        #start_time = perf_counter()
        graph.add_edge(start_idx, end_idx)
        #total_time += perf_counter() - start_time
    
    #print(f"Total time for Torch graph 4 add edge: {total_time:.4f}s")
        
    return graph