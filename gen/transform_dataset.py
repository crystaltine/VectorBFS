from io import TextIOWrapper

def bin_search_loc_in_prefix_sum(prefix_sum: list[int], loc: int) -> int:
    """
    Returns the largest index i such that prefix_sum[i] <= loc
    """
    l, r = 0, len(prefix_sum) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if prefix_sum[mid] <= loc:
            l = mid
        else:
            r = mid - 1
    return l

def transform_dataset_65536():
    """
    Original dataset format:
    (graph65536.txt)
    LINE NUM HERE IS 1-INDEXED

    FIRST LINE - num. of vertices in graph = N
    next N lines: line# - 2 = vertex index, num1 = total edges before that node, num2 = degree of that node

    LINE N+5 - num. of edges in graph = E
    next E lines: line# - N+6 = edge index, num1 = dest_node idx, num2 = weight of edge
    
    This function transforms that into:
    LINE 1: N E
    next E lines: start_idx, end_idx, weight
    """
    
    lines = open("./data/graph65536.txt", "r").readlines()
    N = int(lines[0].strip())
    
    mappings = [] # idx = vertex_idx, value = num of edges before that node (idx of first edge of that node)
    for i in range(1, N+1):
        mappings.append(int(lines[i].split()[0].strip()))
        
    E = int(lines[N+4].strip())
    
    with open("./data/graph65536_transformed.txt", "w") as f:
        f.write(f"{N} {E}\n")
        for i in range(N+5, N+5+E):
            line = lines[i].split()
            start_idx = bin_search_loc_in_prefix_sum(mappings, i - N - 5)
            end_idx = int(line[0].strip())
            weight = int(line[1].strip())
            f.write(f"{start_idx} {end_idx} {weight}\n")

transform_dataset_65536()