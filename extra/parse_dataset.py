from graphs.csr import CSRGraph
from utils import _get_progress_string

FILEPATH = 'data/graph4096.txt'

filecontents = open(FILEPATH, "r").readlines()
num_vertices = 24576 # int(filecontents[0].split(" ")[0])

graph = CSRGraph(num_vertices, directed=False, weighted=False)

ops = 0
for line in filecontents[1:]:
    strip_line = line.strip('\n')
    print(f"Processing connection: {strip_line} {_get_progress_string(ops/len(filecontents))} ({ops}/{len(filecontents)})", end='\r')
    start_node = int(strip_line.split(" ")[0])
    end_node = int(strip_line.split(" ")[1])

    graph.add_edge(start_node, end_node)
    ops += 1
    
    
    
    
