from gen.csrpp3_from_transformed_data import (
    get_csrpp3_from_transformed_data, 
    get_graph_generic
)
from graphs.csrpp3 import BitArrayGraph, TorchGraph3, TorchGraph4
from time import perf_counter
from random import randint

MAX_INDEX = 4096
FILENAME = 'graph4096_transformed'
GRAPH_CLASS = TorchGraph4
TESTSET_LEN = 25

start_time = perf_counter()
graph = get_csrpp3_from_transformed_data(FILENAME)
print(f"Created \x1b[34mPython\x1b[0m Graph ({1000*(perf_counter() - start_time):.1f}ms)")

start_time = perf_counter()
alt_graph = get_graph_generic(FILENAME, GRAPH_CLASS)
print(f"Created \x1b[33mAlt Graph\x1b[0m ({1000*(perf_counter() - start_time):.1f}ms)")

def run_test_pair(start, end):
    """
    Returns tuple of (python answer, alt answer, python time ms, alt time ms, start node, end node)
    """
    start_time = perf_counter()
    python_ans = graph.bfs(start, end)
    python_time = 1000*(perf_counter() - start_time)
    
    start_time = perf_counter()
    alt_ans = alt_graph.bfs(start, end)
    alt_time = 1000*(perf_counter() - start_time)
    
    return python_ans, alt_ans, python_time, alt_time, start, end

testset = []
result = []

TESTSET_1 = [(i, i+1) for i in range(0, TESTSET_LEN*2, 2)]

use_testset = input("Use pre-generated testset? (y/anything else): ")

if use_testset == 'y' or use_testset == 'Y':
    for i in range(TESTSET_LEN):
        v1, v2 = TESTSET_1[i]
        testset.append((v1, v2))

else:
    # gen testset
    for i in range(TESTSET_LEN):
        v1 = randint(0, MAX_INDEX)
        v2 = randint(0, MAX_INDEX)
        testset.append((v1, v2))
    
# run tests
total_speedup_multiplier = 0
total_time_python = 0
total_time_alt = 0

total_correct = 0

count = 0
for i, j in testset:
    print(f"Running {TESTSET_LEN} tests... {count}/{TESTSET_LEN}", end="\r")
    result.append(run_test_pair(i, j))
    count += 1
print(f"Running {TESTSET_LEN} tests... \x1b[32mComplete\x1b[0m")
print('\n')

for entry in result:
    success = entry[0] == entry[1]
    faster_factor = entry[2] / entry[3]
    
    total_time_python += entry[2]
    total_time_alt += entry[3]

    total_speedup_multiplier += faster_factor
    
    if (success): total_correct += 1
    
    test_info_str = f"\x1b[2mBFS\x1b[0m {entry[4]} \x1b[2m->\x1b[0m {entry[5]}"
    prefix_str = "\x1b[32m\x1b[1mPASS\x1b[0m" if success else "\x1b[31m\x1b[1mFAIL\x1b[0m"
    faster_factor_str = f"(\x1b[1m\x1b[35mx{faster_factor:.2f}\x1b[0m)"
    ans_str = f"\x1b[36mPython\x1b[0m/\x1b[33mAlt\x1b[0m: \x1b[36m{entry[0]}\x1b[0m/\x1b[33m{entry[1]}\x1b[0m" 
    time_str = f"\x1b[36m{entry[2]:.2f}ms\x1b[0m/\x1b[33m{entry[3]:.2f}ms\x1b[0m"
    
    print(f"{prefix_str} {test_info_str} {ans_str} {time_str} {faster_factor_str}")

correctness_color = "\x1b[32m" if total_correct == TESTSET_LEN else "\x1b[31m"
print(f"\nTotal correct: {correctness_color}{total_correct}\x1b[0m/\x1b[32m{TESTSET_LEN}\x1b[0m ({correctness_color}{100*total_correct/TESTSET_LEN:.2f}%\x1b[0m)") 
print(f"Comparison: \x1b[35mx{(total_speedup_multiplier / TESTSET_LEN):.2f} average factor\x1b[0m / \x1b[33mx{(total_time_python / total_time_alt):.2f} average timesave\x1b[0m")
print(f"Tested Graph Class: \x1b[33m{GRAPH_CLASS.__name__}\x1b[0m vs Default implementation")
print(f"Graph Info: \x1b[34m4096 nodes 24576 edges\x1b[0m")