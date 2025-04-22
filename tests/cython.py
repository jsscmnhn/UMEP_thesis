import time
from parallel_test import parallel_test

n = 10**7

# Test parallel execution
start_time = time.time()
result = parallel_test(n)
end_time = time.time()
print(f"Parallel execution took {end_time - start_time:.4f} seconds.")