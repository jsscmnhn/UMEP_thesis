# cython_parallel_example.pyx
from cython.parallel import prange
import time

def parallel_test(int n):
    cdef int i
    cdef double sum_result = 0.0

    # Use prange for parallel execution
    for i in prange(n, nogil=True):
        sum_result += i * 0.1

    return sum_result
