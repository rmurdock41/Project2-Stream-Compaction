CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

- Muqiao Lei
  
  [LinkedIn](https://www.linkedin.com/in/muqiao-lei-633304242/) · [GitHub](https://github.com/rmurdock41)

- Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz, NVIDIA GeForce RTX 3060 Laptop GPU (Personal Computer)

## Project Description

This project implements GPU stream compaction in CUDA from scratch. Stream compaction removes unwanted elements (zeros) from an array of integers, creating a packed array of non-zero elements.

The project implements various parallel algorithms for **prefix sum (scan)** operations on GPU using CUDA. Scan is a fundamental parallel algorithm that computes cumulative sums and serves as a building block for many GPU computing applications.

The scan implementations are then used to build **GPU stream compaction**, which removes unwanted elements (zeros) from arrays. This algorithm is essential for accelerating path tracers by efficiently removing terminated light paths from ray arrays.

### Features Implemented

**Core Implementations:**

- **CPU Serial Scan**: Sequential exclusive prefix sum for baseline comparison
- **Naive GPU Scan**: Basic parallel implementation from GPU Gems 3, Section 39.2.1
- **Work-Efficient GPU Scan**: Optimized algorithm from GPU Gems 3, Section 39.2.2
- **Thrust Scan**: Finished Thrust's Implementation
- **CPU Stream Compaction**: Both direct implementation and scan-based approach using map-scan-scatter
- **GPU Stream Compaction**: Efficient parallel implementation using work-efficient scan with map-scan-scatter approach

**Extra Credit:**

- **Radix Sort (+10 points)**: GPU integer sorting using stream compaction primitives with dynamic bit optimization
- **Shared Memory Optimization (+10 points)**: Work-efficient scan implementation using shared memory with block-level optimization, recursive block sum handling, and optimized memory access patterns
- **GPU Performance Optimization (+5 points)**: Achieved work-efficient GPU scan perform much better than CPU

**Algorithm Details:**

The **Naive GPU Scan** uses double-buffering to avoid race conditions. Each iteration doubles the stride for pairwise additions across the array.

The **Work-Efficient GPU Scan** uses up-sweep phase to build a binary tree reduction and down-sweep phase to propagate partial sums. The implementation includes multi-level shared memory optimization and supports non-power-of-two arrays.

**Stream Compaction** uses the three-step map-scan-scatter approach. It converts input to boolean array, performs exclusive scan to get output indices, then scatters non-zero elements to computed positions.

The **Radix Sort** implementation features dynamic bit optimization that processes only necessary bits based on maximum value. It uses combined kernels for bit extraction and Thrust integration.

### Block Size Optimization

<img src="img\blocksize.png" title="" alt="" width="677">

Block sizes were optimized for minimal runtime through empirical testing:

| Algorithm           | Optimal Block Size | Performance vs Block Size 32 |
| ------------------- | ------------------ | ---------------------------- |
| Naive GPU Scan      | 256                | 2.54x improvement            |
| Work-Efficient Scan | 256                | Stable (±15% variation)      |
| GPU Compact         | 256                | Stable (±10% variation)      |

The naive implementation shows dramatic sensitivity to block size, while work-efficient algorithms demonstrate robustness across different block sizes. Block size 256 was determined to be optimal and used for all subsequent performance comparisons. Larger block sizes like 1024 show performance degradation due to reduced occupancy and increased shared memory pressure per streaming multiprocessor.

### GPU Scan Implementations vs CPU Comparison

<img src="img\po2.png" title="" alt="" width="677">

Performance comparison across array sizes from 2^8 to 2^24 elements:

**Power-of-Two Arrays:**

| Array Size        | CPU Scan | Naive GPU | Work-Efficient | Thrust  | Radix Sort |
| ----------------- | -------- | --------- | -------------- | ------- | ---------- |
| 2^8 (256)         | 0.0009ms | 0.503ms   | 0.09ms         | 0.089ms | 1.058ms    |
| 2^12 (4,096)      | 0.009ms  | 0.701ms   | 0.22ms         | 0.079ms | 1.058ms    |
| 2^16 (65,536)     | 0.151ms  | 0.899ms   | 0.278ms        | 0.102ms | 1.006ms    |
| 2^20 (1,048,576)  | 2.222ms  | 1.643ms   | 0.618ms        | 0.237ms | 2.274ms    |
| 2^24 (16,777,216) | 28.419ms | 13.319ms  | 2.875ms        | 0.724ms | 14.581ms   |

**Detailed Performance Analysis:**

The **CPU implementation** gets slower and slower as array size grows. It starts fast at small sizes but becomes very slow at large scales because it processes one element at a time.
The **Naive GPU implementation** performs poorly at small array sizes because of GPU startup costs. It starts beating CPU around 2^20 elements but then hits a wall because of bad memory access patterns.
The **Work-Efficient implementation** performs consistently across all array sizes. It stays fast at small sizes and grows predictably with larger inputs because it uses better memory access and does less unnecessary work. 

**Thrust implementation** stays almost the same speed until 2^20 elements, then grows slowly. This shows it uses smart tricks like combining operations and picking different methods based on input size.

**Radix Sort performance** correlates closely with Naive GPU scan timing because both algorithms perform multiple passes through the data with similar memory access patterns. 

<img title="" src="img\npo2.png" alt="" width="677">

**Non-Power-of-Two Arrays (2^n - 3):** Work-efficient implementation shows superior performance on non-power-of-two arrays, achieving 40% better performance than power-of-two cases due to reduced padding overhead. This counterintuitive result occurs because the algorithm pads to the next power-of-two, and (2^n - 3) elements require less padding than 2^n elements, reducing memory bandwidth requirements and improving cache utilization.

### Performance Bottleneck Analysis

**What are the performance bottlenecks?**

Each implementation has different bottlenecks at different scales. At small array sizes, GPU kernel launch overhead dominates. At large array sizes, memory bandwidth becomes the main limit.

**Is it memory I/O or computation?**

- **CPU Implementation**: Computation bottleneck. The CPU can only process one element at a time, so adding more elements directly increases computation time. Memory access is not the problem here.
- **Naive GPU**: Memory I/O bottleneck. The algorithm does too much redundant work and accesses memory in bad patterns. Threads read from scattered locations instead of nearby addresses, wasting memory bandwidth.
- **Work-Efficient**: Memory I/O bottleneck at large scales. The algorithm does optimal work but still needs to read and write large amounts of data. Shared memory helps but global memory bandwidth becomes the limit.
- **Thrust**: Balanced between memory and computation. NVIDIA optimized both memory access patterns and computation efficiency. It accesses memory in good patterns and does minimal redundant work.
- **Radix Sort**: Memory I/O bottleneck. The scatter operation writes elements to random locations based on their values. This creates irregular memory access patterns that hurt performance.

**How do bottlenecks differ between implementations?**

Small arrays (< 2^16): All GPU implementations suffer from launch overhead. CPU wins because it has no startup cost.

Medium arrays (2^16 - 2^20): Memory access patterns start mattering. Work-efficient and Thrust pull ahead because they use memory better.

Large arrays (> 2^20): Memory bandwidth dominates everything. Thrust wins because of superior memory optimizations. CPU loses completely because it cannot use parallelism to hide memory latency.

The main lesson is that GPU performance depends heavily on memory access patterns, not just parallelism. Bad memory patterns can make a parallel algorithm slower than sequential code.

****************

** SCAN TESTS **

****************

    [  14  35  49   1  26  46  35   5   9  36  37  37  12 ...  12   0 ]

==== cpu scan, power-of-two ====
   elapsed time: 28.419ms    (std::chrono Measured)
    [   0  14  49  98  99 125 171 206 211 220 256 293 330 ... 410936659 410936671 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 30.669ms    (std::chrono Measured)
    [   0  14  49  98  99 125 171 206 211 220 256 293 330 ... 410936591 410936628 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 13.319ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 13.463ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 2.875ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.721ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.724ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.707ms    (CUDA Measured)
    passed

*****************************

** STREAM COMPACTION TESTS **

*****************************

    [   1   2   1   2   3   1   0   2   3   0   3   3   1 ...   2   0 ]

==== cpu compact without scan, power-of-two ====
   elapsed time: 37.8898ms    (std::chrono Measured)
    [   1   2   1   2   3   1   2   3   3   3   1   1   3 ...   1   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 36.8905ms    (std::chrono Measured)
    [   1   2   1   2   3   1   2   3   3   3   1   1   3 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 104.913ms    (std::chrono Measured)
    [   1   2   1   2   3   1   2   3   3   3   1   1   3 ...   1   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 4.37158ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 3.57162ms    (CUDA Measured)
    passed

********************

** RADIX SORT TESTS **

********************

    [  17  34  13  46  41  23  48  28  21   6  39  31  33 ...  40  42 ]

==== radix sort, power-of-two ====
   elapsed time: 14.581ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== radix sort, non-power-of-two ====
   elapsed time: 15.344ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]

## CMakeLists Notes

**CMakeLists.txt Modifications**: Added `radix.h` and `radix.cu` to the source files list for the extra credit radix sort implementation.

**Environment Setup**: CUDA installation located on E: drive required environment variable adjustments for proper compilation and linking.
