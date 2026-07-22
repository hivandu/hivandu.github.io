---
title: "Numba 的 CUDA 示例 (2/4)：穿针引线"
date: 2024-06-02T07:03:00+08:00
draft: false
categories: ["AI"]
summary: "> 本教程为 [Numba CUDA 示例](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3478311056533995521&scene=21#wechat_redirect) 第 2 部分。
>
> 按照本系列从头开始使用 Python 学习 CUDA 编程

## 介绍

在[本系列的第一部分](https://mp.weixin.qq.com/s/LdLEhntmiHewB5z2-NjAqg)中，我们讨论了如何使用 GPU 运行高度并行算法。高度并行任务是指任务完全相互独立的任务，例如对两个数组求和或应用任何元素函数。

![使用“穿针引线赛博朋克”进行稳定扩散](https://miro.medium.com/v2/resize:fit:700/1*giYSelu6J1NR8T3dbFLPhA.png)"
---

## 在本教程中

许多任务虽然不是高度并行的，但仍可从并行化中获益。在本期的*CUDA by Numba Examples*中，我们将介绍一些允许线程协作进行计算的常用技术。本部分的 Google colab 代码：https://colab.research.google.com/drive/1hproEOKvQyBNNxvjr0qM2LPjJWNDfyp9?usp=sharing

## 入门

导入并加载库，确保您有 GPU。

```python
from time import perf_counter
import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)

---
1.25.2
0.59.1

cuda.detect()

---
Found 1 CUDA devices
id 0             b'Tesla T4'                              [SUPPORTED]
                      Compute Capability: 7.5
                           PCI Device ID: 4
                              PCI Bus ID: 0
                                    UUID: GPU-0f022a60-18f8-5de0-1f24-ad861dcd84ae
                                Watchdog: Disabled
             FP32/FP64 Performance Ratio: 32
Summary:
	1/1 devices are supported
True
```



## 线程合作

### 简单并行缩减算法

我们将从一个非常简单的问题开始本节：对数组的所有元素求和。从本质上讲，这个算法非常简单。如果不借助 NumPy，我们可以将其实现为：

```python
def sum_cpu(array):
    s = 0.0
    for i in range(array.size):
        s += array[i]
    return s
```



我知道，这看起来不太符合 Python 风格。但它确实强调了`s`跟踪数组中的*所有*元素。如果依赖于数组的每个元素，我们如何并行化该算法`s`？首先，我们需要重写算法以允许某种并行化。如果有些部分我们无法并行化，我们应该允许线程相互通信。

然而，到目前为止，我们还没有学会如何让线程相互通信……事实上，我们之前说过，不同块中的线程不会通信。我们可以考虑只启动一个块，但请记住，大多数 GPU 中的块只能有 1024 个线程！

我们如何克服这个问题？好吧，如果我们将数组拆分成 1024 个块（或适当数量的`threads_per_block`），然后分别对每个块求和，结果会怎样？最后，我们可以将每个块的总和结果相加。图 2.1 显示了 2 个块拆分的一个非常简单的示例。

![图 2.1。“分而治之”的方法对数组元素求和](https://miro.medium.com/v2/resize:fit:472/1*cafxl-v4A1f5Af-m7MqBXw.png)

我们如何在 GPU 上做到这一点？首先，我们需要将数组拆分成块。每个块只对应一个块，具有固定数量的线程。在每个块中，每个线程可以对多个数组元素求和（网格步长循环）。然后，我们必须在整个块上计算这些每个线程的值。这部分需要线程进行通信。我们将在下一个示例中介绍如何做到这一点。

由于我们是在块上并行化，因此内核的输出应为块大小。为了完成缩减，我们将其复制到 CPU 并在那里完成作业。

```python
threads_per_block = 1024  # Why not!
blocks_per_grid = 32 * 80  # Use 32 * multiple of streaming multiprocessors

# Example 2.1: Naive reduction
@cuda.jit
def reduce_naive(array, partial_reduction):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    s_thread = 0.0
    for i_arr in range(i_start, array.size, threads_per_grid):
        s_thread += array[i_arr]

    # We need to create a special *shared* array which will be able to be read
    # from and written to by every thread in the block. Each block will have its
    # own shared array. See the warning below!
    s_block = cuda.shared.array((threads_per_block,), numba.float32)
    
    # We now store the local temporary sum of a single the thread into the
    # shared array. Since the shared array is sized
    #     threads_per_block == blockDim.x
    # (1024 in this example), we should index it with `threadIdx.x`.
    tid = cuda.threadIdx.x
    s_block[tid] = s_thread
    
    # The next line synchronizes the threads in a block. It ensures that after
    # that line, all values have been written to `s_block`.
    cuda.syncthreads()

    # Finally, we need to sum the values from all threads to yield a single
    # value per block. We only need one thread for this.
    if tid == 0:
        # We store the sum of the elements of the shared array in its first
        # coordinate
        for i in range(1, threads_per_block):
            s_block[0] += s_block[i]
        # Move this partial sum to the output. Only one thread is writing here.
        partial_reduction[cuda.blockIdx.x] = s_block[0]
```



> *⚠️ 注意 ：共享数组必须*
>
> - 尽量“小”。具体大小取决于 GPU 的计算能力，通常在 48 KB 到 163 KB 之间。请参阅本表：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability 中的“*Maximum amount of shared memory per thread block*”项。
> - 在编译时有一个已知的大小（这就是为什么我们要设置共享数组 `threads_per_block` 的大小，而不是 `blockDim.x`）。的确，我们可以为任意大小的共享数组定义一个[*factory function*](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming))......但要注意这些内核的编译时间
> - 用 Numba 类型指定 dtype，而不是 Numpy 类型（别问我为什么！）。



```python
N = 1_000_000_000
a = np.arange(N, dtype=np.float32)
a /= a.sum() # a will have sum = 1 (to float32 precision)

s_cpu = a.sum()

# Highly-optimized NumPy CPU code
timing_cpu = np.empty(21)
for i in range(timing_cpu.size):
    tic = perf_counter()
    a.sum()
    toc = perf_counter()
    timing_cpu[i] = toc - tic
timing_cpu *= 1e3  # convert to ms

print(f"Elapsed time CPU: {timing_cpu.mean():.0f} ± {timing_cpu.std():.0f} ms")

---
Elapsed time CPU: 557 ± 307 ms
```

```python
dev_a = cuda.to_device(a)
dev_partial_reduction = cuda.device_array((blocks_per_grid,), dtype=a.dtype)

reduce_naive[blocks_per_grid, threads_per_block](dev_a, dev_partial_reduction)
s = dev_partial_reduction.copy_to_host().sum()  # Final reduction in CPU

np.isclose(s, s_cpu)  # Ensure we have the right number

---
True
```

```python

timing_naive = np.empty(21)
for i in range(timing_naive.size):
    tic = perf_counter()
    reduce_naive[blocks_per_grid, threads_per_block](dev_a, dev_partial_reduction)
    s = dev_partial_reduction.copy_to_host().sum()
    cuda.synchronize()
    toc = perf_counter()
    assert np.isclose(s, s_cpu)    
    timing_naive[i] = toc - tic
timing_naive *= 1e3  # convert to ms

print(f"Elapsed time naive: {timing_naive.mean():.0f} ± {timing_naive.std():.0f} ms")

---
Elapsed time naive: 30 ± 11 ms
```

我在 Google Colab 上运行了这个程序，速度提高了将近 20 倍。非常棒！

## 一种更好的并行缩减算法

您可能想知道为什么我们将所有内容都命名为“简单”。这意味着有一些非简单的方式来执行相同的功能。事实上，有很多技巧可以加速这种代码（请参阅  [*Optimizing Parallel Reduction in CUDA*](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 演示以获取基准）。

在我们展示更好的方法之前，让我们回顾一下内核的最后一部分：

```python
if tid == 0:  # Single thread taking care of business
    for i in range(1, threads_per_block):
        s_block[0] += s_block[i]
    partial_reduction[cuda.blockIdx.x] = s_block[0]
```

我们几乎把所有事情都并行化了，但在内核末尾，我们让一个线程负责对共享数组 `s_block` 的所有 `threads_per_block` 元素求和。我们为什么不把这个总和也并行化呢？

听起来不错，怎么做呢？图 2.2 显示了如何实现 `threads_per_block` 大小为 16 的函数。我们首先运行 8 个线程，第一个线程将对 `s_block[0]` 和 `s_block[8]` 中的值求和。第二个线程对 `s_block[1] `和 `s_block[9]` 中的值求和，直到最后一个线程将对`s_block[7]` 和 `s_block[15]` 中的值求和。

下一步，只需要前 4 个线程工作。第一个线程将计算 `s_block[0]` 和 `s_block[4]` 的总和；第二个线程将计算 `s_block[1]` 和 `s_block[5]` 的总和；第三个线程将计算 `s_block[2]` 和 `s_block[6]` 的总和；第四个线程和最后一个线程将计算 `s_block[3]` 和 `s_block[7]` 的总和。

在第三步中，我们现在只需要 2 个线程来处理 `s_block`的前 4 个元素。第四步也是最后一步将使用一个线程来对 2 个元素求和。

由于工作已在线程之间分配，因此它是并行的。当然，它不是由每个线程均等分配的，但这是一种改进。从计算上讲，此算法是 O(log2( `threads_per_block`))，而第一个算法是 O( `threads_per_block`)。在我们的示例中，原始算法需要 1024 次操作，而改进算法只需要 10 次！

最后还有一个细节。在每一步中，我们都需要确保所有线程都已写入共享数组。所以我们必须调用`cuda.syncthreads()`。

![图 2.2 通过顺序寻址进行缩减](https://miro.medium.com/v2/resize:fit:700/1*r9aAAWeiXg_stKq_90pfxw.png)

来源：Mark Harris，[*Optimizing Parallel Reduction in CUDA*](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

```python
# Example 2.2: Better reduction
@cuda.jit
def reduce_better(array, partial_reduction):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    s_thread = 0.0
    for i_arr in range(i_start, array.size, threads_per_grid):
        s_thread += array[i_arr]

    # We need to create a special *shared* array which will be able to be read
    # from and written to by every thread in the block. Each block will have its
    # own shared array. See the warning below!
    s_block = cuda.shared.array((threads_per_block,), numba.float32)
    
    # We now store the local temporary sum of the thread into the shared array.
    # Since the shared array is sized threads_per_block == blockDim.x,
    # we should index it with `threadIdx.x`.
    tid = cuda.threadIdx.x
    s_block[tid] = s_thread
    
    # The next line synchronizes the threads in a block. It ensures that after
    # that line, all values have been written to `s_block`.
    cuda.syncthreads()

    i = cuda.blockDim.x // 2
    while (i > 0):
        if (tid < i):
            s_block[tid] += s_block[tid + i]
        cuda.syncthreads()
        i //= 2

    if tid == 0:
        partial_reduction[cuda.blockIdx.x] = s_block[0]


reduce_better[blocks_per_grid, threads_per_block](dev_a, dev_partial_reduction)
s = dev_partial_reduction.copy_to_host().sum()  # Final reduction in CPU

np.isclose(s, s_cpu)

---
True
```

```python
timing_naive = np.empty(21)
for i in range(timing_naive.size):
    tic = perf_counter()
    reduce_better[blocks_per_grid, threads_per_block](dev_a, dev_partial_reduction)
    s = dev_partial_reduction.copy_to_host().sum()
    cuda.synchronize()
    toc = perf_counter()
    assert np.isclose(s, s_cpu)    
    timing_naive[i] = toc - tic
timing_naive *= 1e3  # convert to ms

print(f"Elapsed time better: {timing_naive.mean():.0f} ± {timing_naive.std():.0f} ms")

---
Elapsed time better: 23 ± 1 ms
```

在 Google Colab 上，这比简单方法快约 30%。

> ⚠️ 注意：你可能会想把 `syncthreads` 移到 `if` 块内部，因为每一步之后，超过当前线程数一半的内核将不会被使用。但是，这样做会让调用 `syncthreads` 的 CUDA 线程停止并等待其他线程，而其他线程则会继续运行。因此，停止的线程将永远等待永远不会停止同步的线程。这给我们的启示是：如果要同步线程，请确保所有线程都调用了 `cuda.syncthreads()`。

```
i = cuda.blockDim.x // 2 
while (i > 0): 
    if (tid < i): 
        s_block[tid] += s_block[tid + i] 
        cuda.syncthreads() # 不要放在这里
    cuda.syncthreads() # 而不是这里
    i //= 2
```

## 减少 Numba

由于上述缩减算法并不简单，Numba 提供了一个便捷`cuda.reduce`装饰器，可将二元函数转换为缩减算法。上面的长而复杂的算法可以用以下方法替代：

```python
# Example 2.3: Numba reduction
@cuda.reduce
def reduce_numba(a, b):
    return a + b

# Compile and check
s = reduce_numba(dev_a)

np.isclose(s, s_cpu)

---
True
```

```python
# Time
timing_numba = np.empty(21)
for i in range(timing_numba.size):
    tic = perf_counter()
    s = reduce_numba(dev_a)
    toc = perf_counter()
    assert np.isclose(s, s_cpu)    
    timing_numba[i] = toc - tic
timing_numba *= 1e3  # convert to ms

print(f"Elapsed time better: {timing_numba.mean():.0f} ± {timing_numba.std():.0f} ms")

---
Elapsed time better: 20 ± 0 ms
```



就我个人而言，我发现手写缩减通常要快得多（至少快 2 倍），但 Numba 递归非常容易使用。话虽如此，我还是鼓励大家阅读 [reduction code in the Numba source code](https://github.com/numba/numba/blob/main/numba/cuda/kernels/reduction.py).

还需要注意的是，默认情况下，reduction 会复制到主机，这会强制同步。为了避免这种情况，您可以使用设备数组作为输出来调用 Reduce：

```python
dev_s = cuda.device_array((1,), dtype=s)

reduce_numba(dev_a, res=dev_s)

s = dev_s.copy_to_host()[0]
np.isclose(s, s_cpu)

---
True
```



## 2D 缩减示例

并行缩减技术很棒，但如何将其扩展到更高维度并不明显。虽然我们总是可以使用解开的数组 ( `array2d.ravel()`) 来调用 Numba 缩减，但了解如何手动缩减多维数组非常重要。

在这个例子中，我们将结合所学的关于 2D 内核的知识和所学的关于 1D 缩减的知识来计算 2D 缩减。

```python
threads_per_block_2d = (16, 16)  #  256 threads total
blocks_per_grid_2d = (64, 64)

# Total number of threads in a 2D block (has to be an int)
shared_array_len = int(np.prod(threads_per_block_2d))

# Example 2.4: 2D reduction with 1D shared array
@cuda.jit
def reduce2d(array2d, partial_reduction2d):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)

    s_thread = 0.0
    for i0 in range(iy, array2d.shape[0], threads_per_grid_x):
        for i1 in range(ix, array2d.shape[1], threads_per_grid_y):
            s_thread += array2d[i0, i1]

    # Allocate shared array
    s_block = cuda.shared.array(shared_array_len, numba.float32)

    # Index the threads linearly: each tid identifies a unique thread in the
    # 2D grid.
    tid = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
    s_block[tid] = s_thread

    cuda.syncthreads()

    # We can use the same smart reduction algorithm by remembering that
    #     shared_array_len == blockDim.x * cuda.blockDim.y
    # So we just need to start our indexing accordingly.
    i = (cuda.blockDim.x * cuda.blockDim.y) // 2
    while (i != 0):
        if (tid < i):
            s_block[tid] += s_block[tid + i]
        cuda.syncthreads()
        i //= 2
    
    # Store reduction in a 2D array the same size as the 2D blocks
    if tid == 0:
        partial_reduction2d[cuda.blockIdx.x, cuda.blockIdx.y] = s_block[0]


N_2D = (20_000, 20_000)
a_2d = np.arange(np.prod(N_2D), dtype=np.float32).reshape(N_2D)
a_2d /= a_2d.sum() # a_2d will have sum = 1 (to float32 precision)

s_2d_cpu = a_2d.sum()

dev_a_2d = cuda.to_device(a_2d)
dev_partial_reduction_2d = cuda.device_array(blocks_per_grid_2d, dtype=a.dtype)

reduce2d[blocks_per_grid_2d, threads_per_block_2d](dev_a_2d, dev_partial_reduction_2d)
s_2d = dev_partial_reduction_2d.copy_to_host().sum()  # Final reduction in CPU

np.isclose(s_2d, s_2d_cpu)  # Ensure we have the right number

---
True
```



```python
timing_2d = np.empty(21)
for i in range(timing_2d.size):
    tic = perf_counter()
    reduce2d[blocks_per_grid_2d, threads_per_block_2d](dev_a_2d, dev_partial_reduction_2d)
    s_2d = dev_partial_reduction_2d.copy_to_host().sum()
    cuda.synchronize()
    toc = perf_counter()
    assert np.isclose(s_2d, s_2d_cpu)    
    timing_2d[i] = toc - tic
timing_2d *= 1e3  # convert to ms

print(f"Elapsed time better: {timing_2d.mean():.0f} ± {timing_2d.std():.0f} ms")

---
Elapsed time better: 11 ± 0 ms
```



## 设备功能

到目前为止，我们只讨论了内核，它们是启动线程的特殊 GPU 函数。内核通常依赖于在 GPU 中定义的较小函数，这些函数只能访问 GPU 数组。这些被称为*设备函数*。与内核不同的是，它们可以返回值。

为了结束本部分教程，我们将展示一个跨不同内核使用设备函数的示例。该示例还将强调在使用共享数组时同步线程的重要性。

> 注意：在较新版本的 CUDA 中，内核可以启动其他内核。这称为动态并行，Numba CUDA 尚不支持。*

## 2D 共享数组示例

在此示例中，我们将在固定大小的数组中创建波纹图案。我们首先需要声明将使用的线程数，因为这是共享数组所需的。

```python
threads_16 = 16

import math

@cuda.jit(device=True, inline=True)  # inlining can speed up execution
def amplitude(ix, iy):
    return (1 + math.sin(2 * math.pi * (ix - 64) / 256)) * (
        1 + math.sin(2 * math.pi * (iy - 64) / 256)
    )

# Example 2.5a: 2D Shared Array
@cuda.jit
def blobs_2d(array2d):
    ix, iy = cuda.grid(2)
    tix, tiy = cuda.threadIdx.x, cuda.threadIdx.y

    shared = cuda.shared.array((threads_16, threads_16), numba.float32)
    shared[tiy, tix] = amplitude(iy, ix)
    cuda.syncthreads()

    array2d[iy, ix] = shared[15 - tiy, 15 - tix]

# Example 2.5b: 2D Shared Array without synchronize
@cuda.jit
def blobs_2d_wrong(array2d):
    ix, iy = cuda.grid(2)
    tix, tiy = cuda.threadIdx.x, cuda.threadIdx.y

    shared = cuda.shared.array((threads_16, threads_16), numba.float32)
    shared[tiy, tix] = amplitude(iy, ix)

    # When we don't sync threads, we may have not written to shared
    # yet, or even have overwritten it by the time we write to array2d
    array2d[iy, ix] = shared[15 - tiy, 15 - tix]


N_img = 1024
blocks = (N_img // threads_16, N_img // threads_16)
threads = (threads_16, threads_16)

dev_image = cuda.device_array((N_img, N_img), dtype=np.float32)
dev_image_wrong = cuda.device_array((N_img, N_img), dtype=np.float32)

blobs_2d[blocks, threads](dev_image)
blobs_2d_wrong[blocks, threads](dev_image_wrong)

image = dev_image.copy_to_host()
image_wrong = dev_image_wrong.copy_to_host()

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image.T, cmap="nipy_spectral")
ax2.imshow(image_wrong.T, cmap="nipy_spectral")
for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
```



![图 2.3。左图：同步（正确）内核的结果。右图：未同步（错误）内核的结果](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202405291809038.png)

## 结论

在本教程中，您学习了如何开发需要*缩减*模式来处理一维和二维数组的内核。在此过程中，我们学习了如何利用共享数组和设备功能。