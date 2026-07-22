---
title: "Numba 的 CUDA 示例(3/4)：流和事件"
date: 2024-06-05T07:03:00+08:00
draft: false
categories: ["AI"]
summary: "> 本教程为 [Numba CUDA 示例](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3478311056533995521&scene=21#wechat_redirect) 第 3 部分。
>
> 按照本系列的第 3 部分，了解 Python CUDA 编程中的流和事件

## 介绍

在本系列的前两部分（[第 1 部分](https://mp.weixin.qq.com/s/LdLEhntmiHewB5z2-NjAqg)，[第 2 部分](https://mp.weixin.qq.com/s/w4qpHO7hVSZhgr0C6ZzDyQ)）中，我们学习了如何使用 GPU 编程执行简单的任务，例如高度并行的任务、使用共享内存的缩减以及设备功能。我们还学习了如何从主机对函数进行计时 — 以及为什么这可能不是对代码进行计时的最佳方式。

![使用“墨西哥湾流多彩空间平静”运行](https://miro.medium.com/v2/resize:fit:700/1*bC0prdsQmLoLoSSY2sWgmQ.png)"
---

## 在本教程中

为了提高我们的计时能力，我们将介绍 CUDA 事件及其使用方法。但在深入研究之前，我们将讨论 CUDA 流及其重要性。

Google colab 中的代码：https://colab.research.google.com/drive/1jujdw9f6rf0GoOGRHCvi82mAoXD3ufmt?usp=sharing

## 入门

导入并加载库，确保你有 GPU。

```python
import warnings
from time import perf_counter, sleep
import numpy as np
import numba
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

print(np.__version__)
print(numba.__version__)

# 忽略 NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

---
1.25.2
0.59.1
```

```python
cuda.detect()

---
Found 1 CUDA devices
id 0             b'Tesla T4'                              [SUPPORTED]
                      Compute Capability: 7.5
                           PCI Device ID: 4
                              PCI Bus ID: 0
                                    UUID: GPU-34d689f4-4c3c-eeb0-ecce-bbaabec33618
                                Watchdog: Disabled
             FP32/FP64 Performance Ratio: 32
Summary:
	1/1 devices are supported
True
```



## 流（Streams）

当我们从主机启动内核时，它的执行会在 GPU 中排队，只要 GPU 完成了之前启动的所有任务就会执行。

用户在设备中启动的许多任务可能依赖于先前的任务，因此“将它们放在同一个队列中”是有意义的。例如，如果你将数据异步复制到 GPU 以使用某个内核进行处理，则该副本必须在内核运行之前完成。

但是，如果你有两个彼此独立的内核，将它们放在同一个队列中是否有意义？可能没有！对于这些情况，CUDA 有*流*。你可以将流视为彼此独立运行的单独队列。它们也可以并发运行，即同时运行。这可以在运行许多独立任务时大大加快总运行时间。

![图 3.1。使用不同的流可以实现并发执行，从而缩短运行时间。](https://miro.medium.com/v2/resize:fit:700/0*mYDmxjFYQQ86Saw2.png)

来源：[Zhang et al. 2021](https://www.mdpi.com/1045598) (CC BY 4.0).

### Numba CUDA 中的流语义

我们将采取迄今为止学到的两个任务并将它们排队以创建规范化管道。给定一个（主机）数组`a`，我们将用其规范化版本覆盖它：

a ← a / ∑a[i]

为此，我们将使用三个内核。第一个内核 `partial_reduce` 是第二部分中的部分还原。它将返回一个`threads_per_block`-sized 数组，我们将把它传递给另一个内核 `single_thread_sum`，后者将进一步将其还原为一个单子数组（大小为 1）。这个内核将在单个区块和单个线程上运行。最后，我们将使用 `divide_by` 对原始数组和之前计算出的总和进行就地分割。所有这些操作都将在 GPU 中进行，并且应该一个接一个地运行。

```python
threads_per_block = 256
blocks_per_grid = 32 * 40

@cuda.jit
def partial_reduce(array, partial_reduction):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    s_thread = 0.0
    for i_arr in range(i_start, array.size, threads_per_grid):
        s_thread += array[i_arr]

    s_block = cuda.shared.array((threads_per_block,), numba.float32)
    tid = cuda.threadIdx.x
    s_block[tid] = s_thread
    cuda.syncthreads()

    i = cuda.blockDim.x // 2
    while (i > 0):
        if (tid < i):
            s_block[tid] += s_block[tid + i]
        cuda.syncthreads()
        i //= 2

    if tid == 0:
        partial_reduction[cuda.blockIdx.x] = s_block[0]

@cuda.jit
def single_thread_sum(partial_reduction, sum):
    sum[0] = 0.0
    for element in partial_reduction:
        sum[0] += element


@cuda.jit
def divide_by(array, val_array):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.gridsize(1)
    for i in range(i_start, array.size, threads_per_grid):
        array[i] /= val_array[0]
```

当内核调用和其他操作没有指定流时，它们将在默认流中运行。默认流是一种特殊流，其行为取决于运行的是[旧式流还是每个线程的流](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)。对我们来说，只要说如果你想实现并发，就应该在非默认流中运行任务就足够了。让我们看看如何对某些操作（例如内核启动、数组复制和数组创建复制）做到这一点。

```python
# Define host array
a = np.ones(10_000_000, dtype=np.float32)
print(f"Old sum: {a.sum():.2f}")
# Old sum: 10000000.00

# Example 3.1: Numba CUDA Stream Semantics

# Pin memory
with cuda.pinned(a):
    # Create a CUDA stream
    stream = cuda.stream()

		# 将数组复制到设备并在设备中创建。使用 Numba 时，可将数据流作为附加信息传递给 API 函数。
    dev_a = cuda.to_device(a, stream=stream)
    dev_a_reduce = cuda.device_array((blocks_per_grid,), dtype=dev_a.dtype, stream=stream)
    dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)

    # 在启动内核时，流会被传给内核启动器（"dispatcher"）配置，它位于块维度（`threads_per_block`）之后。
    partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
    single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
    divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)

    # 数组复制到主机：与复制到设备一样，当传递数据流时，复制是异步的。注意：由于写入尚未同步，打印输出很可能是无意义的。
    dev_a.copy_to_host(a, stream=stream)

# 无论何时，只要我们想确保从主机的角度来看，流中的所有操作都已完成，我们就会调用：
stream.synchronize()

# 调用后，我们可以确定 `a` 已被其规范化版本覆盖
print(f"New sum: {a.sum():.2f}")

---
New sum: 1.00
```



在我们真正谈论流之前，我们需要谈论房间里的大象：`cuda.pinned`。此上下文管理器创建一种称为*页面锁定*或*固定*内存的特殊类型的内存，CUDA 在将内存从主机传输到设备时将受益于这种内存。

主机 RAM 中的内存可随时[*分页*](https://en.wikipedia.org/wiki/Memory_paging)，也就是说，操作系统可以秘密地将对象从 RAM 移动到硬盘。这样做的目的是将不经常使用的对象移动到较慢的内存位置，让较快的 RAM 内存可用于更急需的对象。对我们来说重要的是，CUDA 不允许从可分页对象到 GPU 的异步传输。这样做是为了防止出现持续的非常慢的传输流：磁盘（分页）→ RAM → GPU。

要异步传输数据，我们必须确保数据*始终*位于 RAM 中，方法是以某种方式防止操作系统偷偷将数据隐藏在磁盘的某个地方。这就是内存固定发挥作用的地方，它创建了一个上下文，在该上下文中，参数将被“页面锁定”，即强制位于 RAM 中。参见图 3.2。

![图 3.2。可分页内存与固定（页面锁定）内存](https://miro.medium.com/v2/resize:fit:700/0*ZrWF-XxN4bznjmS0.png)

来源：[Rizvi et al. 2017](https://www.mdpi.com/215954) (CC BY 4.0).

从此以后，代码就变得非常简单了。创建一个流，然后将其传递给我们想要在该流上操作的每个 CUDA 函数。重要的是，Numba CUDA 内核配置（方括号）要求流位于块维度大小之后的第三个参数中。

> ⚠️ 注意：
>
> 通常，将流传递给 Numba CUDA API 函数不会改变其行为，只会改变其运行的流。从设备到主机的复制是一个例外。调用 `device_array.copy_to_host()`（不带参数）时，复制会同步进行。调用 `device_array.copy_to_host(stream=stream)`（带流）时，如果`device_array`未固定，则复制将同步进行。只有在`device_array`固定并传递流时，复制才会异步进行。

> 信息：
>
> Numba 提供了一个有用的上下文管理器，用于在其上下文中排队所有操作；退出上下文时，操作将同步，包括内存传输。示例 3.1 也可以写成：

```python
with cuda.pinned(a):
    stream = cuda.stream()
    with stream.auto_synchronize():
        dev_a = cuda.to_device(a, stream=stream)
        dev_a_reduce = cuda.device_array((blocks_per_grid,), dtype=dev_a.dtype, stream=stream)
        dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)
        partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
        single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
        divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)
        dev_a.copy_to_host(a, stream=stream)
```

## 将独立内核与流分离

假设我们要标准化的不是一个数组，而是多个数组。各个数组的标准化操作完全相互独立。因此，GPU 没有必要等到一个标准化结束之后再开始下一个标准化。因此，我们应该将这些任务分成单独的流。

让我们看一个规范化 10 个数组的示例 —— 每个数组都使用自己的流。

```python
# Example 3.2: Multiple streams

N_streams = 10
# 不要在此上下文中进行内存收集（去分配数组）
with cuda.defer_cleanup():
    # Create 10 streams
    streams = [cuda.stream() for _ in range(1, N_streams + 1)]

    # Create base arrays
    arrays = [
        i * np.ones(10_000_000, dtype=np.float32) for i in range(1, N_streams + 1)
    ]

    for i, arr in enumerate(arrays):
        print(f"Old sum (array {i}): {arr.sum():12.2f}")

    tics = []  # Launch start times
    for i, (stream, arr) in enumerate(zip(streams, arrays)):
        tic = perf_counter()
        with cuda.pinned(arr):
            dev_a = cuda.to_device(arr, stream=stream)
            dev_a_reduce = cuda.device_array(
                (blocks_per_grid,), dtype=dev_a.dtype, stream=stream
            )
            dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)

            partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
            single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
            divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)

            dev_a.copy_to_host(arr, stream=stream)

        toc = perf_counter()  # Stop time of launches
        print(f"Launched processing {i} in {1e3 * (toc - tic):.2f} ms")

        # 确保删除 GPU 数组的引用，这将确保在退出上下文时进行垃圾回收。
        del dev_a, dev_a_reduce, dev_a_sum

        tics.append(tic)

    tocs = []
    for i, (stream, arr) in enumerate(zip(streams, arrays)):
        stream.synchronize()
        toc = perf_counter()  # Stop time of sync
        tocs.append(toc)
        print(f"New sum (array {i}): {arr.sum():12.2f}")
    for i in range(4):
        print(f"Performed processing {i} in {1e3 * (tocs[i] - tics[i]):.2f} ms")

    print(f"Total time {1e3 * (tocs[-1] - tics[0]):.2f} ms")

---
Old sum (array 0):  10000000.00
Old sum (array 1):  20000000.00
Old sum (array 2):  30000000.00
Old sum (array 3):  40000000.00
Old sum (array 4):  50000000.00
Old sum (array 5):  60000000.00
Old sum (array 6):  70000000.00
Old sum (array 7):  80000000.00
Old sum (array 8):  90000000.00
Old sum (array 9): 100000000.00
Launched processing 0 in 14.40 ms
Launched processing 1 in 13.89 ms
Launched processing 2 in 13.79 ms
Launched processing 3 in 13.75 ms
Launched processing 4 in 13.62 ms
Launched processing 5 in 13.95 ms
Launched processing 6 in 13.99 ms
Launched processing 7 in 14.32 ms
Launched processing 8 in 13.14 ms
Launched processing 9 in 13.47 ms
New sum (array 0):         1.00
New sum (array 1):         1.00
New sum (array 2):         1.00
New sum (array 3):         1.00
New sum (array 4):         1.00
New sum (array 5):         1.00
New sum (array 6):         1.00
New sum (array 7):         1.00
New sum (array 8):         1.00
New sum (array 9):         1.00
Performed processing 0 in 145.23 ms
Performed processing 1 in 137.10 ms
Performed processing 2 in 129.31 ms
Performed processing 3 in 121.48 ms
Total time 207.54 ms
```



现在让我们与单个流进行比较。

```python
# Example 3.3: Single stream

# 不要在此上下文中进行内存收集（去分配数组）
with cuda.defer_cleanup():
    # Create 1 streams
    streams = [cuda.stream()] * N_streams

    # Create base arrays
    arrays = [
        i * np.ones(10_000_000, dtype=np.float32) for i in range(1, N_streams + 1)
    ]

    for i, arr in enumerate(arrays):
        print(f"Old sum (array {i}): {arr.sum():12.2f}")

    tics = []  # Launch start times
    for i, (stream, arr) in enumerate(zip(streams, arrays)):
        tic = perf_counter()
        
        with cuda.pinned(arr):
            dev_a = cuda.to_device(arr, stream=stream)
            dev_a_reduce = cuda.device_array(
                (blocks_per_grid,), dtype=dev_a.dtype, stream=stream
            )
            dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)

            partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
            single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
            divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)

            dev_a.copy_to_host(arr, stream=stream)

        toc = perf_counter()  # Stop time of launches
        print(f"Launched processing {i} in {1e3 * (toc - tic):.2f} ms")

        # 确保删除 GPU 数组的引用，这将确保在退出上下文时进行垃圾回收。
        del dev_a, dev_a_reduce, dev_a_sum

        tics.append(tic)

    tocs = []
    for i, (stream, arr) in enumerate(zip(streams, arrays)):
        stream.synchronize()
        toc = perf_counter()  # Stop time of sync
        tocs.append(toc)
        print(f"New sum (array {i}): {arr.sum():12.2f}")
    for i in range(4):
        print(f"Performed processing {i} in {1e3 * (tocs[i] - tics[i]):.2f} ms")

    print(f"Total time {1e3 * (tocs[-1] - tics[0]):.2f} ms")

---
Old sum (array 0):  10000000.00
Old sum (array 1):  20000000.00
Old sum (array 2):  30000000.00
Old sum (array 3):  40000000.00
Old sum (array 4):  50000000.00
Old sum (array 5):  60000000.00
Old sum (array 6):  70000000.00
Old sum (array 7):  80000000.00
Old sum (array 8):  90000000.00
Old sum (array 9): 100000000.00
Launched processing 0 in 13.26 ms
Launched processing 1 in 11.84 ms
Launched processing 2 in 11.83 ms
Launched processing 3 in 12.08 ms
Launched processing 4 in 14.21 ms
Launched processing 5 in 11.98 ms
Launched processing 6 in 11.91 ms
Launched processing 7 in 12.08 ms
Launched processing 8 in 12.13 ms
Launched processing 9 in 11.80 ms
New sum (array 0):         1.00
New sum (array 1):         1.00
New sum (array 2):         1.00
New sum (array 3):         1.00
New sum (array 4):         1.00
New sum (array 5):         1.00
New sum (array 6):         1.00
New sum (array 7):         1.00
New sum (array 8):         1.00
New sum (array 9):         1.00
Performed processing 0 in 124.64 ms
Performed processing 1 in 115.35 ms
Performed processing 2 in 107.26 ms
Performed processing 3 in 99.11 ms
Total time 159.02 ms
```

但是哪一个更快呢？运行这些示例时，使用多个流时，总时间并没有得到一致的改善。造成这种情况的原因有很多。例如，要使流并发运行，本地内存中必须有足够的空间。此外，我们从 CPU 进行计时。虽然很难知道本地内存中是否有足够的空间，但从 GPU 进行计时相对容易。让我们学习如何操作！

> 信息： 
>
> Nvidia 提供了多种用于调试 CUDA 的工具，包括用于调试 CUDA 流的工具。查看*[*Nsight Systems*](https://developer.nvidia.com/nsight-systems)*了解更多信息。

## 事件

CPU 计时代码的一个问题是，它将包含除 GPU 之外的更多操作。

值得庆幸的是，通过 CUDA 事件可以直接从 GPU 获取时间。事件只是一个时间寄存器，它记录了 GPU 中发生的事情。在某种程度上，它类似于 `time.time` 和 `time.perf_counter`，但与之不同的是，我们需要处理这样一个事实：当我们在 CPU 上编程时，我们希望对来自 GPU 的事件进行计时。

因此，除了创建时间戳（“记录”事件）之外，我们还需要确保事件与 CPU 同步，然后才能访问其值。让我们看一个简单的例子。

### 内核执行计时事件

```python
# Example 3.4: Simple events

# 事件需要初始化，但这并不影响计时。
# 我们创建两个事件，一个在计算开始时，另一个在计算结束时。
event_beg = cuda.event()
event_end = cuda.event()

# Create CUDA stream
stream = cuda.stream()

with cuda.pinned(arr):
    # 在`stream`中复制/创建队列数组
    dev_a = cuda.to_device(arr, stream=stream)
    dev_a_reduce = cuda.device_array((blocks_per_grid,), dtype=dev_a.dtype, stream=stream)

    # 从这一行开始，`event_beg` 将包含 GPU 中这一时刻的时间。
    event_beg.record(stream=stream)

    # 异步启动内核
    partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)

    # 启动 "记录"，内核运行结束时触发该记录
    event_end.record(stream=stream)

    # 未来提交到流的任务将等待 `event_end` 完成。
    event_end.wait(stream=stream)

    # 将此事件与 CPU 同步，以便我们可以使用其值。
    event_end.synchronize()

# 现在我们来计算执行内核所需的时间。请注意，我们不需要等待/同步`event_beg`，因为它的执行取决于 event_end 是否等待/同步了`event_beg`。
timing_ms = event_beg.elapsed_time(event_end)  # in miliseconds

print(f"Elapsed time {timing_ms:.2f} ms")

---
Elapsed time 0.79 ms
```



对 GPU 操作进行计时的一个有用方法是使用上下文管理器：

```python
# Example 3.5: Context Manager for CUDA Timer using Events
class CUDATimer:
    def __init__(self, stream):
        self.stream = stream
        self.event = None  # in ms

    def __enter__(self):
        self.event_beg = cuda.event()
        self.event_end = cuda.event()
        self.event_beg.record(stream=self.stream)
        return self

    def __exit__(self, type, value, traceback):
        self.event_end.record(stream=self.stream)
        self.event_end.wait(stream=self.stream)
        self.event_end.synchronize()
        self.elapsed = self.event_beg.elapsed_time(self.event_end)


stream = cuda.stream()
dev_a = cuda.to_device(arrays[0], stream=stream)
dev_a_reduce = cuda.device_array((blocks_per_grid,), dtype=dev_a.dtype, stream=stream)
with CUDATimer(stream) as cudatimer:
    partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
print(f"Elapsed time {cudatimer.elapsed:.2f} ms")

---
Elapsed time 0.65 ms
```



### 时间流事件

为了结束本系列的这一部分，我们将使用流来更好、更准确地了解我们的示例是否受益于流。

```python
# Example 3.6: Timing a single streams with events

N_streams = 10

# 不要在此上下文中进行内存收集（去分配数组）
with cuda.defer_cleanup():
    # Create 1 stream
    streams = [cuda.stream()] * N_streams

    # Create base arrays
    arrays = [
        i * np.ones(10_000_000, dtype=np.float32) for i in range(1, N_streams + 1)
    ]

    events_beg = []  # Launch start times
    events_end = []  # End start times
    for i, (stream, arr) in enumerate(zip(streams, arrays)):
        with cuda.pinned(arr):
            # 宣布事件并记录开始
            event_beg = cuda.event()
            event_end = cuda.event()
            event_beg.record(stream=stream)

            # 执行所有 CUDA 操作
            dev_a = cuda.to_device(arr, stream=stream)
            dev_a_reduce = cuda.device_array(
                (blocks_per_grid,), dtype=dev_a.dtype, stream=stream
            )
            dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)
            partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
            single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
            divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)
            dev_a.copy_to_host(arr, stream=stream)

            # Record end
            event_end.record(stream=stream)

        events_beg.append(event_beg)
        events_end.append(event_end)

        del dev_a, dev_a_reduce, dev_a_sum

sleep(5)  # 等待所有事件结束，不影响 GPU 计时
for event_end in events_end:
    event_end.synchronize()

# 启动的第一个 `event_beg` 是最早的事件。但最后一个 `event_end` 事件是事先不知道的。我们要找出是哪个事件：
elapsed_times = [events_beg[0].elapsed_time(event_end) for event_end in events_end]
i_stream_last = np.argmax(elapsed_times)

print(f"Last stream: {i_stream_last}")
print(f"Total time {elapsed_times[i_stream_last]:.2f} ms")

---
Last stream: 9
Total time 117.90 ms
```

```python

# Example 3.7: Timing multiple streams with events

# 不要在此上下文中进行内存收集（去分配数组）
with cuda.defer_cleanup():
    # Create 10 streams
    streams = [cuda.stream() for _ in range(1, N_streams + 1)]

    # Create base arrays
    arrays = [
        i * np.ones(10_000_000, dtype=np.float32) for i in range(1, N_streams + 1)
    ]

    events_beg = []  # Launch start times
    events_end = []  # End start times
    for i, (stream, arr) in enumerate(zip(streams, arrays)):
        with cuda.pinned(arr):
            # 宣布事件并记录开始
            event_beg = cuda.event()
            event_end = cuda.event()
            event_beg.record(stream=stream)

            # 执行所有 CUDA 操作
            dev_a = cuda.to_device(arr, stream=stream)
            dev_a_reduce = cuda.device_array(
                (blocks_per_grid,), dtype=dev_a.dtype, stream=stream
            )
            dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)
            partial_reduce[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_reduce)
            single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
            divide_by[blocks_per_grid, threads_per_block, stream](dev_a, dev_a_sum)
            dev_a.copy_to_host(arr, stream=stream)

            # Record end
            event_end.record(stream=stream)

        events_beg.append(event_beg)
        events_end.append(event_end)

        del dev_a, dev_a_reduce, dev_a_sum

sleep(5)  # 等待所有事件完成，不影响 GPU 时序
for event_end in events_end:
    event_end.synchronize()

# 启动的第一个 `event_beg` 是最早的事件。但最后一个 `event_end` 事件是事先不知道的。我们要找出是哪个事件：
elapsed_times = [events_beg[0].elapsed_time(event_end) for event_end in events_end]
i_stream_last = np.argmax(elapsed_times)

print(f"Last stream: {i_stream_last}")
print(f"Total time {elapsed_times[i_stream_last]:.2f} ms")

---
Last stream: 9
Total time 130.66 ms
```

## 结尾

CUDA 的核心在于性能。在本教程中，你学习了如何使用*Events*（事件）准确测量内核的执行时间，以便对代码进行分析。你还了解了*Streams*（流）以及如何使用它们来始终保持 GPU 忙碌，以及*pinned*（固定）或*mapped arrays*（映射数组），以及如何改善内存访问。