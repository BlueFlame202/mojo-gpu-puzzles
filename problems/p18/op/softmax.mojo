from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = Layout.row_major(SIZE)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # FILL IN (roughly 31 lines)
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)

    shared = LayoutTensor[
        dtype, 
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if global_i < input_size:
        shared[local_i] = input[global_i] # 1 global read per thread, but they operate simultaneously?
    else:
        shared[local_i] = input[0] # to ensure that we don't cause a problem with the max operation
    barrier()
    
    # Stage 1: Max Reduction   
    # idea: each thread looks at its neighbor and stores the max of the two in the left node
    stride = 2
    while stride <= SIZE:
        if local_i % stride == 0:
            # look at the guy to your right
            if shared[local_i + stride//2] > shared[local_i]:
                shared[local_i] = shared[local_i + stride//2]
        barrier()
        stride *= 2
    # now shared[0] has the max element, but that means we need to restore

    max_elem = shared[0]

    # Stage 2: Sum Reduction
    if global_i < input_size:
        # numerical stability
        shared[local_i] = exp(input[global_i] - max_elem) # 1 global read per thread, but they operate simultaneously?
    else:
        shared[local_i] = 0 # to ensure that we don't cause a problem with the sum operation
    barrier()

    # idea: each thread looks at its neighbor and stores the sum of the two in the left node
    stride = 2
    while stride <= SIZE:
        if local_i % stride == 0:
            # look at the guy to your right
            shared[local_i] += shared[local_i + stride//2]
        barrier()
        stride *= 2
    total = shared[0]

    # Stage 3: Finish output
    if global_i < input_size:
        # numerical stability
        output[global_i] = exp(input[global_i] - max_elem) / total # 1 global read per thread, but they operate simultaneously?


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # FILL IN (roughly 10 lines)
    # I wonder if there's a better idiomatic way to do this like with LayoutTensor
    var max_in : output.element_type = input[0]
    for i in range(1, input_size):
        if input[i] > max_in:
            max_in = input[i]

    var total : output.element_type = 0
    for i in range(input_size):
        total += exp(input[i] - max_in)
    
    for i in range(input_size):
        output[i] = exp(input[i] - max_in) / total


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
            output.to_layout_tensor()
        )
        var input_tensor = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
            input.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[layout, input_size, dtype]
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
