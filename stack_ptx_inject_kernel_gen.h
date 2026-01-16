#ifndef ELITE_NLE_STACK_PTX_INJECT_KERNEL_GEN_H
#define ELITE_NLE_STACK_PTX_INJECT_KERNEL_GEN_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    STACK_PTX_KERNEL_GEN_SUCCESS = 0,
    STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE = 1,
    STACK_PTX_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER = 2,
    STACK_PTX_KERNEL_GEN_ERROR_OUT_OF_MEMORY = 3,
    STACK_PTX_KERNEL_GEN_ERROR_FORMAT = 4
} StackPtxKernelGenResult;

#ifdef __cplusplus
extern "C" {
#endif

StackPtxKernelGenResult elite_nle_stack_ptx_inject_kernel_gen(
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
);

#ifdef __cplusplus
}
#endif

#endif // ELITE_NLE_STACK_PTX_INJECT_KERNEL_GEN_H
