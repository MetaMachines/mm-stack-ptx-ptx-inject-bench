cmake_minimum_required(VERSION 3.24)
project(app_stack_ptx_ptx_inject_bench C)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(Threads REQUIRED)

set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET "")
if(TARGET CUDA::nvptxcompiler_static)
  set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler_static)
elseif(TARGET CUDA::nvptxcompiler)
  set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler)
else()
  message(FATAL_ERROR "CUDAToolkit nvptxcompiler target not found. Install nvptxcompiler or update CMake/CUDA.")
endif()

set(STACK_PTX_PTX_INJECT_BENCH_MAX_UNIQUE_INJECTS 16384 CACHE STRING
  "Max unique PTX inject sites per module")

add_executable(stack_ptx_ptx_inject_bench
  stack_ptx_cubin_bench_standalone.c
  stack_ptx_inject_kernel_gen.c
)

target_include_directories(stack_ptx_ptx_inject_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(stack_ptx_ptx_inject_bench SYSTEM PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(stack_ptx_ptx_inject_bench PRIVATE
  CUDA::nvrtc
  ${STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET}
  Threads::Threads
  m
)

target_compile_features(stack_ptx_ptx_inject_bench PRIVATE c_std_11)

target_compile_definitions(stack_ptx_ptx_inject_bench PRIVATE
  PTX_INJECT_MAX_UNIQUE_INJECTS=${STACK_PTX_PTX_INJECT_BENCH_MAX_UNIQUE_INJECTS}
)

if(OpenMP_C_FOUND)
  target_link_libraries(stack_ptx_ptx_inject_bench PRIVATE OpenMP::OpenMP_C)
endif()
