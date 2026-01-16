# Stack PTX / PTX Inject Benchmark

Standalone benchmark that measures stack-ptx stub generation plus PTX-to-cubin
compilation. It generates large CUDA kernels, injects stack-ptx stubs, and
compiles a cubin per module in parallel. Each thread has its own bump allocator
and reuses the same memory pool inside the hot loop.

This version embeds `ptx_inject.h`, `stack_ptx.h`, and the kernel generator. The `stack_ptx.h` and
`ptx_inject.h` headers are copied from the mm-ptx repo: https://github.com/MetaMachines/mm-ptx

## Why this is useful

- Measures end-to-end compile throughput (stub render + PTX inject + cubin).
- Stress tests nvPTXCompiler with large generated kernels and many inject sites.
- Compares compile scaling across OpenMP threads without extra runtime overhead.
- Reproducible: all program generation happens locally with a fixed seed.

## How it works (in depth)

1) Kernel generation

- The tool emits `num_kernels` CUDA kernels. Each kernel contains
  `groups_per_kernel` inject sites, selected by `blockIdx.x` (the “group”).
- The total number of inject sites (programs) per module is
  `num_kernels * groups_per_kernel`. Each inject site is named `func_<id>`.

2) Program generation (random stack-ptx)

- For each program, it builds a stack-ptx instruction list of length
  `ptx-instructions-per-program`.
- The last instruction is always `return`.
- Earlier instructions are randomly sampled from a pool of simple F32 ops
  (add, mul, fma, div, sin, etc.), plus random input reads or random constants.
- If the input type is U32, the generator inserts a cast to F32 when needed.
- This means each program is a different random instruction sequence up to the
  requested length, and every module is filled with unique programs.

3) Stub compilation and injection

- Each program is compiled into a PTX stub via `stack_ptx_compile` from
  `stack_ptx.h`.
- Stubs are injected into the kernel PTX using `ptx_inject` before cubin
  compilation.
- The benchmark measures the full time for stub compilation + injection +
  nvPTXCompiler cubin generation.

4) Kernel execution model (what a CTA does)

- Each CTA (CUDA block) is tied to a specific program via `blockIdx.x`.
  That program’s injected PTX executes for every element in the tile handled
  by that CTA.
- `blockIdx.y` selects a tile in the training dimension. Threads in the block
  iterate over `i` in `[tile_start, tile_end)` with stride `blockDim.x`.
- For each `i`, the kernel loads an input vector from `data` and runs the
  injected PTX to produce `embed_dims` outputs.

## Tensor layout

Inputs are passed as:

- `data`: a 2D tensor with shape `[input_dims, num_train]`, stored with stride
  `ld_input` between input dimensions. Indexing is:
  `data[dim * ld_input + i]` for `dim in [0, input_dims)`.

Outputs are written as:

- `embed`: a 3D tensor with shape `[programs, embed_dims, num_train]`.
  The program index is the inject site id (`local_idx`), and the layout is:
  `embed[local_idx * batch_stride + dim * ld_embed + i]`.

So each CTA executes its own random program and writes a slice of the output
tensor for its program id.

## Dependencies

- CUDA Toolkit (nvrtc + nvptxcompiler).
- A C toolchain and Make or CMake.

Beyond a standard C compiler, the only external dependency is the CUDA Toolkit.

OpenMP is optional. If it is not found or disabled, the benchmark runs single-core.

## Build

From the repo root:

```bash
cmake -S . -B build
cmake --build build --target stack_ptx_ptx_inject_bench
```

### Makefile build

From this folder:

```bash
make
```

Override CUDA path or disable OpenMP:

```bash
make CUDA_HOME=/opt/cuda OPENMP=0
```

If your toolkit only ships `libnvptxcompiler_static.a`, the Makefile auto-detects
it. You can also override explicitly:

```bash
make NVPTXCOMPILER_LIB=nvptxcompiler_static
```

## Usage

```bash
./build/stack_ptx_ptx_inject_bench \
  --sm sm_80 --modules 128 --kernels 16 --groups-per-kernel 128
```

## Benchmarks

Collected runs live in [benchmarks/](benchmarks/).

## Options (exhaustive)

- `-h`, `--help` Show usage.
- `--modules N` Total modules to compile (default: 128).
- `--kernels N` Kernels per module (default: 16).
- `--groups-per-kernel N` Inject sites per kernel (default: 128).
- `--tile-size N` Kernel tile size (default: 256).
- `--embed-dims N` Output dims per inject (default: 1).
- `--input-dims N` Input dims per inject (default: 1).
- `--input-type STR` Input type: `F32` or `U32` (default: `F32`).
- `--ptx-instructions-per-program N` Instructions per program, includes return (default: 32, must be >= 2).
- `--program-execution-limit N` Stack-ptx execution limit (default: 100).
- `--workspace-bytes N` Per-thread workspace bytes (default: auto).
- `--cores N` OpenMP threads (default: OpenMP runtime).
- `--seed N` RNG seed (default: 1).
- `--sm SM` GPU SM for both stages (e.g. `sm_90`, `90`, `9.0`).
- `--sm-ptx SM` GPU SM for CUDA -> PTX (NVRTC).
- `--sm-cubin SM` GPU SM for PTX -> cubin (nvPTXCompiler).
- `--dump-cu PATH` Write generated CUDA source to PATH.
- `--dump-ptx PATH` Write NVRTC PTX to PATH.
- `--dump-module-ptx PATH` Write injected PTX for one module to PATH.
- `--dump-module-index N` Module index to dump (default: 0).
- `--verbose` Print NVRTC info logs.

## Output

The benchmark prints a compact summary including PTX version, nvPTXCompiler API
version, SMs for both stages, CPU label, configuration, wall time, modules/sec
(and per thread), and full compile throughput in us/prog and progs/sec (plus
per-thread equivalents).

## Notes

- The cubin is compiled for `--sm-cubin` (or `--sm`); PTX is generated for `--sm-ptx` (or `--sm`).
- SM fallback for unset stages uses `STACK_PTX_COMPILER_SM`, then `STACK_PTX_NNG_SM`, else `8.0`.
- If you see failures, increase `--workspace-bytes` or lower `groups-per-kernel`.

## Further documentation

- mm-ptx repo: https://github.com/MetaMachines/mm-ptx
- mm-ptx README: https://github.com/MetaMachines/mm-ptx/blob/master/README.md
- stack-ptx docs: https://github.com/MetaMachines/mm-ptx/blob/master/STACK_PTX.md
- ptx-inject docs: https://github.com/MetaMachines/mm-ptx/blob/master/PTX_INJECT.md
