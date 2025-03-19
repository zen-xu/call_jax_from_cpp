# Introduction

This repository shows examples of running JAX models from C++ code.

One option (which involves JIT compilation) is to use an HLO file to run the JAX mode.
This can be tried by running `bazel run //cpp:hlo_example`.

Another approach (relying on AOT compilation) is to serialize a pre-compiled executable
beforehand. Once the executable exists, this can be achieved by running `bazel run //cpp:aot_example`.

# Setting up the Python environment

Run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

# Generate compile_commands.json

```bash
bazel run @hedron_compile_commands//:refresh_all
```

# Generating the HLO and serialized files

```bash
python3 python/simple_jax_example.py
```
