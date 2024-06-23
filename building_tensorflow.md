# XLA setup

It's a lot easier to have XLA handle its own bazel build / deps and just have my project live inside, so my strategy is to clone xla locally and create a separate file `mytest` that lives in the same root dir. So, setup steps are:

## Create mytest
Created a dir called `mytest` with a `BUILD` file as below:
```
exports_files([
    "lit.cfg.py",
])

cc_binary(
    name = "mhlo_to_linalg",
    srcs = ["mhlo_to_linalg.cc"],
    deps = [
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:MathDialect",
    ],
)
```
and a `lit.cfg.py` file that is the same as https://github.com/openxla/xla/blob/main/xla/lit.cfg.py

## Clone and configure XLA
First steps:
```
git clone https://github.com/openxla/xla.git
cd xla
python3.9 configure.py --backend=CPU --host_compiler=GCC
ln -s ../mytest
```

## Build
Then, from `xla` directory run:
```
bazel build //mytest:mhlo_to_linalg
```
