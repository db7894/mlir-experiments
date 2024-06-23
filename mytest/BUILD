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