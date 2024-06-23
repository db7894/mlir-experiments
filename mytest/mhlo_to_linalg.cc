#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

int main(int argc, char **argv)
{
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::mhlo::MhloDialect, mlir::linalg::LinalgDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module)
  {
    llvm::errs() << "Failed to load input file.\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());

  if (mlir::failed(pm.run(*module)))
  {
    llvm::errs() << "Failed to lower MHLO to linalg.\n";
    module->dump();
    return 1;
  }

  module->dump();
  return 0;
}