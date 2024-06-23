// CMD: bazel run //mytest:mhlo_to_linalg -- $(pwd)/mytest/tests/conv2d_example.mlir

module  {
  func.func @pad_conv2d_NHWC(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<2x2x1x2xf32>) -> tensor<1x4x4x2xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x3x3x1xf32>, tensor<f32>) -> tensor<1x5x5x1xf32>
    %2 = mhlo.convolution(%1, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x1xf32>, tensor<2x2x1x2xf32>) -> tensor<1x4x4x2xf32>
    return %2 : tensor<1x4x4x2xf32>
  }
}