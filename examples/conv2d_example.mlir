module  {
  func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<1x3x224x224xf32>) -> tuple<tensor<1x64x112x112xf32>> {
    %0 = "mhlo.convolution"(%arg1, %arg0) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 1 : i64, input_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>, kernel_input_feature_dimension = 1 : i64, kernel_output_feature_dimension = 0 : i64, kernel_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 1 : i64, output_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<3> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %1 = "mhlo.tuple"(%0) : (tensor<1x64x112x112xf32>) -> tuple<tensor<1x64x112x112xf32>>
    return %1 : tuple<tensor<1x64x112x112xf32>>
  }
}