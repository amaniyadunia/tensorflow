#include "tensorflow/core/kernels/kdnn/kdnn_unary_op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("_KdnnSigmoid").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
                        KdnnUnaryOp<KDNN::ActivationFunction::SIGMOID, float>);
} 