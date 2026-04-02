#ifndef TENSORFLOW_CORE_KERNELS_KDNN_UNARY_OP_H_
#define TENSORFLOW_CORE_KERNELS_KDNN_UNARY_OP_H_

#define _USE_MATH_DEFINES
#include <cmath>

#define EIGEN_USE_THREADS

#include "tensorflow/core/platform/bfloat16.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"

#include "kdnn_adapter.h"

namespace tensorflow {
template <KDNN::ActivationFunction Func, typename T>
class KdnnUnaryOp : public OpKernel {
 public:
  explicit KdnnUnaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto type = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({type}, {type}));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    Tensor* out = nullptr;

    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, inp.shape(), &out));

    const T* src = inp.flat<T>().data();
    T* dst = out->flat<T>().data();
    if (inp.shape().num_elements() == 0) {
      return;
    }
    KDNN::Shape tensorShape({inp.shape().num_elements()});
    KDNN::TensorInfo inputTensorInfo(tensorShape, KDNN::Element::TypeAdapter<T>::value, KDNN::Layout::A);
    KDNN::TensorInfo outputTensorInfo(tensorShape, KDNN::Element::TypeAdapter<T>::value, KDNN::Layout::A);

    thread::ThreadPool* thread_pool = 
        ctx->device()
        ->tensorflow_cpu_worker_threads()
        ->workers;
    kdnn::KDNNThreadPool kdnn_tp(thread_pool);
    KDNN::Threading::ActivateThreadpool(&kdnn_tp);
    KDNN::ActivationLayerFWD layer(inputTensorInfo, outputTensorInfo, Func);
    layer.Run(src, dst);
    KDNN::Threading::DeactivateThreadpool();
  }
};
}
#endif