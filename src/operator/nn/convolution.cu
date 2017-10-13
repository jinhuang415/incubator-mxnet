/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
*/

#include "./convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

// This is to maintain one copy for each type.
template<typename DType>
static ConvolutionOp<gpu, DType> &get_op(const ConvolutionParam& param)
{
  static thread_local ConvolutionOp<gpu, DType> op;
  op.Init(param);
  return op;
}

template<typename DType>
static CuDNNConvolutionOp<DType> &get_cudnn_op(const ConvolutionParam& param,
    int forward_compute_type, int backward_compute_type,
    const std::vector<TShape>& in_shape, const std::vector<TShape>& out_shape,
    const Context& ctx, bool backward) {
  // Convolution forward has to be called before backward for this operator.
  // So we can't make this operator thread local. backward might be called
  // in another thread.
  static CuDNNConvolutionOp<DType> op;
  if (!backward)
  op.Init(param, forward_compute_type, backward_compute_type,
      in_shape, out_shape, ctx);
  return op;
}

template<>
void ConvolutionCompute<gpu>(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  int dtype = inputs[conv::kData].type_flag_;

  // If 1D convolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
      op.Forward(ctx, inputs, req, outputs);
    })
    return;
  }
  // TODO depth wise conv
#if 0
  else if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == (*in_shape)[conv::kData][1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
    op = new DepthwiseConvolutionOp<float>(param, *in_shape, *out_shape);
    return op;
  }
#endif

#if MXNET_USE_CUDNN == 1
  // On fp16-I/O instances, use fp32 compute (i.e. pseudo-fp16).
  int compute_type = (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
      op.Forward(ctx, inputs, req, outputs);
    } else if (!CuDNNConvolutionOp<DType>::Supports(param,
          compute_type, compute_type, ctx.run_ctx.ctx)) {
      LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
      ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
      op.Forward(ctx, inputs, req, outputs);
    } else {
      // The first element stores out grad.
      std::vector<TShape> in_shape(inputs.size());
      std::vector<TShape> out_shape(1, outputs[0].shape_);
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = inputs[i].shape_;
      CuDNNConvolutionOp<DType> &op = get_cudnn_op<DType>(param,
          compute_type, compute_type, in_shape, out_shape, ctx.run_ctx.ctx, false);
      op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
    op.Forward(ctx, inputs, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

template<>
void ConvolutionGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;
  int dtype = out_grad.type_flag_;

  // If 1D convolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
      // We only need in_data and weight
      in_data.resize(2);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    })
    return;
  }
  // TODO depth wise conv
#if 0
  else if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == (*in_shape)[conv::kData][1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
    op = new DepthwiseConvolutionOp<float>(param, *in_shape, *out_shape);
    return op;
  }
#endif

#if MXNET_USE_CUDNN == 1
  // On fp16-I/O instances, use fp32 compute (i.e. pseudo-fp16).
  int compute_type = (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
      // We only need in_data and weight
      in_data.resize(2);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    } else if (!CuDNNConvolutionOp<DType>::Supports(param,
          compute_type, compute_type, ctx.run_ctx.ctx)) {
      LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
      ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
      // We only need in_data and weight
      in_data.resize(2);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    } else {
      // The first element stores out grad.
      std::vector<TShape> in_shape(in_data.size());
      std::vector<TShape> out_shape(1, out_grad.shape_);
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = in_data[i].shape_;
      CuDNNConvolutionOp<DType> &op = get_cudnn_op<DType>(param,
          compute_type, compute_type, in_shape, out_shape, ctx.run_ctx.ctx, true);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    ConvolutionOp<gpu, DType> &op = get_op<DType>(param);
    // We only need in_data and weight
    in_data.resize(2);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  })
#endif  // MXNET_USE_CUDNN
}

NNVM_REGISTER_OP(Convolution)
.set_attr<FCompute>("FCompute<gpu>", ConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_Convolution)
.set_attr<FCompute>("FCompute<gpu>", ConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

