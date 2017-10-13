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
 * \file cudnn_batch_norm.cc
 * \brief
 * \author Junyuan Xie
*/

#include "../elemwise_op_common.h"
#include "./cudnn_batch_norm-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4

static bool BatchNormShape(const nnvm::NodeAttrs& attrs, std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, moving_mean, moving_var]";
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  in_shape->at(1) = TShape(Shape1(dshape[1]));
  in_shape->at(2) = TShape(Shape1(dshape[1]));
  in_shape->at(3) = TShape(Shape1(dshape[1]));
  in_shape->at(4) = TShape(Shape1(dshape[1]));

  out_shape->clear();
  out_shape->push_back(dshape);
  out_shape->push_back(Shape1(dshape[1]));
  out_shape->push_back(Shape1(dshape[1]));

  return true;
}

static void BatchNormCompute_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "CuDNNBatchNormOp is only available for gpu.";
}

static void BatchNormGradCompute_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "CuDNNBatchNormOp is only available for gpu.";
}

NNVM_REGISTER_OP(CuDNNBatchNorm)
.describe("Apply batch normalization to input.")
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr_parser(ParamParser<BatchNormParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta", "moving_mean", "moving_var"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "var"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{3, 4};
})
.set_attr<nnvm::FInferShape>("FInferShape", BatchNormShape)
.set_attr<FCompute>("FCompute<cpu>", BatchNormCompute_CPU)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_CuDNNBatchNorm"})
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(BatchNormParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == 3) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == 4) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

NNVM_REGISTER_OP(_backward_CuDNNBatchNorm)
.set_num_outputs(5)
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{6, 7};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<BatchNormParam>)
.set_attr<FCompute>("FCompute<cpu>", BatchNormGradCompute_CPU);

#endif  // CUDNN_MAJOR >= 4

}  // namespace op
}  // namespace mxnet
