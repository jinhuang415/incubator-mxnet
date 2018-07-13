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
 * \file mkldnn_quantized_conv.cc
 * \brief
 * \author Wenting Jiang, Xinyu Chen
*/

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_convolution-inl.h"
#include "../../nn/convolution-inl.h"
#include "../quantization_utils.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../elemwise_op_common.h"
#include "../../../operator/nn/mkldnn/mkldnn_base-inl.h"
namespace mxnet {
namespace op {

static void MKLDNNQuantizedConvForward(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  CHECK(in_data[0].dtype() == mshadow::kUint8 || in_data[0].dtype() == mshadow::kInt8)
    << "mkldnn_quantized_conv op only supports uint8 or int8 as input type";
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  const size_t num_inputs = (param.no_bias ? 2 : 3) + (param.with_sum ? 1 : 0);

  float data_range, weight_range, out_range;
  float quantized_data_range, quantized_weight_range, quantized_out_range;
  float data_scale, weight_scale, out_scale;
  float conv_scale = MKLDNNConvForward::NO_SCALE;
  float postsum_scale = 1.0f;
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    using red::limits::MaxValue;
    using red::limits::MinValue;
    data_range = MaxAbs(*in_data[num_inputs].data().dptr<float>(),
                        *in_data[num_inputs+1].data().dptr<float>());
    out_range =
        MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
    weight_range = MaxAbs(*in_data[num_inputs+2].data().dptr<float>(),
                          *in_data[num_inputs+3].data().dptr<float>());
    quantized_data_range = MaxAbs(MaxValue<uint8_t>(), MinValue<uint8_t>());
    quantized_weight_range = MinAbs(MaxValue<int8_t>(), MinValue<int8_t>());
    if (param.with_relu) {
      quantized_out_range = MaxAbs(MaxValue<uint8_t>(), MinValue<uint8_t>());
    } else {
      quantized_out_range = MinAbs(MaxValue<int8_t>(), MinValue<int8_t>());
    }
    data_scale = quantized_data_range / data_range;
    weight_scale = quantized_weight_range / weight_range;
    out_scale = quantized_out_range / out_range;
    conv_scale = out_scale / data_scale / weight_scale;

    if (param.with_sum) {
      float sum_range;
      size_t sum_index = param.no_bias ? 2 : 3;
      sum_range = MaxAbs(*in_data[sum_index+1].data().dptr<float>(),
                        *in_data[sum_index+2].data().dptr<float>());
      postsum_scale = sum_range/out_range;
    }
  }

  NDArray weight = in_data[conv::kWeight];
  MKLDNNConvForward &fwd = GetConvFwd(attrs, ctx.is_train,
      in_data[conv::kData], weight,
      param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut],
      conv_scale, postsum_scale);

  const mkldnn::memory *data_mem = nullptr;
  // If input is int8 and previous OP is conv+sum+relu fusion OP, then treat
  // the input data as uint8 directly so don't need to do the real reorder
  if (in_data[conv::kData].dtype() == mshadow::kInt8 && param.with_convsumrelu_in) {
    const mkldnn::memory *mem = in_data[conv::kData].GetMKLDNNData();
    data_mem = GetMKLDNNExact(mem, fwd.fwd_pd.src_primitive_desc());
  } else {
    data_mem = in_data[conv::kData].GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc());
  }
  const mkldnn::memory *weight_mem;
  // For inference, we want to reorder the weight array so we don't need to
  // reorder data every time.
  if (weight.IsDefaultData()) {
    weight_mem = GetWeights(weight, fwd.fwd_pd.weights_primitive_desc(), param.num_group);
    // We also need to modify the layout on the original weight array. The
    // data conversion happens after the weight array is used.
    weight.MKLDNNDataReorderAsync(fwd.fwd_pd.weights_primitive_desc());
  } else {
    weight_mem = weight.GetMKLDNNData();
    CHECK(weight_mem->get_primitive_desc() == fwd.fwd_pd.weights_primitive_desc());
  }
  mkldnn_output_t out_mem;
  if (param.with_sum) {
    size_t sum_index = param.no_bias ? 2 : 3;
    out_mem = mkldnn_output_t(OutDataOp::Noop, const_cast<mkldnn::memory *>(in_data[sum_index].GetMKLDNNData()));
  } else {
    out_mem = CreateMKLDNNMem(out_data[conv::kOut], fwd.fwd_pd.dst_primitive_desc(),
                              req[conv::kOut]);
  }
  const mkldnn::memory *bias_mem = nullptr;
  if (!param.no_bias)
    bias_mem = in_data[conv::kBias].GetMKLDNNDataReorder(fwd.fwd_pd.bias_primitive_desc());
  fwd.SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());

  CommitOutput(out_data[conv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
  Stream<cpu> *s = ctx.get_stream<cpu>();
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    *out_data[1].data().dptr<float>() = param.min_calib_range.value();
    *out_data[2].data().dptr<float>() = param.max_calib_range.value();
  } else {
    mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
           out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
           in_data[num_inputs].data().dptr<float>(),
           in_data[num_inputs+1].data().dptr<float>(),
           in_data[num_inputs+2].data().dptr<float>(),
           in_data[num_inputs+3].data().dptr<float>());
  }
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedConvForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
