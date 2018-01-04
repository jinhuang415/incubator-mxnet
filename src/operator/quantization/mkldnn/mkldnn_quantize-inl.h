/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_quantized_conv-inl.h
* \brief
* \author young.jin.kim@intel.com
*         deepthi.karkada@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../quantize-inl.h"

namespace mxnet {
namespace op {


template<typename SrcType, typename DstType>
void MKLQuantizeComputeKer(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);

  // check shapes
  int i_dim = inputs[0].ndim();
  int o_dim = outputs[0].ndim();

  CHECK_EQ(i_dim, o_dim);

  int total_len = 0;
  memory::dims tensor_shape;

  for (int ii = 0; ii < i_dim; ++ii) {
    CHECK_EQ(inputs[0].size(ii), outputs[0].size(ii));
    total_len += inputs[0].size(ii);
  }
  tensor_shape.push_back(total_len);

  // find a exponents from the min-max values
  float real_range = MaxAbs(*inputs[1].dptr<float>(), *inputs[2].dptr<float>());
  float quantized_range = MaxAbs(MaxValue<DstType>(), MinValue<DstType>());
  float scale = quantized_range / real_range;

  int shift_exponent = static_cast<int>(log2(scale));
  primitive_attr attr;
  int mask = 0;
  scale = pow(2. , shift_exponent);
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  attr.set_int_output_round_mode(round_down);

  // *outputs[1].dptr<float>() = -real_range;
  // *outputs[2].dptr<float>() = real_range;
  *outputs[1].dptr<float>() = scale;
  *outputs[2].dptr<float>() = scale;

  memory::format mfmt_any = memory::format::any;
  auto cpu_engine = CpuEngine::Get()->get_engine();

  auto i_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<SrcType>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto o_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<DstType>::type,
                                      memory::format::x},
                                      cpu_engine);

  auto reorder_pd  = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto input = memory(i_mpd, inputs[0].dptr<SrcType>());
  auto output = memory(o_mpd, outputs[0].dptr<DstType>());

  auto r = reorder(reorder_pd, input, output);
  stream(stream::kind::lazy).submit({r}).wait();
}

void MKLQuantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  auto out_type = param.out_type;

  if (out_type == mshadow::kInt8) {
    MKLQuantizeComputeKer<float, int8_t>(attrs,
                                         ctx,
                                         inputs,
                                         req,
                                         outputs);
  } else if (out_type == mshadow::kUint8) {
    MKLQuantizeComputeKer<float, uint8_t>(attrs,
                                         ctx,
                                         inputs,
                                         req,
                                         outputs);
  } else if (out_type == mshadow::kInt32) {
    MKLQuantizeComputeKer<float, int32_t>(attrs,
                                         ctx,
                                         inputs,
                                         req,
                                         outputs);
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
