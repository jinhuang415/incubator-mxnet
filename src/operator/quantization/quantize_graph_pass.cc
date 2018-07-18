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
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>
#include <unordered_set>

namespace mxnet {
namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::Graph;

NodePtr CreateNode(std::string op_name, std::string node_name) {
  NodePtr node = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed =
      nnvm::Symbol::CreateVariable(node->attrs.name).outputs[0].node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

/*!
 * \brief Insert a node named with node_name holding the op of op_name
 * before the node current and after the node previous.
 */
NodePtr InsertNode(std::string op_name,
    std::string node_name, NodePtr current, NodeEntry previous) {
  NodePtr node = CreateNode(op_name, node_name);
  node->inputs.emplace_back(previous);
  current->inputs.emplace_back(NodeEntry{node, 0, 0});
  return node;
}

std::vector<NodeEntry> OfflineParams(std::vector<NodeEntry>&& outputs,
                                     std::unordered_set<std::string>&& offline_params) {
  std::string node_suffixs[3] = {"", "_min", "_max"};
  std::unordered_map<Node*, NodePtr> mirror_map;
  nnvm::NodeEntryMap<NodePtr> entry_var;
  auto need_offline = [&](NodePtr n) {
    return n->op() &&
           (n->op()->name == "_contrib_quantize") &&
           n->inputs[0].node->is_variable() &&
           offline_params.count(n->inputs[0].node->attrs.name);
  };
  DFSVisit(outputs, [&](const NodePtr& node) {
    for (NodeEntry& e : node->inputs) {
      if (need_offline(e.node)) {
        std::string node_name = e.node->attrs.name;
        if (!entry_var.count(e)) {
          entry_var[e] = CreateNode("nullptr", node_name + node_suffixs[e.index]);
        }
        e.node = entry_var[e];
        e.index = 0;
        e.version = 0;
      }
    }
  });
  return outputs;
}

inline bool NeedQuantize(NodePtr node, const std::unordered_set<NodePtr> excluded_nodes) {
  static auto& quantized_op_map = Op::GetAttr<mxnet::FQuantizedOp>("FQuantizedOp");
  return quantized_op_map.count(node->op()) && !excluded_nodes.count(node);
}

Graph QuantizeGraph(Graph &&src) {
  static auto& quantized_op_map = Op::GetAttr<mxnet::FQuantizedOp>("FQuantizedOp");
  static auto& need_requantize_map = Op::GetAttr<mxnet::FNeedRequantize>("FNeedRequantize");
  auto offline_params = src.GetAttr<std::unordered_set<std::string>>("offline_params");
  auto excluded_nodes = src.GetAttr<std::unordered_set<NodePtr>>("excluded_nodes");
  auto quantized_dtype = src.GetAttr<std::string>("quantized_dtype");
  auto disable_requantize = src.GetAttr<bool>("disable_requantize");
  auto input_calib_layers = src.GetAttr<std::unordered_set<std::string>>("input_calib_layers");

  // mirror_map stores the mapping from the currently visited graph to the newly created quantized
  // graph. Key is the currently visited graph's node pointer, and value is a copied node of the key
  // node. The existing key's value may be updated with the newly created quantize/dequantize op.
  std::unordered_map<Node*, NodePtr> mirror_map;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    NodePtr new_node = Node::Create();
    // If the currently visited node needs quantization, insert a quantize op node before the
    // current node and replace the current node with the quantized version in the new graph.
    if (NeedQuantize(node, excluded_nodes)) {
      auto fquantized_op = quantized_op_map[node->op()];
      // If the currently visited node's op registered the FQuantizedOp property, new_node is a
      // quantizated version of a that op, such as quantized_conv2d.
      new_node = fquantized_op(node->attrs);

      // add data into quantized op input
      for (const auto& e : node->inputs) {
        NodePtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        // If the NodeEntry e's node does not need quantization, and (the mirror_node is a variable,
        // or the mirror_node's op is not a quantize op), create quantize op, min op, and max op
        // taking mirror_entry as input to generate a quantized NDArray. Save the mapping between
        // e's source node and the newly created quantize op so that the quantize op can be
        // reused next time when the same entry is visited again.
        if (!NeedQuantize(e.node, excluded_nodes) &&
            (mirror_node->op() == nullptr ||
             mirror_node->op()->name != "_contrib_quantize")) {
          NodePtr quantize_node = InsertNode("_contrib_quantize",
            e.node->attrs.name + "_quantize", new_node, mirror_entry);
          quantize_node->attrs.dict["out_type"] = quantized_dtype;
          quantize_node->op()->attr_parser(&(quantize_node->attrs));

          // If node's input needs offline, and if node's input is an OP which
          // doesn't need quantize, then add _min/_max variables to the quantize
          // node which connects between node's input and node which means
          // the _min/_max will be calculated offline and save into parameter file
          if (input_calib_layers.count(node->attrs.name) && mirror_node->op()) {
            NodePtr min_var = CreateNode("nullptr", e.node->attrs.name + "_min");
            quantize_node->inputs.emplace_back(NodeEntry{min_var, 0, 0});
            NodePtr max_var = CreateNode("nullptr", e.node->attrs.name + "_max");
            quantize_node->inputs.emplace_back(NodeEntry{max_var, 0, 0});
          } else {
            NodePtr min_node = InsertNode("min",
                e.node->attrs.name + "_min", quantize_node, mirror_entry);
            min_node->op()->attr_parser(&(min_node->attrs));

            NodePtr max_node = InsertNode("max",
                e.node->attrs.name + "_max", quantize_node, mirror_entry);
            max_node->op()->attr_parser(&(max_node->attrs));
          }

          mirror_map[e.node.get()] = std::move(quantize_node);
        } else {
          // If the entry e's node needs quantization, or mirror_entry is from a quantize op,
          // simply add mirror_entry to the input of the new_node.
          new_node->inputs.emplace_back(mirror_entry);
        }
        // the input should be `quantize` or quantized version op now
      }

      // add min and max into quantized op input assume order of quantized op inputs is:
      // data1, data2, ..., min1, max1, min2, max2, ...
      for (const auto& e : node->inputs) {
        NodePtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        // for quantize node
        uint32_t min_index = 1;
        uint32_t max_index = 2;
        if (quantized_op_map.count(e.node->op())) {
          // here we calculate the output number (exclude min/max, in order to
          // calculate min/max index from mirror node) based on assumption that
          // there is only 1min and 1max output from mirror node (which is
          // currently true)
          size_t  num_outputs = mirror_node->num_outputs() - 2;
          min_index = num_outputs + 2 * e.index;
          max_index = num_outputs + 2 * e.index + 1;
        } else {
          CHECK(mirror_node->op()->name == "_contrib_quantize")
            << "The input is not quantize or quantized_op";
        }
        new_node->inputs.emplace_back(NodeEntry{mirror_node, min_index, 0});
        new_node->inputs.emplace_back(NodeEntry{mirror_node, max_index, 0});
      }

      // If the new_node op registered attr FNeedRequantize, insert requantize node after it.
      // Here it's assumed that the quantized_op node only produces three outputs:
      // out_data, min_range, and max_range.
      if (need_requantize_map.count(new_node->op()) > 0
          && need_requantize_map[new_node->op()](new_node->attrs)) {
        if (disable_requantize) {
          if (new_node->attrs.dict.count("with_relu") 
              || new_node->attrs.dict.count("with_postsum_relu")) {
            // Set quantized OP's out type to int8 if disable requantize
            new_node->attrs.dict["out_type"] = "uint8";
          } else {
            new_node->attrs.dict["out_type"] = "int8";
          }
        } else {
          NodePtr requantize_node = Node::Create();
          requantize_node->attrs.op = Op::Get("_contrib_requantize");
          requantize_node->attrs.name = "requantize_" + node->attrs.name;
          if (requantize_node->op()->attr_parser != nullptr) {
            requantize_node->op()->attr_parser(&(requantize_node->attrs));
          }
          for (size_t i = 0; i < 3; ++i) {
            requantize_node->inputs.emplace_back(NodeEntry{new_node, static_cast<uint32_t>(i), 0});
          }
          new_node = requantize_node;
        }
      }
    } else {
      // If the currently visited node does not need quantization, copy the current node to become
      // the new_node. Meanwhile, check whether any inputs of the current node need quantization
      // (e.g., a quantized_conv2d node), and insert a dequantize op node in the new graph if there
      // are any. Otherwise, simply add a copy of the current node's entry to the inputs of
      // the new_node.
      *new_node = *node;
      new_node->inputs.clear();
      for (const auto& e : node->inputs) {
        NodePtr mirror_node = mirror_map.at(e.node.get());
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        // if input node is quantized operator, add dequantize node
        if (NeedQuantize(e.node, excluded_nodes)) {
          // here we calculate the output number (exclude min/max, in order to
          // calculate min/max index from mirror node) based on assumption that
          // there is only 1min and 1max output from mirror node (which is
          // currently true)
          size_t num_outputs = mirror_node->num_outputs() - 2;
          uint32_t min_index = num_outputs + 2 * e.index;
          uint32_t max_index = num_outputs + 2 * e.index + 1;
          NodePtr dequantize_node = CreateNode("_contrib_dequantize",
            e.node->attrs.name + "_dequantize");
          dequantize_node->inputs.emplace_back(mirror_entry);
          dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, min_index, 0});
          dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, max_index, 0});
          dequantize_node->op()->attr_parser(&(dequantize_node->attrs));

          new_node->inputs.emplace_back(NodeEntry{dequantize_node, 0, 0});
          mirror_map[e.node.get()] = std::move(dequantize_node);
        } else if (mirror_node->op() != nullptr
                   && mirror_node->op()->name == "_contrib_quantize") {
          new_node->inputs.emplace_back(NodeEntry{mirror_node->inputs[0].node, e.index, e.version});
        } else {
          new_node->inputs.emplace_back(NodeEntry{mirror_node, e.index, e.version});
        }
      }
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    if (quantized_op_map.count(e.node->op())) {
      NodePtr mirror_node = mirror_map.at(e.node.get());
      NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
      size_t num_inputs = e.node->num_inputs();
      uint32_t min_index = num_inputs + 2 * e.index;
      uint32_t max_index = num_inputs + 2 * e.index + 1;

      NodePtr dequantize_node = CreateNode("_contrib_dequantize",
          e.node->attrs.name + "_dequantize");
      dequantize_node->inputs.emplace_back(mirror_entry);
      dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, min_index, 0});
      dequantize_node->inputs.emplace_back(NodeEntry{mirror_node, max_index, 0});
      dequantize_node->op()->attr_parser(&(dequantize_node->attrs));
      outputs.emplace_back(NodeEntry{dequantize_node, 0, 0});
    } else {
      outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
    }
  }

  if (!offline_params.empty()) outputs =
    OfflineParams(std::move(outputs), std::move(offline_params));

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

Graph GraphFusionConvSum(Graph &src) {
  std::unordered_map<Node*, NodePtr> mirror_map;
  bool patten_match;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    // for one node, create a new node as mirror node, and then go through 
    // the node's each input node, if the input node and current node doesn't
    // form a fusion pattern, then add the input node's mirror node to this
    // node's input, otherwise
    NodePtr new_node = Node::Create();
    *new_node = *node;
    new_node->inputs.clear();
    patten_match = false;
    for (const auto& e : node->inputs) {
      NodePtr mirror_node = mirror_map.at(e.node.get());
      if (e.node->op() != nullptr && e.node->op()->name == "Convolution"
             && node->op() != nullptr && node->op()->name == "elemwise_add") {
        // if matched, set current Conv mirrow node with sum
        mirror_map[node.get()] = mirror_node;
        // assume only one match for all inputs
        patten_match = true;
        mirror_node->attrs.dict["with_sum"] = "True";
        mirror_node->op()->attr_parser(&(mirror_node->attrs));
        mirror_node->inputs.emplace_back(NodeEntry{mirror_map.at(node->inputs[1].node.get()), 0,0});
        // break here have problem?
        break;
      } else {
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        new_node->inputs.emplace_back(NodeEntry{mirror_node, e.index, e.version});
      }
    }
    if (!patten_match) {
      mirror_map[node.get()] = std::move(new_node);
    }
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
  }

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

Graph GraphFusionConvRelu(Graph &src) {
  std::unordered_map<Node*, NodePtr> mirror_map;
  bool patten_match;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    // for one node, create a new node as mirror node, and then go through 
    // the node's each input node, if the input node and current node doesn't
    // form a fusion pattern, then add the input node's mirror node to this
    // node's input, otherwise
    NodePtr new_node = Node::Create();
    *new_node = *node;
    new_node->inputs.clear();
    patten_match = false;
    for (const auto& e : node->inputs) {
      NodePtr mirror_node = mirror_map.at(e.node.get());
      if (e.node->op() != nullptr && e.node->op()->name == "Convolution"
             && node->op() != nullptr && node->op()->name == "Activation"
             && node->attrs.dict["act_type"] == "relu") {
        // if matched, set current batchnorm mirrow node to convolution so 
        // batchnorm will be skipped
        mirror_map[node.get()] = mirror_node;
        // assume only one match for all inputs
        patten_match = true;
	auto iter = mirror_node->attrs.dict.find("with_sum");
	if (iter != mirror_node->attrs.dict.end()) {
	  if(iter->second == "True") {
	    mirror_node->attrs.dict["with_postsum_relu"] = "True";
	  } else{
	    mirror_node->attrs.dict["with_relu"] = "True";
	  }
	} else {
	  mirror_node->attrs.dict["with_relu"] = "True";
	}
        mirror_node->op()->attr_parser(&(mirror_node->attrs));
        // break here have problem?
        break;
      } else {
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        new_node->inputs.emplace_back(NodeEntry{mirror_node, e.index, e.version});
      }
    }
    if (!patten_match) {
      mirror_map[node.get()] = std::move(new_node);
    }
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
  }

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

Graph GraphFusionConvBN(Graph &src) {
  std::unordered_map<Node*, NodePtr> mirror_map;
  bool patten_match;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    // for one node, create a new node as mirror node, and then go through 
    // the node's each input node, if the input node and current node doesn't
    // form a fusion pattern, then add the input node's mirror node to this
    // node's input, otherwise
    NodePtr new_node = Node::Create();
    *new_node = *node;
    new_node->inputs.clear();
    patten_match = false;
    for (const auto& e : node->inputs) {
      NodePtr mirror_node = mirror_map.at(e.node.get());
      if (e.node->op() != nullptr && e.node->op()->name == "Convolution"
             && node->op() != nullptr && node->op()->name == "BatchNorm") {
        // if matched, set current batchnorm mirrow node to convolution so 
        // batchnorm will be skipped
        mirror_map[node.get()] = mirror_node;
        // assume only one match for all inputs
        patten_match = true;
        mirror_node->attrs.dict["no_bias"] = "False";
        mirror_node->op()->attr_parser(&(mirror_node->attrs));
        bool find_bias = false;
        std::string bias_name;
        for (auto& conv_input_nodeEntry : mirror_node->inputs) {
          if (conv_input_nodeEntry.node->attrs.name.find("weight") != std::string::npos) {
            std::string& weight_name = conv_input_nodeEntry.node->attrs.name;
            bias_name = weight_name;
            bias_name.replace(bias_name.find("weight"),strlen("weight"), "bias");
            weight_name.insert(0, "convBNReluPara_");
          }
          if (conv_input_nodeEntry.node->attrs.name.find("bias") != std::string::npos) {
            find_bias = true;
          }
        }
        
        if (!find_bias)
        {
          NodePtr conv_bias_node = CreateNode("nullptr", bias_name);
          NodeEntry conv_bias_entry = NodeEntry{ conv_bias_node, 0, 0 };
          mirror_node->inputs.emplace_back(conv_bias_entry);
        }
        break;
      } else {
        NodeEntry mirror_entry = NodeEntry{
          mirror_node, e.index, e.version};
        new_node->inputs.emplace_back(NodeEntry{mirror_node, e.index, e.version});
      }
    }
    if (!patten_match) {
      mirror_map[node.get()] = std::move(new_node);
    }
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
  }

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

Graph GraphFusionSparsity(Graph &src) {
  std::unordered_map<Node*, NodePtr> mirror_map;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    NodePtr new_node = Node::Create();
    *new_node = *node;
    new_node->inputs.clear();
    // find the convolution node which need to adjust stride to (1, 1)
    if (node->op() != nullptr && node->op()->name == "Convolution" &&
        node->attrs.dict["stride"] == "(2, 2)" && node->attrs.dict["kernel"] == "(1, 1)") {
      new_node->attrs.dict["stride"] = "(1, 1)";
      new_node->op()->attr_parser(&(new_node->attrs));
      // from the convolution node, go through input iteratively to find chain
      // relu --> elemwise_add --> conv --> conv, and adjust last convolution's
      // stride to (2, 2), then find the elemwise_add non-conv input path to
      // insert a pooling node with stride (2, 2) between elemwise_add and its
      // non-conv input node
      for (const auto& e : node->inputs) {
        NodePtr mirror_e = mirror_map.at(e.node.get());
        if (mirror_e->op() != nullptr && mirror_e->op()->name == "Activation") {
          for (auto& relu_in : mirror_e->inputs) {
            if (relu_in.node->op() != nullptr && relu_in.node->op()->name == "elemwise_add" 
                 && !relu_in.node->attrs.dict.count("updated")) {
              for (size_t i = 0; i < relu_in.node->inputs.size(); i++) {
                auto sum_in = relu_in.node->inputs[i];
                if (sum_in.node->op() != nullptr && sum_in.node->op()->name == "Convolution") { 
                  for (auto& conv_in : sum_in.node->inputs) {
                    if (conv_in.node->op() != nullptr && conv_in.node->op()->name == "Convolution") { 
                      conv_in.node->attrs.dict["stride"] = "(2, 2)";
                      conv_in.node->op()->attr_parser(&(conv_in.node->attrs));
                    }
                  }
                } else {
                  // append sum's non-conv input node to new pooling's input
                  NodePtr pooling_node = CreateNode("Pooling", e.node->attrs.name + "_prior_pool");
                  pooling_node->inputs.emplace_back(sum_in);
                  pooling_node->attrs.dict["stride"] = "(2, 2)"; 
                  pooling_node->attrs.dict["kernel"] = "(1, 1)"; 
                  pooling_node->attrs.dict["pad"] = "(0, 0)"; 
                  pooling_node->attrs.dict["pool_type"] = "max";
                  pooling_node->op()->attr_parser(&(pooling_node->attrs));

                  // modify sum's non-conv input node to new pooling node
                  relu_in.node->inputs[i] = NodeEntry{pooling_node, 0, 0};
                }
              }
              // set "updated" so the next convolution after elemwise will only
              // update its own stride and no need to execute above logic again
              relu_in.node->attrs.dict["updated"] = "True";
            }
          }
        }
        new_node->inputs.emplace_back(NodeEntry{mirror_e, e.index, e.version});
      }
    } else {
      for (const auto& e : node->inputs) {
        NodePtr mirror_e = mirror_map.at(e.node.get());
        new_node->inputs.emplace_back(NodeEntry{mirror_e, e.index, e.version});
      }
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
  }

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

Graph GraphFusionAddSumFlag(Graph &src) {
  std::unordered_map<Node*, NodePtr> mirror_map;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    // for one node, create a new node as mirror node, and then go through 
    // the node's each input node, if the input node and current node doesn't
    // form a fusion pattern, then add the input node's mirror node to this
    // node's input, otherwise
    NodePtr new_node = Node::Create();
    *new_node = *node;
    new_node->inputs.clear();
    for (const auto& e : node->inputs) {
      NodePtr mirror_node = mirror_map.at(e.node.get());
      if (e.node->op() != nullptr && e.node->op()->name == "Convolution"
             && e.node->attrs.dict["with_sum"] == "True"
             && e.node->attrs.dict["with_postsum_relu"] == "True"
             && node->op() != nullptr && node->op()->name == "Convolution") {
	    new_node->attrs.dict["with_convsumrelu_in"] = "True";
        new_node->op()->attr_parser(&(new_node->attrs));
      } else if (e.node->op() != nullptr && e.node->op()->name == "Pooling"
          && node->op() != nullptr && node->op()->name == "Convolution") {
        for (const auto& ee : e.node->inputs) {
          if (ee.node->op() != nullptr && ee.node->op()->name == "Convolution"
               && ee.node->attrs.dict["with_sum"] == "True"
               && ee.node->attrs.dict["with_postsum_relu"] == "True") {
	        new_node->attrs.dict["with_convsumrelu_in"] = "True";
            new_node->op()->attr_parser(&(new_node->attrs));
          }
        }
      }
      NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
      new_node->inputs.emplace_back(NodeEntry{mirror_node, e.index, e.version});
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
  }

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

Graph FuseGraph(Graph &&src) {
  Graph fused_graph = GraphFusionConvBN(src);
  fused_graph = GraphFusionConvRelu(fused_graph);
  //fused_graph = GraphFusionSparsity(fused_graph);
  fused_graph = GraphFusionConvSum(fused_graph);
  fused_graph = GraphFusionConvRelu(fused_graph);
  fused_graph = GraphFusionAddSumFlag(fused_graph);
  return fused_graph;
}

Graph SetCalibTableToQuantizedGraph(Graph&& g) {
  static const auto& flist_outputs =
    nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  static const auto& need_requantize_map =
    nnvm::Op::GetAttr<mxnet::FNeedRequantize>("FNeedRequantize");
  const auto& calib_table =
    g.GetAttr<std::unordered_map<std::string, std::pair<float, float>>>("calib_table");
  auto disable_requantize = g.GetAttr<bool>("disable_requantize");

  DFSVisit(g.outputs, [&](const NodePtr& node) {
    bool found = false;
    NodePtr quantized_op_node;
    // If requantize is not disabled, find requantize OP and
    // the thresholds from the calibration table with the key equal
    // to the requantize OP's input node name, e.g. a quantized_conv node.
    if (!disable_requantize &&
        node->op() != nullptr && node->op()->name == "_contrib_requantize") {
      quantized_op_node = node->inputs[0].node;
      CHECK(quantized_op_node->op() != nullptr) << quantized_op_node->attrs.name
                                                << " must be an quantized op node";
      CHECK(need_requantize_map.count(quantized_op_node->op()) > 0
          && need_requantize_map[quantized_op_node->op()](quantized_op_node->attrs))
          << quantized_op_node->attrs.name << " op must register FNeedRequantize attr"
                                              " and the attr func should return true";
      found = true;
    // If requantize is disabled, find OPs that needed requantize and
    // the thresholds from the calibration table with the key equal
    // to the found OP's name, e.g. a quantized_conv node.
    } else if (disable_requantize && node->op() != nullptr &&
            need_requantize_map.count(node->op()) > 0 &&
            need_requantize_map[node->op()](node->attrs)) {
      quantized_op_node = node;
      found = true;
    }
    if (!found) {
      return;
    }
    std::string out_data_name = quantized_op_node->attrs.name + "_";
    auto list_output_names_func = flist_outputs.get(quantized_op_node->op(), nullptr);
    // Here it's assumed that the quantized_op node only produces three outputs:
    // out_data, min_range, and max_range. So we want to get the pre-calculated min_calib_range
    // and max_calib_range from the calibration table for out_data. Here we create the output
    // data name same as its constructed in GraphExecutor::ExecuteMonCallback.
    if (list_output_names_func != nullptr) {
      std::vector<std::string> names = list_output_names_func(quantized_op_node->attrs);
      CHECK_EQ(names.size(), 3U) << "ListOutputNames is expected to return three string for"
                                    " quantized operators";
      out_data_name += names[0];
    } else {
      out_data_name += "0";
    }
    const auto calib_table_iter = calib_table.find(out_data_name);
    if (calib_table_iter != calib_table.end()) {
      node->attrs.dict["min_calib_range"] = std::to_string(calib_table_iter->second.first);
      node->attrs.dict["max_calib_range"] = std::to_string(calib_table_iter->second.second);
      node->op()->attr_parser(&(node->attrs));
    }
  });
  return g;
}

NNVM_REGISTER_PASS(FuseGraph)
.describe("")
.set_body(FuseGraph)
.set_change_graph(true);

NNVM_REGISTER_PASS(QuantizeGraph)
.describe("")
.set_body(QuantizeGraph)
.set_change_graph(true);

NNVM_REGISTER_PASS(SetCalibTableToQuantizedGraph)
.describe("")
.set_body(SetCalibTableToQuantizedGraph)
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
