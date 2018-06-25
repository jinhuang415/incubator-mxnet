# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Quantization module for generating quantized (INT8) models from FP32 models."""

from __future__ import absolute_import

try:
    from scipy import stats
except ImportError:
    stats = None

import ctypes
import logging
import os
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import NDArray
from ..io import DataIter
from ..context import cpu, Context
from ..module import Module

def _fuse_update_params(qsym, params, aux_params):
    """Given a quantized symbol and a dict of params that have not been quantized, generate quantized params.
    Currently only supports quantizing the arg_params with names of `weight` or `bias`, not aux_params.
    If `qsym` contains symbols that are excluded from being quantized, their corresponding params will
    not be quantized, but saved together with quantized params of the symbols that have been quantized.

    Parameters
    ----------
    qsym : Symbol
        Quantized symbol from FP32 symbol.
    params : dict of str->NDArray
    """
    inputs_name = qsym.list_arguments()
    fused_params = {}
    for name in inputs_name:
        if name.startswith('convBNReluPara_'):
            print "para name is ",name            
            original_name = name[len('convBNReluPara_'):]
            print "original_name is ",original_name
            if original_name.endswith('conv0_weight'):
                bn_num = 'bn0'
            elif original_name.endswith('conv1_weight'):
              bn_num = 'bn2'
            elif original_name.endswith('conv2_weight'):
                bn_num = 'bn3'
            stage_unit_name = original_name[:-len('convX_weight')]
            bn_gamma_name = stage_unit_name + bn_num + '_gamma'
            bn_beta_name = stage_unit_name + bn_num + '_beta'
            bn_moving_mean_name = stage_unit_name + bn_num + '_moving_mean'
            bn_moving_var_name = stage_unit_name + bn_num + '_moving_var'
            bn_gamma = params[bn_gamma_name]
            bn_beta = params[bn_beta_name]
            bn_moving_mean = aux_params[bn_moving_mean_name]
            bn_moving_var = aux_params[bn_moving_var_name]

            conv_weight = params[original_name]
            conv_bias_name = original_name[:-len('weight')] + 'bias'
            print "conv_weight_name is ", name
            print "conv_bias_name is ", conv_bias_name
            if(params.has_key(conv_bias_name)):
                conv_bias = params[conv_bias_name]
            else:
                conv_bias = ndarray.zeros_like(bn_gamma)
#            print "conv_bias_type ", type(conv_bias[0]),conv_bias.shape,conv_bias
 #           print type(conv_weight),len(conv_weight),conv_weight.shape,bn_gamma.shape,bn_beta.shape,bn_moving_mean.shape,bn_moving_var.shape
            
  #          print "bn para is ",bn_gamma_name,bn_beta_name,bn_moving_mean_name,bn_moving_var_name

            conv_weight_after_bn = conv_weight
            print type(bn_moving_var[0])," value is ", bn_moving_var[0]
            for i in range (len(conv_weight)):
                conv_weight_after_bn[i,:,:,:] = conv_weight[i,:,:,:]*bn_gamma[i]/NDArray.sqrt(bn_moving_var[i] + 2e-05)
                conv_bias[i] = (conv_bias[i] - bn_moving_mean[i])*bn_gamma[i]/NDArray.sqrt(bn_moving_var[i] + 2e-05) + bn_beta[i]
   #         print "save original_name is ",original_name   
            fused_params[name] = conv_weight_after_bn;
            fused_params[conv_bias_name] = conv_bias
    #        print "bias shape is ",conv_bias.shape
        elif name in params:
            print "else name is ", name
            fused_params[name] = params[name]
#    for k, v in params.items():
#         print "params key are ",k
#    for k, v in aux_params.items():
#         print "aux_params name is ",k 
#         print "aux_params name is ",k ," val are ", v
    return fused_params

def _quantize_params(qsym, params, th_in_dict={}):
    """Given a quantized symbol and a dict of params that have not been quantized,
    generate quantized params. Currently only supports quantizing the arg_params
    with names of `weight` or `bias`, not aux_params. If `qsym` contains symbols
    that are excluded from being quantized, their corresponding params will
    not be quantized, but saved together with quantized params of the symbols that
    have been quantized.

    Parameters
    ----------
    qsym : Symbol
        Quantized symbol from FP32 symbol.
    params : dict of str->NDArray
    th_in_dict : dict of threshold for input calibration
    """
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith(('weight_quantize', 'bias_quantize')):
            original_name = name[:-len('_quantize')]
            param = params[original_name]
            val, vmin, vmax = ndarray.contrib.quantize(data=param,
                                                       min_range=ndarray.min(param),
                                                       max_range=ndarray.max(param),
                                                       out_type='int8')
            print("quantize param {}, min/max {}/{}".format(name, vmin, vmax))
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
    for name in th_in_dict:
        layer_name = name.replace('_data', '')
        quantized_params[layer_name+'_min'] = ndarray.array([th_in_dict[name][0]])
        quantized_params[layer_name+'_max'] = ndarray.array([th_in_dict[name][1]])
    return quantized_params

def _fuse_symbol(sym):
    out = SymbolHandle()
    check_call(_LIB.MXFuseSymbol(sym.handle,
                                 ctypes.byref(out)))
    return Symbol(out)

def _quantize_symbol(sym, excluded_symbols=None, offline_params=None,
                     quantized_dtype='int8', disable_requantize=False,
                     input_calib_layers=None):
    """Given a symbol object representing a neural network of data type FP32,
    quantize it into a INT8 network.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol.
    excluded_symbols : list of symbols
        Nodes in the network that users do not want to replace with a symbol of INT8 data type.
    offline_params : list of strs
        Names of the parameters that users want to quantize offline. It's always recommended to
        quantize parameters offline so that quantizing parameters during the inference can be
        avoided.
    quantized_dtype: str
        The quantized destination type for input data.
    disable_requantize : bool
        Whether disable requantize OP functionality.
    input_calib_layers : list of strs
        Layer names in the network that users want to perform input offline calibration.
    """
    num_excluded_symbols = 0
    excluded_handles = []
    if excluded_symbols is not None:
        assert isinstance(excluded_symbols, list)
        num_excluded_symbols = len(excluded_symbols)
        for s in excluded_symbols:
            excluded_handles.append(s.handle)

    num_offline = 0
    offline = []
    if offline_params is not None:
        num_offline = len(offline_params)
        for k in offline_params:
            offline.append(c_str(k))

    num_input_calib = 0
    input_calib = []
    if input_calib_layers is not None:
        num_input_calib = len(input_calib_layers)
        for k in input_calib_layers:
            input_calib.append(c_str(k))

    out = SymbolHandle()
    check_call(_LIB.MXQuantizeSymbol(sym.handle,
                                     ctypes.byref(out),
                                     mx_uint(num_excluded_symbols),
                                     c_array(SymbolHandle, excluded_handles),
                                     mx_uint(num_offline),
                                     c_array(ctypes.c_char_p, offline),
                                     c_str(quantized_dtype),
                                     ctypes.c_bool(disable_requantize),
                                     mx_uint(num_input_calib),
                                     c_array(ctypes.c_char_p, input_calib)))
    return Symbol(out)


class _LayerStatsCollector(object):
    """Saves layer stats NDArray in dicts with layer names as keys and lists of NDArrays as
    values. The collected NDArrays will be used for calculating the optimal thresholds for
    quantization using KL divergence.
    """
    def __init__(self, include_layer=None, input_calib_layer=None, logger=None):
        self.nd_in_dict = {}
        self.nd_out_dict = {}
        self.include_layer = include_layer
        self.input_calib_layer = input_calib_layer
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting layer stats NDArrays."""
        name = py_str(name)
        input_calib_prev_name = None
        input_calib_name = None
        if name.endswith('_data'):
            # For input calib, the name format passing by graph_executor.cc is
            # "<prev-layer>-<calib-layer>_data", we want to check if calib-layer
            # is within our input calibrated layers and should save the layer
            # input stats with previous layer name as key and finally save into
            # parameter file
            input_calib_prev_name = name.split('-')[1]
            input_calib_name = name.split('-')[0] + '_data'
        if not (self.include_layer is not None and self.include_layer(name)) and \
           not (self.input_calib_layer is not None and input_calib_name is not None
                   and self.input_calib_layer(input_calib_name)):
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False).copyto(cpu())
        if name.endswith('_output'):
            if name in self.nd_out_dict:
                self.nd_out_dict[name].append(arr)
            else:
                self.nd_out_dict[name] = [arr]
        elif name.endswith('_data'):
            if input_calib_prev_name in self.nd_in_dict:
                self.nd_in_dict[input_calib_prev_name].append(arr)
            else:
                self.nd_in_dict[input_calib_prev_name] = [arr]
        if self.logger is not None:
            self.logger.info("Collecting layer %s stats of shape %s" % (name, arr.shape))


class _LayerStatsMinMaxCollector(object):
    """Saves layer stats min and max values in dicts with layer names as keys.
    The collected min and max values will be directly used as thresholds for quantization.
    """
    def __init__(self, include_layer=None, input_calib_layer=None, logger=None):
        self.min_max_in_dict = {}
        self.min_max_out_dict = {}
        self.include_layer = include_layer
        self.input_calib_layer = input_calib_layer
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting min and max values from an NDArray."""
        name = py_str(name)
        input_calib_prev_name = None
        input_calib_name = None
        if name.endswith('_data'):
            # For input calib, the name format passing by graph_executor.cc is
            # "<prev-layer>-<calib-layer>_data", we want to check if calib-layer
            # is within our input calibrated layers and should save the layer
            # input stats with previous layer name as key and finally save into
            # parameter file
            input_calib_prev_name = name.split('-')[1]
            input_calib_name = name.split('-')[0] + '_data'
        if not (self.include_layer is not None and self.include_layer(name)) and \
           not (self.input_calib_layer is not None and input_calib_name is not None
                   and self.input_calib_layer(input_calib_name)):
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False)
        min_range = ndarray.min(arr).asscalar()
        max_range = ndarray.max(arr).asscalar()
        if name.endswith('_output'):
            if name in self.min_max_out_dict:
                cur_min_max = self.min_max_out_dict[name]
                self.min_max_out_dict[name] = (min(cur_min_max[0], min_range),
                                               max(cur_min_max[1], max_range))
            else:
                self.min_max_out_dict[name] = (min_range, max_range)
        elif name.endswith('_data'):
            if input_calib_prev_name in self.min_max_in_dict:
                cur_min_max = self.min_max_in_dict[input_calib_prev_name]
                self.min_max_in_dict[input_calib_prev_name] = \
                                          (min(cur_min_max[0], min_range),
                                           max(cur_min_max[1], max_range))
            else:
                self.min_max_in_dict[input_calib_prev_name] = (min_range, max_range)
            name = input_calib_name
        if self.logger is not None:
            self.logger.info("Collecting layer %s stats min_range=%f, max_range=%f"
                             % (name, min_range, max_range))


def _calibrate_quantized_sym(qsym, th_dict, disable_requantize=False):
    """Given a dictionary containing the thresholds for quantizing the layers,
    set the thresholds into the quantized symbol as the params of requantize operators.
    """
    if th_dict is None or len(th_dict) == 0:
        return qsym
    num_layer_outputs = len(th_dict)
    layer_output_names = []
    min_vals = []
    max_vals = []
    for k, v in th_dict.items():
        layer_output_names.append(k)
        min_vals.append(v[0])
        max_vals.append(v[1])

    calibrated_sym = SymbolHandle()
    check_call(_LIB.MXSetCalibTableToQuantizedSymbol(qsym.handle,
                                                     mx_uint(num_layer_outputs),
                                                     c_str_array(layer_output_names),
                                                     c_array(ctypes.c_float, min_vals),
                                                     c_array(ctypes.c_float, max_vals),
                                                     ctypes.byref(calibrated_sym),
                                                     ctypes.c_bool(disable_requantize)))
    return Symbol(calibrated_sym)


def _collect_layer_statistics(mod, data, collector, max_num_examples=None,
                              enable_input_calib=False, logger=None):
    if not isinstance(data, DataIter):
        raise ValueError('Only supports data as a type of DataIter, while received type %s'
                         % str(type(data)))
    mod._exec_group.execs[0].set_monitor_callback(collector.collect)
    if enable_input_calib:
        mod._exec_group.execs[0].set_input_monitor_callback(collector.collect)
    num_batches = 0
    num_examples = 0
    for batch in data:
        mod.forward(data_batch=batch, is_train=False)
        num_batches += 1
        num_examples += data.batch_size
        if max_num_examples is not None and num_examples >= max_num_examples:
            break
    if logger is not None:
        logger.info("Collected statistics from %d batches with batch_size=%d"
                    % (num_batches, data.batch_size))
    return num_examples


def _collect_layer_stats_min_max(mod, data, include_layer=None,
                                 max_num_examples=None,
                                 input_calib_layer=None, logger=None):
    """Collect min and max values from layer stats and save them in
    a dictionary mapped by layer names.
    """
    collector = _LayerStatsMinMaxCollector(include_layer=include_layer,
                                           input_calib_layer=input_calib_layer,
                                           logger=logger)
    num_examples = _collect_layer_statistics(mod, data, collector,
                                             max_num_examples,
                                             input_calib_layer != None, logger)
    return collector.min_max_in_dict, collector.min_max_out_dict, num_examples


def _collect_layer_stats(mod, data, include_layer=None, max_num_examples=None,
                         input_calib_layer=None, logger=None):
    """Collect layer stats and save them in a dictionary mapped by layer names."""
    collector = _LayerStatsCollector(include_layer=include_layer,
                                     input_calib_layer=input_calib_layer,
                                     logger=logger)
    num_examples = _collect_layer_statistics(mod, data, collector,
                                             max_num_examples,
                                             input_calib_layer != None, logger)
    return collector.nd_in_dict, collector.nd_out_dict, num_examples


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


# pylint: disable=line-too-long
def _get_optimal_threshold(arr, num_bins=8001, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    if isinstance(arr, NDArray):
        arr = arr.asnumpy()
    elif isinstance(arr, list):
        assert len(arr) != 0
        for i, nd in enumerate(arr):
            if isinstance(nd, NDArray):
                arr[i] = nd.asnumpy()
            elif not isinstance(nd, np.ndarray):
                raise TypeError('get_optimal_threshold only supports input type of NDArray,'
                                ' list of np.ndarrays or NDArrays, and np.ndarray,'
                                ' while received type=%s' % (str(type(nd))))
        arr = np.concatenate(arr)
    elif not isinstance(arr, np.ndarray):
        raise TypeError('get_optimal_threshold only supports input type of NDArray,'
                        ' list of NDArrays and np.ndarray,'
                        ' while received type=%s' % (str(type(arr))))
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edeges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2
    assert np.allclose(hist_edeges[zero_bin_idx] + hist_edeges[zero_bin_idx + 1],
                       0, rtol=1e-5, atol=1e-7)

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin
    for i in range(num_quantized_bins // 2,
                   num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edeges[p_bin_idx_stop]
        # sliced_nd_hist is used to generate candidate distribution q
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (sliced_nd_hist != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = p.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(p.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[sliced_nd_hist == 0] = 0
        p = _smooth_distribution(p)
        q = _smooth_distribution(q)
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)
        quantized_bins[:] = 0

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    return min_val, max_val, min_divergence, opt_th
# pylint: enable=line-too-long


def _get_optimal_thresholds(nd_dict, num_bins=8001, num_quantized_bins=255, logger=None):
    """Given a ndarray dict, find the optimal threshold for quantizing each value of the key."""
    if stats is None:
        raise ImportError('scipy.stats is required for running entropy mode of calculating'
                          ' the optimal thresholds for quantizing FP32 ndarrays into int8.'
                          ' Please check if the scipy python bindings are installed.')
    assert isinstance(nd_dict, dict)
    if logger is not None:
        logger.info('Calculating optimal thresholds for quantization using KL divergence'
                    ' with num_bins=%d and num_quantized_bins=%d' % (num_bins, num_quantized_bins))
    th_dict = {}
    # copy nd_dict keys since the keys() only returns a view in python3
    layer_names = list(nd_dict.keys())
    for name in layer_names:
        assert name in nd_dict
        min_val, max_val, min_divergence, opt_th =\
            _get_optimal_threshold(nd_dict[name], num_bins=num_bins,
                                   num_quantized_bins=num_quantized_bins)
        del nd_dict[name]  # release the memory of ndarray
        th_dict[name] = (-opt_th, opt_th)
        if logger is not None:
            logger.info('layer=%s, min_val=%f, max_val=%f, min_divergence=%f, optimal_threshold=%f'
                        % (name, min_val, max_val, min_divergence, opt_th))
    return th_dict


def _load_sym(sym, logger=logging):
    """Given a str as a path the symbol .json file or a symbol, returns a Symbol object."""
    if isinstance(sym, str):  # sym is a symbol file path
        cur_path = os.path.dirname(os.path.realpath(__file__))
        symbol_file_path = os.path.join(cur_path, sym)
        logger.info('Loading symbol from file %s' % symbol_file_path)
        return sym_load(symbol_file_path)
    elif isinstance(sym, Symbol):
        return sym
    else:
        raise ValueError('_load_sym only accepts Symbol or path to the symbol file,'
                         ' while received type %s' % str(type(sym)))


def _load_params(params, logger=logging):
    """Given a str as a path to the .params file or a pair of params,
    returns two dictionaries representing arg_params and aux_params.
    """
    if isinstance(params, str):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        param_file_path = os.path.join(cur_path, params)
        logger.info('Loading params from file %s' % param_file_path)
        save_dict = nd_load(param_file_path)
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        return arg_params, aux_params
    elif isinstance(params, (tuple, list)) and len(params) == 2:
        return params[0], params[1]
    else:
        raise ValueError('Unsupported params provided. Must be either a path to the param file or'
                         ' a pair of dictionaries representing arg_params and aux_params')

def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    ndarray.save(fname, save_dict)

def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)

def quantize_model(sym, arg_params, aux_params,
                   data_names=('data',), label_names=('softmax_label',),
                   ctx=cpu(), excluded_sym_names=None, calib_mode='entropy',
                   calib_data=None, num_calib_examples=None, calib_layer=None,
                   quantized_dtype='int8', disable_requantize=False,
                   input_calib_layer=None, logger=logging):
    """User-level API for generating a quantized model from a FP32 model w/ or w/o calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.
    The quantization implementation adopts the TensorFlow's approach:
    https://www.tensorflow.org/performance/quantization.
    The calibration implementation borrows the idea of Nvidia's 8-bit Inference with TensorRT:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    and adapts the method to MXNet.

    Parameters
    ----------
    sym : str or Symbol
        Defines the structure of a neural network for FP32 data types.
    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    data_names : a list of strs
        Data names required for creating a Module object to run forward propagation on the
        calibration dataset.
    label_names : a list of strs
        Label names required for creating a Module object to run forward propagation on the
        calibration dataset.
    ctx : Context
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single context.
    excluded_sym_names : list of strings
        A list of strings representing the names of the symbols that users want to excluding
        from being quantized.
    calib_mode : str
        If calib_mode='none', no calibration will be used and the thresholds for
        requantization after the corresponding layers will be calculated at runtime by
        calling min and max operators. The quantized models generated in this
        mode are normally 10-20% slower than those with calibrations during inference.
        If calib_mode='naive', the min and max values of the layer outputs from a calibration
        dataset will be directly taken as the thresholds for quantization.
        If calib_mode='entropy' (default mode), the thresholds for quantization will be
        derived such that the KL divergence between the distributions of FP32 layer outputs and
        quantized layer outputs is minimized based upon the calibration dataset.
    calib_data : DataIter
        A data iterator initialized by the calibration dataset.
    num_calib_examples : int or None
        The maximum number of examples that user would like to use for calibration. If not provided,
        the whole calibration dataset will be used.
    calib_layer : function
        Given a layer's output name in string, return True or False for deciding whether to
        calibrate this layer. If yes, the statistics of the layer's output will be collected;
        otherwise, no information of the layer's output will be collected. If not provided,
        all the layers' outputs that need requantization will be collected.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8'
        and 'uint8', default value is 'int8'.
    disable_requantize : bool
        Whether disable requantize OP during quantization. If disabled, the related
        quantized OP needed requantize will output int8 directly and hence requantize
        OP is not needed during symbol quantization
    input_calib_layer: function
        Given a layer's input name in string, return True or False for deciding whether to
        calibrate the input for this layer. If yes, the statistics of the layer's input
        will be collected; otherwise, no information of the layer's input will be collected.
        If not provided, no layers' inputs will be collected.
    logger : Object
        A logging object for printing information during the process of quantization.

    Returns
    -------
    tuple
        A tuple of quantized symbol, quantized arg_params, and aux_params.
    -------
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
    if not isinstance(excluded_sym_names, list):
        raise ValueError('excluded_sym_names must be a list of strings representing'
                         ' the names of the symbols that will not be quantized,'
                         ' while received type %s' % str(type(excluded_sym_names)))
    fsym = _fuse_symbol(sym)
    arg_params = _fuse_update_params(fsym, arg_params, aux_params)
    #save_symbol("jin-debug.json", fsym, logger)
    #save_params("jin-debug.params", arg_params, aux_params, logger)
    #import sys
    #sys.exit()

    excluded_syms = []
    if excluded_sym_names is not None:
        for sym_name in excluded_sym_names:
            nodes = fsym.get_internals()
            idx = nodes.list_outputs().index(sym_name + '_output')
            excluded_syms.append(nodes[idx])
    logger.info('Quantizing symbol')

    input_calib_layers = []
    if input_calib_layer is not None:
        # we strip '_output' from list_outputs() to get the layer name we want
        # to perform input calibration and pass to quantize graph logic
        for output in fsym.get_internals().list_outputs():
            if (output.endswith('_output') and input_calib_layer is not None
                    and input_calib_layer(output.replace('_output', '_data'))):
                input_calib_layers.append(output.replace('_output', ''))

    if quantized_dtype != 'int8' and quantized_dtype != 'uint8':
        raise ValueError('unknown quantized_dtype %s received,'
                         ' expected `int8` or `uint8`' % quantized_dtype)
    qsym = _quantize_symbol(fsym, excluded_symbols=excluded_syms,
                            offline_params=list(arg_params.keys()),
                            quantized_dtype=quantized_dtype,
                            disable_requantize=disable_requantize,
                            input_calib_layers=input_calib_layers)
    

    if calib_mode is not None and calib_mode != 'none':
        if not isinstance(ctx, Context):
            raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
        if calib_data is None:
            raise ValueError('calib_data must be provided when calib_mode=%s' % calib_mode)
        if not isinstance(calib_data, DataIter):
            raise ValueError('calib_data must be of DataIter type when calib_mode=%s,'
                             ' while received type %s' % (calib_mode, str(type(calib_data))))
        if calib_layer is None:
            calib_layer = lambda name: name.endswith('_output')

        mod = Module(symbol=fsym, data_names=data_names, label_names=label_names, context=ctx)
        if len(calib_data.provide_label) > 0:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data,
                     label_shapes=calib_data.provide_label)
        else:
            mod.bind(for_training=False, data_shapes=calib_data.provide_data)
        mod.set_params(arg_params, aux_params)
        if calib_mode == 'entropy':
            nd_in_dict, nd_out_dict, num_examples = \
                _collect_layer_stats(mod, calib_data, include_layer=calib_layer,
                                     max_num_examples=num_calib_examples,
                                     input_calib_layer=input_calib_layer,
                                     logger=logger)
            logger.info('Collected layer stats from FP32 model using %d examples' % num_examples)
            logger.info('Calculating optimal thresholds for quantization')
            th_in_dict = _get_optimal_thresholds(nd_in_dict, logger=logger)
            th_out_dict = _get_optimal_thresholds(nd_out_dict, logger=logger)
        elif calib_mode == 'naive':
            th_in_dict, th_out_dict, num_examples = \
                _collect_layer_stats_min_max(mod, calib_data,
                                             include_layer=calib_layer,
                                             max_num_examples=num_calib_examples,
                                             input_calib_layer=input_calib_layer,
                                             logger=logger)
            logger.info('Collected layer stats min/max values from FP32 model using %d examples'
                        % num_examples)
        else:
            raise ValueError('unknown calibration mode %s received,'
                             ' expected `none`, `naive`, or `entropy`' % calib_mode)
        logger.info('Calibrating quantized symbol')
        qsym = _calibrate_quantized_sym(qsym, th_out_dict, disable_requantize)
    else:
        th_in_dict = {}

    logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, th_in_dict)

    return qsym, qarg_params, aux_params
