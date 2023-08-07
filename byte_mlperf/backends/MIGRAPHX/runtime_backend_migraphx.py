import os
import json
import logging
import copy

import torch
# import lego

import tensorflow as tf

import onnx
# import onnx_tensorrt.backend as onnx_convert

tf.get_logger().setLevel('ERROR')

import numpy as np
import time
from byte_mlperf.backends import runtime_backend

log = logging.getLogger("RuntimeBackendMIGRAPHX")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "INT32": torch.int32,
    "LONG": torch.long,
    "BOOL": torch.bool
}

tf_dtype_map = {
    "FLOAT32": tf.float32,
    "FLOAT16": tf.float16,
    "INT32": tf.int32,
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool,
}

class RuntimeBackendMIGRAPHX(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendMIGRAPHX, self).__init__()
        self.hardware_type = 'MIGRAPHX'
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.batch_size = -1

    def predict(self, feeds):
        if not self.model_runtimes:
            self._load()
        if(self.old_batch_size != self.batch_size):
            self.load()

        results = {}
        # Since in MIGRAPHX, for tensorflow, we just convert it to onnx. The logic is the same for them 
        if self.framework == "Tensorflow" or self.framework == "Onnx":
            params = {}
            key_id = 0
            for key in feeds.keys():
                ###new_key = key.replace(":", "_")
                # Some of the axles of the input needs to be converted before passing into the model
                if( ( 'layout' in self.model_info ) and ( self.model_info['layout'] == 'NHWC' ) and ( feeds[ key ].shape[3] != 3 ) ):
                    params[ key ] = np.ascontiguousarray(np.transpose( feeds[ key ] , axes=[0, 2, 3, 1]))
                elif( isinstance( feeds[key] , list ) ):
                    params[ key ] = np.array( feeds[ key ] , dtype = INPUT_TYPE[ self.input_type[key_id] ] )
                else:
                    params[ key ] = feeds[ key ]
                key_id += 1


            results = {}
            for model_runtime in self.model_runtimes:
                _results = model_runtime.run( params )
                # For videobert, the shape of the output is quite different from that of the other models
                if( 'videobert' in self.configs['model'] ):
                    if( isinstance( _results , dict ) ):
                        assert( ( len( _results.keys() ) == 1 ) and ( 'output' in _results.keys() ) )
                        assert( ( len( _results[ 'output' ][0] ) % self.batch_size ) == 0 )
                        results = np.array( _results[ 'output' ] ).reshape( self.batch_size , len( _results[ 'output' ] ) * len( _results[ 'output' ][0] ) // self.batch_size )
                    else:
                        results = np.array( _results[0] )
                        results = results.reshape( self.batch_size , results.shape[0] * results.shape[1] // self.batch_size )
                    results = [ results ]
                elif('resnet50-tf-fp16' in self.configs['model'] ):
                    assert( len(_results) == 2 )
                    results[ self.outputs[0] ] = np.array( _results[1].tolist() ).reshape( _results[1].get_shape().lens() )
                else:
                    if( len( self.outputs ) == 1 ):
                        results[ self.outputs[0] ] = np.array( _results[0].tolist() ).reshape( _results[0].get_shape().lens() )
                    else:
                        for i in range( len( self.outputs ) ):
                            results[ self.outputs[i] ] = [ np.array( _results[i] ).reshape( _results[i].get_shape().lens() ) ]
            assert len(results) != 0

        elif self.framework == "Pytorch":
            # currently this path of code has not been implemented yet
            raise NotImplementedError("MIGraphX backend for models of PyTorch framework has not been implemented yet.")


        else:
            for model_runtime in self.model_runtimes:
                results = model_runtime.run(feeds)
        # for videobert, the output has two components, but for others it only has one
        if( 'videobert' in self.configs['model'] ):
            return results, {}
        else:
            return results

    def benchmark(self, dataloader):
        batch_sizes = self.workload['batch_sizes']
        if( self.batch_size is not None ):
            batch_size = self.batch_size
        else:
            batch_size = self.workload[ 'batch_sizes' ][ 0 ]
        iterations = self.workload['iterations']
        reports = []
        if True:
            times_range = []
            report = {}
            report['BS'] = batch_size
            test_data = dataloader.get_fake_samples(
                batch_size, self.configs['segments'][0]['input_tensor_map'],
                self.configs['input_type'])

            for _ in range(30):
                self.predict(test_data)

            for _ in range(iterations):
                start_time = time.time()
                self.predict(test_data)
                end_time = time.time()
                times_range.append(end_time - start_time)

            times_range.sort()
            tail_latency = round(
                times_range[int(len(times_range) * 0.99)] * 1000, 2)
            avg_latency = round(sum(times_range) / iterations * 1000, 2)
            qps = int(1000.0 * batch_size / avg_latency)

            report['QPS'] = qps
            report['AVG Latency'] = avg_latency
            report['P99 Latency'] = tail_latency
            reports.append(report)

        return reports

    def get_loaded_batch_size(self):
        return self.batch_size

    def _load(self):
        for i, segment in enumerate(self.configs['segments']):
            # there is no input/output meta data i the graph so it need to come from config.
            if not segment['input_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs inputs")
            if not segment['output_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs outputs")

            self.input_shapes = segment['input_tensor_map']
            self.outputs = segment['output_tensor_map'].split(",")

            if( self.framework == "Tensorflow" or self.framework == "Onnx"):
                onnx_file_path = segment['compiled_model'][0]['compiled_obj']
                import migraphx
                onnx_file_path_for_batch_size = onnx_file_path.rsplit("/",2)
                onnx_file_path=os.path.join(onnx_file_path_for_batch_size[0],str(self.batch_size),onnx_file_path_for_batch_size[-1])
                model=migraphx.load(onnx_file_path,format='msgpack')
            elif self.framework == "Pytorch":
                raise NotImplementedError("MIGraphX backend for models of PyTorch framework has not been implemented yet.")
            else:
                # original_model = onnx.load(segment['compiled_model'][0]['compiled_obj'])
                # model =  onnx_convert.prepare(original_model, device='CUDA:1')
                pass

            self.model_runtimes.append(model)

    def _get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if input_type:
            i = 0
            for key, val in shape.items():
                if key != "text":
                    val = [val[0] * batch_size] + val[1:]
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                else:
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")

    def load(self, batch_size = 1) -> None:
        self.batch_size = batch_size
        self.model_runtimes = []
        self.input_type = self.configs['input_type']
        self.framework = self.configs['framework']

        self.model_name = self.configs['model']
        
        self._load()
        self.old_batch_size = batch_size
