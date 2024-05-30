import tvm
from tvm import relay
from tvm.relay import Expr
from tvm.relay.dataflow_pattern import wildcard, is_constant, is_op, DFPatternCallback, rewrite
from tvm.relay.expr_functor import ExprMutator

#TODO(chen.chen): we should move this class to igie repo
class MainFunctionParamsRewriter(ExprMutator):
    def __init__(self, target_input_dict, preprocess_rewriter=None):        
        self.target_input = target_input_dict
        self.preprocess_rewriter = preprocess_rewriter
        self.target_input_name_list = list(self.target_input.keys())
        
        super().__init__()
        
    def visit_function(self, fn):
        params = [self.visit(i) for i in fn.params]
        body  = self.visit(fn.body)
        
        original_input_name_list = [param.name_hint for param in params]
        assert len(set(self.target_input_name_list) - set(original_input_name_list)) == 0, f"invalid target_input_name: {set(self.target_input_name_list) - set(original_input_name_list)}"
        
        new_params = []
        bind = {}
        for param in params:
            old_param = param
            name = param.name_hint
            
            new_param = old_param
            if name in self.target_input:
                shape = self.target_input[name][0]
                if len(self.target_input[name]) == 2:
                    dtype = self.target_input[name][1]
                else:
                    dtype = old_param.type_annotation.dtype
                new_param = relay.var(name_hint=name, shape=shape, dtype=dtype)

            new_params.append(new_param)
            bind[old_param] = new_param
            
        new_body = relay.expr.bind(body, bind)
        
        new_function = relay.Function(params=new_params,
                                      body=new_body,
                                      ret_type=None,
                                      type_params=fn.type_params,
                                      attrs=fn.attrs)
        return new_function            
               
    def __call__(self, mod):
        if self.preprocess_rewriter:
            mod["main"] = rewrite(self.preprocess_rewriter, mod["main"])
        mod["main"] = self.visit(mod["main"])
        return mod
    
    
# TODO(chen.chen) this function is designeg for bert model, but it doesn't work now
# the reason is that, position_embedding is fixed when mod is generated from onnx
# e.g. the meta[relay.Constant][51] is fixed as 256
# even if we rewrite the seq_len to 384, the InferType will failed for %9 = add(%8, meta[relay.Constant][51] /* ty=Tensor[(1, 256, 768), float32] */)

# def @main(%input_ids: Tensor[(8, 256), int64], %attention_mask: Tensor[(8, 256), int64], %token_type_ids: Tensor[(8, 256), int64]) -> (Tensor[(8, 256), float32], Tensor[(8, 256), float32]) {
#   %0 = less(%input_ids, 0 /* ty=int64 */) /* ty=Tensor[(8, 256), bool] */;
#   %1 = add(%input_ids, 30522 /* ty=int64 */) /* ty=Tensor[(8, 256), int64] */;
#   %2 = where(%0, %1, %input_ids) /* ty=Tensor[(8, 256), int64] */;
#   %3 = less(%token_type_ids, 0 /* ty=int64 */) /* ty=Tensor[(8, 256), bool] */;
#   %4 = add(%token_type_ids, 2 /* ty=int64 */) /* ty=Tensor[(8, 256), int64] */;
#   %5 = where(%3, %4, %token_type_ids) /* ty=Tensor[(8, 256), int64] */;
#   %6 = take(meta[relay.Constant][49] /* ty=Tensor[(30522, 768), float32] */, %2, axis=0) /* ty=Tensor[(8, 256, 768), float32] */;
#   %7 = take(meta[relay.Constant][50] /* ty=Tensor[(2, 768), float32] */, %5, axis=0) /* ty=Tensor[(8, 256, 768), float32] */;
#   %8 = add(%6, %7) /* ty=Tensor[(8, 256, 768), float32] */;
#   %9 = add(%8, meta[relay.Constant][51] /* ty=Tensor[(1, 256, 768), float32] */) /* ty=Tensor[(8, 256, 768), float32] */;
  
  
def modify_seq_len_for_nlp(mod, input_dict, target_seq_len):
    target_input_dict = {}
    for name, shape in input_dict.items():
        target_input_dict[name] = [(shape[0], target_seq_len)]
    mod = relay.transform.InferType()(mod)
    mod = MainFunctionParamsRewriter(target_input_dict=target_input_dict)(mod)
    return mod