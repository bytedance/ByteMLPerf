#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;

    unsigned int _p1;
};
struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_X;
    p2 _p1;
    void *ptr_G;
    p2 _p2;
    void *ptr_XC;
    p2 _p3;
    void *ptr_D;
    p2 _p4;
    void *ptr_XQ;
    p2 _p5;
    void *ptr_GQ;
    p2 _p6;
    void *ptr_DQ;
    p2 _p7;
    void *ptr_SMQ;
    p2 _p8;
    void *ptr_STP;
    p2 _p9;
    void *ptr_SW;
    p2 _p10;
    void *ptr_SEP;
    p2 _p11;
    unsigned int dim;
    p3 _p12;
    unsigned int hidden_dim;
    p3 _p13;
    unsigned int token_cnt;
    p3 _p14;
    unsigned int eprt_cnt;
    p3 _p15;
    unsigned int Xs;
    p3 _p16;
    unsigned int GUs;
    p3 _p17;
    unsigned int Ds;
    p3 _p18;
    unsigned int Os;
    p3 _p19;
    unsigned int eGUs;
    p3 _p20;
    unsigned int eDs;
    p3 _p21;
    unsigned int eGUQs;
    p3 _p22;
    unsigned int eDQs;
    p3 _p23;
    unsigned int eSMQs;
    p3 _p24;
};

#define HIP_CALL(call)                                                 \
    do                                                                 \
    {                                                                  \
        hipError_t err = call;                                         \
        if (err != hipSuccess)                                         \
        {                                                              \
            printf("[hiperror](%d) fail to call %s", (int)err, #call); \
            exit(0);                                                   \
        }                                                              \
    } while (0)

class FMoeKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    FMoeKernel(const char *name)
    {
        HIP_CALL(hipModuleLoad(&module, FMOE_HSACO));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    };

    void launch_kernel(torch::Tensor &out,                   // [token_cnt, dim]
                       torch::Tensor &input,                 // [token_cnt, dim] M,K
                       torch::Tensor &gate,                  // [expert, hidden_dim, dim] N,K
                       torch::Tensor &down,                  // [expert, hidden_dim, dim]
                       torch::Tensor &sorted_token_ids,      // [max_num_tokens_padded]
                       torch::Tensor &sorted_weight_buf,     // [max_num_tokens_padded]
                       torch::Tensor &sorted_expert_ids,     // [max_num_m_blocks]
                       torch::Tensor &num_tokens_post_padded // [1]
    )
    {
        unsigned int sub_GU = 512;
        int token_cnt = input.size(0);
        int dim = input.size(1);
        int sub_X_cnt = sorted_expert_ids.size(0);
        int eprt = gate.size(0);
        int hidden_dim = gate.size(1);

        int stride_X = dim * sizeof(float) / 2;
        int stride_GU = dim * sizeof(float) / 2;
        int stride_D = hidden_dim * sizeof(float) / 2;
        int stride_expert_GU = stride_GU * hidden_dim;
        int stride_expert_D = stride_D * dim;
        int stride_expert_GUDQN = hidden_dim * sizeof(float);
        int stride_expert_DDQN = dim * sizeof(float);
        int stride_expert_SMTDQN = hidden_dim * sizeof(float);
        int stride_O = dim * sizeof(float) / 2;

        KernelArgs args;
        size_t arg_size = sizeof(args);
        args.ptr_O = out.data_ptr();
        args.ptr_X = input.data_ptr();
        args.ptr_G = gate.data_ptr();
        args.ptr_XC = num_tokens_post_padded.data_ptr();
        args.ptr_D = down.data_ptr();
        args.ptr_XQ = (void *)NULL;
        args.ptr_GQ = (void *)NULL;
        args.ptr_DQ = (void *)NULL;
        args.ptr_SMQ = (void *)NULL;
        args.ptr_STP = sorted_token_ids.data_ptr();
        args.ptr_SW = sorted_weight_buf.data_ptr();
        args.ptr_SEP = sorted_expert_ids.data_ptr();
        args.dim = dim;
        args.hidden_dim = hidden_dim;
        args.token_cnt = token_cnt;
        args.eprt_cnt = eprt;
        args.Xs = stride_X;
        args.GUs = stride_GU;
        args.Ds = stride_D;
        args.Os = stride_O;
        args.eGUs = stride_expert_GU;
        args.eDs = stride_expert_D;
        args.eGUQs = stride_expert_GUDQN;
        args.eDQs = stride_expert_DDQN;
        args.eSMQs = stride_expert_SMTDQN;

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size, HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = ((hidden_dim + sub_GU - 1) / sub_GU);
        int gdy = sub_X_cnt;
        int gdz = 1;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx, gdy, gdz,
                                       bdx, 1, 1,
                                       0, stream, nullptr, (void **)&config));
    };
};

void fmoe(torch::Tensor &out,                   // [token_cnt, dim]
          torch::Tensor &input,                 // [token_cnt, dim] M,K
          torch::Tensor &gate,                  // [expert, hidden_dim, dim] N,K
          torch::Tensor &down,                  // [expert, hidden_dim, dim]
          torch::Tensor &sorted_token_ids,      // [max_num_tokens_padded]
          torch::Tensor &sorted_weight_buf,     // [max_num_tokens_padded]
          torch::Tensor &sorted_expert_ids,     // [max_num_m_blocks]
          torch::Tensor &num_tokens_post_padded // [1]
)
{
    static FMoeKernel impl("fmoe_kernel_func");
    impl.launch_kernel(out,
                       input,
                       gate,
                       down,
                       sorted_token_ids,
                       sorted_weight_buf,
                       sorted_expert_ids,
                       num_tokens_post_padded);
}
