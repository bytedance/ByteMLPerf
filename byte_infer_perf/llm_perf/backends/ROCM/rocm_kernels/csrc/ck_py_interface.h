#pragma once

#include <torch/extension.h>

// ck kernel
void layernorm2d(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight, torch::Tensor &bias, double epsilon);
void layernorm2d_with_add(torch::Tensor &out, torch::Tensor &input, torch::Tensor &residual_in, torch::Tensor &residual_out, torch::Tensor &weight, torch::Tensor &bias, double epsilon);
void layernorm2d_with_smoothquant(torch::Tensor &out,    // [m ,n]
                                  torch::Tensor &input,  // [m ,n]
                                  torch::Tensor &xscale, // [1 ,n]
                                  torch::Tensor &yscale, // [m ,1]
                                  torch::Tensor &weight, // [1 ,n]
                                  torch::Tensor &bias,   // [1 ,n]
                                  double epsilon);
void layernorm2d_with_add_smoothquant(torch::Tensor &out,          // [m ,n]
                                      torch::Tensor &input,        // [m ,n]
                                      torch::Tensor &residual_in,  // [m ,n]
                                      torch::Tensor &residual_out, // [m ,n]
                                      torch::Tensor &xscale,       // [1 ,n]
                                      torch::Tensor &yscale,       // [m ,1]
                                      torch::Tensor &weight,       // [1 ,n]
                                      torch::Tensor &bias,         // [1 ,n]
                                      double epsilon);
void layernorm2d_with_dynamicquant(torch::Tensor &out,    // [m ,n]
                                   torch::Tensor &input,  // [m ,n]
                                   torch::Tensor &yscale, // [m ,1]
                                   torch::Tensor &weight, // [1 ,n]
                                   torch::Tensor &bias,   // [1 ,n]
                                   double epsilon);
void layernorm2d_with_add_dynamicquant(torch::Tensor &out,          // [m ,n]
                                       torch::Tensor &input,        // [m ,n]
                                       torch::Tensor &residual_in,  // [m ,n]
                                       torch::Tensor &residual_out, // [m ,n]
                                       torch::Tensor &yscale,       // [m ,1]
                                       torch::Tensor &weight,       // [1 ,n]
                                       torch::Tensor &bias,         // [1 ,n]
                                       double epsilon);
void smoothquant_fwd(torch::Tensor &out,      // [m ,n]
                     torch::Tensor &input,    // [m ,n]
                     torch::Tensor &x_scale,  // [1 ,n]
                     torch::Tensor &y_scale); // [m ,1]