/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "custom_knn.h"

std::tuple<torch::Tensor, torch::Tensor>
topKdistCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor topKDist = torch::full({P*8}, 0.0, float_opts);
  torch::Tensor topKIndices = torch::full({P*8}, 0.0, int_opts);
  
  CustomKNN::custom_knn(P, (float3*)points.contiguous().data<float>(), 
  topKDist.contiguous().data<float>(), topKIndices.contiguous().data<int>());

  return std::make_tuple(topKDist, topKIndices);
}