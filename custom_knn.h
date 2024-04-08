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

#ifndef COSTOMKNN_H_INCLUDED
#define COSTOMKNN_H_INCLUDED

class CustomKNN
{
public:
	static void custom_knn(int P, float3* points, float* topKDist, int* topKIndices);
};

#endif