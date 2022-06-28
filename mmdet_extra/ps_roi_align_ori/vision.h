// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>

////
#include <THC/THC.h>







int PSROIAlignForwardLaucher(at::Tensor bottom_data,
                             at::Tensor bottom_rois,
                             at::Tensor top_data,
                             at::Tensor argmax_data,
                             float spatial_scale,
                             int group_size,
                             int sampling_ratio,
                             cudaStream_t stream);


int PSROIAlignBackwardLaucher(at::Tensor top_diff,
                              at::Tensor argmax_data,
                              at::Tensor bottom_rois,
                              at::Tensor bottom_diff,
                              float spatial_scale,
                              int group_size,
                              int sampling_ratio,
                              cudaStream_t stream);