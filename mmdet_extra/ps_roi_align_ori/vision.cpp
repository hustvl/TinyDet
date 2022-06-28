// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "PSROIAlign.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("ps_roi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
  m.def("ps_roi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");

}
