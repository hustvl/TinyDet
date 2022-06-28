#!/usr/bin/env sh


# calculate flops of TinyDet-M
python tools/get_flops.py configs/tinydet_M.py --shape 320

# calculate flops of TinyDet-S
# python tools/get_flops.py configs/tinydet_S.py --shape 320

# # calculate flops of TinyDet-L
# python tools/get_flops.py configs/tinydet_L.py --shape 512