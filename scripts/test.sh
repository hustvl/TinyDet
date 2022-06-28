#!/usr/bin/env sh

#test TinyDet-M
python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py \
            ./configs/tinydet_M.py     \
            ./pth_file/tinydet_M.pth             \
            --launcher pytorch                 


# #test TinyDet-S
# python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py \
#             ./configs/tinydet_S.py     \
#             ./pth_file/tinydet_S.pth             \
#             --launcher pytorch                 


# #test TinyDet-L
# python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py \
#             ./configs/tinydet_L.py     \
#             ./pth_file/tinydet_L.pth             \
#             --launcher pytorch                 
