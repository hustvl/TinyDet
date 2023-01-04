<div align="center">
<span><font size="5", > TinyDet: Accurately Detecting Small Objects within
1 GFLOPs </font></span>
<br>
</div>

### Requirements
- Python 3.5 or higher
- PyTorch 1.2 or higher
- CUDA 9.2 or higher
- GCC(G++) 4.9 or higher

### Installation

a. Create a conda virtual environment and activate it (Optional but recommended).

```shell
conda create --name tinydet python=3.7
conda activate tinydet
```

b. Install pytorch and torchvision.

```shell
conda install pytorch=1.2.0  torchvision=0.4.0  -c pytorch
```


c. Install mmdet (other dependencies wil be installed automatically).

```shell
pip install -r requirements.txt
python setup.py build develop
```

d. Install PSRoI align.

```shell
cd mmdet_extra/ps_roi_align_ori
python setup.py build_ext --inplace
cd ../..
```

e. Prepare dataset and checkpoint file.

Download [coco dataset](http://cocodataset.org/#download) and [checkpoint file](https://drive.google.com/drive/folders/1GC3AlgTRo9xgH_ydVYatLS8BGW0hV_w-?usp=sharing)

Folder structure:
```
TinyDet
├── mmdet
├── mmdet_extra
├── tools
├── scripts
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
├── pth_file
│   ├── mobilenetv3_bc.pt
│   ├── mobilenetv3_d.pt
│   ├── tinydet_L.pth
│   ├── tinydet_M.pth
│   ├── tinydet_S.pth
```

### Inference
```
python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py \
            ./configs/tinydet_M.py     \
            ./pth_file/tinydet_M.pth             \
            --launcher pytorch                 
```


### Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :


```
@article{chen2023tinydet,
  title={TinyDet: accurately detecting small objects within 1 GFLOPs},
  author={Chen, Shaoyu and Cheng, Tianheng and Fang, Jiemin and Zhang, Qian and Li, Yuan and Liu, Wenyu and Wang, Xinggang},
  journal={Science China Information Sciences},
  volume={66},
  number={1},
  pages={1--2},
  year={2023},
  publisher={Springer}
}  
```

