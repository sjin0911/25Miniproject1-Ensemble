# 25Miniproject1-Ensemble

### MWFormer .pth drive link
https://drive.google.com/drive/folders/1ekgrefSK4p6G9gtevtjy0-GNNWVa670T?usp=drive_link

### MWFormer Inference
Please check & modify >> **MWFormer/MWF_Inf_Ens.ipynb** <<

### AirNet .pth drive link
[Pre-trained Model](https://drive.google.com/drive/folders/1DS_iJsP5Epzz78fZRz8lEINcnhBF6Uws)
use All.pth

### AirNet Inference
Please check >> **AirNet/Demo.py** <<
Not yet implemented

**single** file voting ensemble
<pre> ```bash python vote_ensemble_multi.py \ --model1 /path/m1.png \ --model2 /path/m2.png \ --model3 /path/m3.png \ --gt /path/gt.png \ --outdir /path/out --smooth 5 ``` </pre>
