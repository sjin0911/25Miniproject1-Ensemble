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

![Voting Ensemble Result](Ensemble_result/voting.png)

### single file (= 1 image) voting ensemble
<pre> python vote_ensemble_multi.py \
  --model1 /path/m1.png \
  --model2 /path/m2.png \
  --model3 /path/m3.png \
  --gt /path/gt.png \
  --outdir /path/out \
  --smooth 5 </pre>
### multi files (= n images) voting ensemble
<pre> python vote_ensemble_multi.py \
  --m1_dir /content/.../model1_folder \
  --m2_dir /content/.../model2_folder \
  --m3_dir /content/.../model3_folder \
  --gt_dir /content/.../GT_folder \
  --outdir /content/.../vote_result --smooth 5 \
  --csv /content/.../vote_summary.csv </pre>

### single file (= 1 image) blending ensemble
<pre> python blend_ensemble_patch_batch_multi.py \
  --model1 /path/m1.png \
  --model2 /path/m2.png \
  --model3 /path/m3.png \
  --model4 /path/m4.png \
  --gt /path/gt.png \
  --outdir /path/out </pre>
### multi files (= n images) blending ensemble
<pre> python blend_ensemble_patch_batch_multi.py \
  --m1_dir /content/.../model1 \
  --m2_dir /content/.../model2 \
  --m3_dir /content/.../model3 \
  --gt_dir /content/.../GT \
  --outdir /content/.../blend_result \
  --csv /content/.../blend_summary.csv </pre>
