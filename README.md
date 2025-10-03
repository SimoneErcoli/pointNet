# pointNet
PointNet training with PyTorch and GPU acceleration on Mac M1/M1 Pro/M2 with Metal (MPS)

works with CSV files with point with x,y,z coordinates and "classification" column as label


Commands

python pointnet_pt_seg.py train \\
  --train-dir data/train \\
  --val-dir data/val \\
  --label-col classification \\
  --n-points 8192 \\
  --batch-size 4 \\
  --epochs 120 \\
  --lr 5e-4 \\
  --tree-weight 3.0 \\
  --balance stratified \\
  --target-ratios 0.25,0.25,0.50 \\
  --augment strong \\
  --label-smoothing 0.05 \\
  --out-model pointnet_pt_seg.pth


  python pointnet_pt_seg.py infer \
  --model pointnet_pt_seg.pth \
  --input-csv data/test_scene.csv \
  --output-csv out_pred.csv \
  --tta-rot "0,90,180,270" \
  --dbscan-eps 0.6 \
  --dbscan-min-samples 40
