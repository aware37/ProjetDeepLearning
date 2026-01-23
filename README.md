```bash
conda create -n crossvit python=3.8
conda activate crossvit
conda install pytorch=1.7.1 torchvision  cudatoolkit=11.0 -c pytorch -c nvidia
pip install -r requirements.txt

conda remove pillow --force
pip install pillow
```

