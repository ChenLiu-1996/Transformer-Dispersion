


## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name transformer pytorch==2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c anaconda -c conda-forge -y
conda activate transformer
conda install scikit-learn scikit-image pandas matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge -y

python -m pip install webdataset einops open-clip-torch
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install diffusers["torch"]==0.21.4 transformers huggingface_hub==0.25.2
python -m pip install datasets sentencepiece
python -m pip install numpy==1.26
python -m pip install nltk

python -m pip install -U phate
```