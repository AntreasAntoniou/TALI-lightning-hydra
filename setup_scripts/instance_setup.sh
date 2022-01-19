
########################################################################################
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O $HOME/conda.sh; bash $HOME/conda.sh -bf -p $HOME/conda/

CONDA_DIR=$HOME/conda/

source $CONDA_DIR/bin/activate

echo "source $CONDA_DIR/bin/activate" >> $HOME/.bashrc

conda create -n tali python=3.8 -y
conda activate tali

conda install -c conda-forge git-lfs -y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly -y
conda install google-cloud-sdk -y
conda install opencv -y

echo "export CODE_DIR=$HOME/TALI-lightning-hydra" >> $HOME/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> $HOME/.bashrc
echo "conda activate tali" >> $HOME/.bashrc
source $HOME/.bashrc

cd $CODE_DIR
pip install -r $CODE_DIR/requirements.txt
pip install -e .

cd $HOME
git clone https://huggingface.co/openai/clip-vit-base-patch32

# /usr/local/cuda-11.2/