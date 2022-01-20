
########################################################################################
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O $HOME/conda.sh; bash $HOME/conda.sh -bf -p $HOME/conda/

CONDA_DIR=$HOME/conda/

source $CONDA_DIR/bin/activate
########################################################################################

conda create -n tali python=3.8 -y
conda activate tali

conda install -c conda-forge git-lfs -y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly -y
conda install opencv -y

########################################################################################
echo "export CODE_DIR=$HOME/TALI-lightning-hydra" >> $HOME/.bashrc
echo "export MOUNT_DIR=/mnt/disk/filestore/" >> $HOME/.bashrc
echo "export EXPERIMENTS_DIR=/mnt/disk/filestore/experiments/" >> $HOME/.bashrc
echo "export DATASET_DIR=/mnt/disk/filestore/tali-dataset/" >> $HOME/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> $HOME/.bashrc

echo "source $CONDA_DIR/bin/activate" >> $HOME/.bashrc
echo "conda activate tali" >> $HOME/.bashrc

source $HOME/.bashrc
########################################################################################
cd $HOME
git clone https://github.com/AntreasAntoniou/TALI-lightning-hydra.git $CODE_DIR
cd $CODE_DIR

pip install -r $CODE_DIR/requirements.txt
pip install -e $CODE_DIR

cd $HOME
git clone https://huggingface.co/openai/clip-vit-base-patch32

########################################################################################
conda install gh --channel conda-forge
sudo apt install htop nvtop
conda install google-cloud-sdk bat micro -y
