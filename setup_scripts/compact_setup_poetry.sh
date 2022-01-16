#wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O /home/evolvingfungus/conda.sh; bash /home/evolvingfungus/conda.sh -bf -p /home/evolvingfungus/conda/
#source $HOME/conda/bin/activate
#conda init bash
#source $HOME/.bashrc
#conda create -n tali-env python=3.8 -y
#echo "conda activate tali-env" >> $HOME/.bashrc
#source $HOME/.bashrc
#conda activate tali-env
#conda install poetry git pytorch torchvision torchaudio cudatoolkit=11.3 opencv -c pytorch -c conda-forge -y
#########################################################################################
#cd /home/evolvingfungus/current_research_forge/TALI
#poetry install
#conda install poetry git pytorch torchvision torchaudio torchmetrics tensorboardx tensorboard cudatoolkit=11.3 opencv -c pytorch -c conda-forge -y
########################################################################################
#conda install setuptools==59.5.0
pip install -e $HOME/current_research_forge/TALI-lightning-hydra/
pip install -e $HOME/current_research_forge/GATE/
pip install "ray[default]"
echo "export TOKENIZERS_PARALLELISM=false" >> $HOME/.bashrc
echo "export GOOGLE_APPLICATION_CREDENTIALS=/home/evolvingfungus/current_research_forge/TALI-lightning-hydra/gcp/credentials/ray.json" >> $HOME/.bashrc
sudo apt install nvtop -y
conda install -c conda-forge git-lfs
git clone https://huggingface.co/openai/clip-vit-base-patch32
########################################################################################
#DIR="/home/evolvingfungus/build_open_cv/"
#if [ -d "$DIR" ]; then
#  yes | sudo rm -rf /home/evolvingfungus/build_open_cv/ /home/evolvingfungus/experiment_storage_disk/ /home/evolvingfungus/opencv/ /home/evolvingfungus/opencv_contrib/ /home/antrikohs/ /home/evolvingfungus/.cache
#fi
########################################################################################
#echo "cd $HOME/current_research_forge/TALI/" >> $HOME/.profile
#echo "poetry install" >> $HOME/.profile
#echo source "\$(poetry env info --path)/bin/activate" >> $HOME/.profile
#echo "cd $HOME" >> $HOME/.profile
#gsutil rsync -r -u -d  /mnt/disk/filestore/experiments/base-tali-milli_modus_prime_resnet50-22122021 gs://tali-experiments/base-tali-milli_modus_prime_resnet50-22122021