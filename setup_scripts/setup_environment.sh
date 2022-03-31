########################################################################################
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O $HOME/conda.sh

bash $HOME/conda.sh -bf -p $HOME/conda/

CONDA_DIR=$HOME/conda/

echo "export "CONDA_DIR=${CONDA_DIR}"" >> $HOME/.bashrc

source $CONDA_DIR/bin/activate
########################################################################################

conda create -n gate python=3.8 -y
conda activate gate

#conda install -c conda-forge git-lfs -y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly -y
conda install opencv -y
conda install h5py -y
# optional conda install starship tmux -y
conda install gh --channel conda-forge -y
#apt install htop nvtop -y
#apt-get install ffmpeg libsm6 libxext6  -y

# optional conda install bat micro -y
########################################################################################
echo "export CODE_DIR=$HOME/target_codebase" >> $HOME/.bashrc
echo "export MOUNT_DIR=/mnt/disk/tali/" >> $HOME/.bashrc
#echo "export MOUNT_DIR=/mnt/scratch_ssd/antreas" >> $HOME/.bashrc
echo "export EXPERIMENTS_DIR=$MOUNT_DIR/experiments/" >> $HOME/.bashrc
echo "export DATASET_DIR=$MOUNT_DIR/dataset/" >> $HOME/.bashrc
echo "export TOKENIZERS_PARALLELISM=false" >> $HOME/.bashrc
echo "export FFREPORT=file=ffreport.log:level=32" >> $HOME/.bashrc
echo "export OPENCV_LOG_LEVEL=SILENT" >> $HOME/.bashrc
echo "export TMPDIR=$MOUNT_DIR/tmp" >> $HOME/.bashrc

echo "source $CONDA_DIR/bin/activate" >> $HOME/.bashrc
echo "conda activate tali" >> $HOME/.bashrc

source $HOME/.bashrc
########################################################################################
cd $HOME
git clone https://github.com/AntreasAntoniou/TALI-lightning-hydra.git $CODE_DIR
cd $CODE_DIR

pip install -r $CODE_DIR/requirements.txt
pip install -e $CODE_DIR

#cd $HOME
#git clone https://huggingface.co/openai/clip-vit-base-patch32

########################################################################################

 # ~/.config/starship.toml

command_timeout = 10000
[battery]
full_symbol = "🔋"
   8   │ charging_symbol = "🔌"
   9   │ discharging_symbol = "⚡"
  10   │
  11   │ [[battery.display]]
  12   │ threshold = 30
  13   │ style = "bold red"
  14   │
  15   │ [character]
  16   │ error_symbol = "[✖](bold red) "
  17   │
  18   │ [cmd_duration]
  19   │ min_time = 10_000  # Show command duration over 10,000 milliseconds (=10 sec)
  20   │ format = " took [$duration]($style)"
  21   │
  22   │ [directory]
  23   │ truncation_length = 5
  24   │ format = "[$path]($style)[$lock_symbol]($lock_style) "
  25   │
  26   │ [git_branch]
  27   │ format = " [$symbol$branch]($style) "
  28   │ symbol = "🍣 "
  29   │ style = "bold yellow"
  30   │
  31   │ [git_commit]
  32   │ commit_hash_length = 8
  33   │ style = "bold white"
  34   │
  35   │ [git_state]
  36   │ format = '[\($state( $progress_current of $progress_total)\)]($style) '
  37   │
  38   │ [git_status]
  39   │ conflicted = "⚔️ "
  40   │ ahead = "🏎️ 💨 ×${count}"
  41   │ behind = "🐢 ×${count}"
  42   │ diverged = "🔱 🏎️ 💨 ×${ahead_count} 🐢 ×${behind_count}"
  43   │ untracked = "🛤️  ×${count}"
  44   │ stashed = "📦 "
  45   │ modified = "📝 ×${count}"
  46   │ staged = "🗃️  ×${count}"
  47   │ renamed = "📛 ×${count}"
  48   │ deleted = "🗑️  ×${count}"
  49   │ style = "bright-white"
  50   │ format = "$all_status$ahead_behind"
  51   │
  52   │ [hostname]
  53   │ ssh_only = false
  54   │ format = "<[$hostname]($style)>"
  55   │ trim_at = "-"
  56   │ style = "bold dimmed white"
  57   │ disabled = true
  58   │
  59   │ [julia]
  60   │ format = "[$symbol$version]($style) "
  61   │ symbol = "ஃ "
  62   │ style = "bold green"
  63   │
  64   │ [memory_usage]
  65   │ format = "$symbol[${ram}( | ${swap})]($style) "
  66   │ threshold = 70
  67   │ style = "bold dimmed white"
  68   │ disabled = false
  69   │
  70   │ [package]
  71   │ disabled = true
  72   │
  73   │ [python]
  74   │ format = "[$symbol$version]($style) "
  75   │ style = "bold green"
  76   │
  77   │ [rust]
  78   │ format = "[$symbol$version]($style) "
  79   │ style = "bold green"
  80   │
  81   │ [time]
  82   │ time_format = "%T"
  83   │ format = "🕙 $time($style) "
  84   │ style = "bright-white"
  85   │ disabled = false
  86   │
  87   │ [username]
  88   │ style_user = "bold dimmed blue"
  89   │ show_always = false