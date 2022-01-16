export MOUNT_DIR="/mnt/disk/filestore/"
export EXPERIMENTS_DIR="/mnt/disk/filestore/experiments/"

if [ ! -d "$MOUNT_DIR" ]; then
  sudo mkdir -p $MOUNT_DIR
  sudo chmod -Rv 777 $MOUNT_DIR
fi

if [ ! -d "$EXPERIMENTS_DIR" ]; then
  sudo mkdir -p $EXPERIMENTS_DIR
  sudo chmod -Rv 777 $EXPERIMENTS_DIR
fi
########################################################################################