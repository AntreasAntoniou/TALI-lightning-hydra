export MOUNT_DIR="/mnt/disk/tali/"
export DATASET_DIR="/mnt/disk/tali"

if [ ! -d "$DATASET_DIR" ]; then
  mkdir -p $DATASET_DIR
  chmod -Rv 777 $DATASET_DIR
fi

mount -o discard,defaults /dev/sdb $DATASET_DIR

#sudo chmod -Rv 777 $DATASET_DIR
########################################################################################