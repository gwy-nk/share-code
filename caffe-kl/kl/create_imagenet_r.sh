# create lmdb for distribution dataset

#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/ubuntu/sdy/train/data/lmdb
DATA=/home/ubuntu/sdy/train/data/windows
TOOLS=/home/ubuntu/sdy/caffe-kl/build/tools

TRAIN_DATA_ROOT=/home/ubuntu/sdy/dataset/fi/
VAL_DATA_ROOT=/home/ubuntu/sdy/dataset/fi/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset_r \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/window_file_train_fi_1.txt \
    $EXAMPLE/fi_1_train_lmdb

echo "Creating test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset_r \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/window_file_test_fi_1.txt \
    $EXAMPLE/fi_1_test_lmdb


#echo "Creating mean..."
#$TOOLS/compute_image_mean $EXAMPLE/fi_ld1_train_lmdb \
#$EXAMPLE/fi_ld1_mean.binaryproto

echo "Done."
