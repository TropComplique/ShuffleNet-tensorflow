
# tiny-imagenet dataset has 200 categories
# and all images have 64x64 size,
# but i use 56x56 images because i do
# data augmentation by making random crops
IMAGE_SIZE = 56
NUM_CLASSES = 200
# if input image has spatial size [56, 56]
# then spatial size before the global average pooling is [4, 4]


BATCH_NORM_MOMENTUM = 0.1
# it differs from the default Tensorflow value (0.9),
# sometimes right momentum value is very important

N_SHUFFLE_UNITS = (1, 3, 1)
# number of shuffle units of stride 1 in each stage,
# in the original paper: [3, 7, 3].

# number of layers that the network will have:
# (sum(N_SHUFFLE_UNITS) + 3)*3 + 1 + 1

# stride in the first convolution layer:
# in the original paper they are using stride=2
# but because i use small 64x64 images i chose stride=1
FIRST_STRIDE = 1


# optimizer settings
MOMENTUM = 0.9
USE_NESTEROV = True
LR_REDUCE_FACTOR = 0.1


# input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = 1000
NUM_THREADS = 4
# read here about buffer sizes:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
