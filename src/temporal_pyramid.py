from keras.layers import concatenate, UpSampling1D
from .multi_branch_conv_block import MultiBranchConv1D


# Define a temporal pyramid network
def TempPyramid(input_f, input_2, input_4, seq_len, n_dims):
    #### Full scale sequences
    conv1 = MultiBranchConv1D(input_f, 16, 3, 2, 2)

    #### Half scale sequences
    conv2 = MultiBranchConv1D(input_2, 16, 3, 2, 1)

    #### Quarter scale sequences
    conv3 = MultiBranchConv1D(input_4, 16, 3, 1, 1)

    #### Recurrent layers
    x = concatenate([conv1, conv2, conv3], axis=-1)
    return x
