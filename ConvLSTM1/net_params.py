from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell



# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [6, 16, 6, 5, 1]}),  ### 输入先经过此卷积，卷积后输入rnn，卷积后长宽决定了rnncell输入shape
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [    
        CLSTM_cell(shape=(92,92), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(46,46), input_channels=64, filter_size=5, num_features=96),
        CLSTM_cell(shape=(23,23), input_channels=96, filter_size=5, num_features=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 16, 6, 5, 1],
            'conv3_leaky_1': [16, 8, 3, 1, 1],
            'conv4_leaky_1': [8, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(23,23), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(46,46), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(92,92), input_channels=96, filter_size=5, num_features=64),
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [3, 16, 6, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(92,92), input_channels=16, filter_size=5, num_features=64),
        CGRU_cell(shape=(46,46), input_channels=64, filter_size=5, num_features=96),
        CGRU_cell(shape=(23,23), input_channels=96, filter_size=5, num_features=96)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 16, 6, 5, 1],
            'conv3_leaky_1': [16, 8, 3, 1, 1],
            'conv4_leaky_1': [8, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(23,23), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(46,46), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(92,92), input_channels=96, filter_size=5, num_features=64),
    ]
]