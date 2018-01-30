# Load t7 files
# Required package: torchfile. 
# $ pip install torchfile

import torchfile
import numpy as np
import pdb

# Make xrange compatible in both Python 2, 3
try:
    xrange
except NameError:
    xrange = range

keys = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
        'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']

def load(o, param_list):
    """ Get torch7 weights into numpy array """
    try:
        num = len(o['modules'])
    except:
        num = 0

    print("num", num)
    for i in xrange(num):
        # 2D conv
        print("o['modules'][i]._typename", o['modules'][i]._typename)
        if o['modules'][i]._typename.decode("utf-8")  == 'nn.SpatialConvolution' or \
            o['modules'][i]._typename == 'cudnn.SpatialConvolution':
            temp = {'weights': o['modules'][i]['weight'].transpose((2,3,1,0)),
                    'biases': o['modules'][i]['bias']}
            param_list.append(temp)

        if o['modules'][i]._typename.decode("utf-8") == 'nn.Linear' or \
                        o['modules'][i]._typename == 'cudnn.Linear':
            print("Linear")
            temp = {'weights': o['modules'][i]['weight'],
                    'biases': o['modules'][i]['bias']}
            param_list.append(temp)

        # load(o['modules'][i], param_list)


def show(o):
    """ Show nn information """
    nn = {}
    nn_keys = {}
    nn_info = {}
    num = len(o['modules']) if o['modules'] else 0
    mylist = get_mylist()

    for i in xrange(num):
        # Get _obj and keys from torchfile
        nn[i] = o['modules'][i]._obj
        nn_keys[i] = o['modules'][i]._obj.keys()
        
        # Get information from _obj
        # {layer i: {mylist keys: value}}
        nn_info[i] = {key: nn[i][key] for key in sorted(nn_keys[i]) if key in mylist}
        nn_info[i]['name'] = o['modules'][i]._typename
        print(i, nn_info[i]['name'])
        for item in sorted(nn_info[i].keys()): 
            print("  {}:{}".format(item, nn_info[i][item] if 'running' not in item \
                                                        else nn_info[i][item].shape))
        bias_key = 'gradBias'.encode('ASCII')
        weight_key = 'gradWeight'.encode('ASCII')
        if bias_key in list(nn_keys[i]) or weight_key in list(nn_keys[i]):
            if nn_info[i]['name'].find('Convolution'.encode('ASCII')) != -1 or \
                            nn_info[i]['name'].find('Batch'.encode('ASCII')) != -1:
                print(i, nn_info[i]['name'])
                print("Bias", nn[i][bias_key].shape)
                print("Weights", nn[i][weight_key].shape)
                if i == 0:
                    np.array(nn[i][weight_key]).tofile('conv1_1_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv1_1_bs.npy')
                elif i == 2:
                    np.array(nn[i][weight_key]).tofile('conv1_2_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv1_2_bs.npy')
                elif i == 5:
                    np.array(nn[i][weight_key]).tofile('conv2_1_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv2_1_bs.npy')
                elif i == 7:
                    np.array(nn[i][weight_key]).tofile('conv2_2_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv2_2_bs.npy')
                elif i == 10:
                    np.array(nn[i][weight_key]).tofile('conv3_1_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv3_1_bs.npy')
                elif i == 12:
                    np.array(nn[i][weight_key]).tofile('conv3_2_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv3_2_bs.npy')
                elif i == 14:
                    np.array(nn[i][weight_key]).tofile('conv3_3_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv3_3_bs.npy')
                elif i == 17:
                    np.array(nn[i][weight_key]).tofile('conv4_1_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv4_1_bs.npy')
                elif i == 19:
                    np.array(nn[i][weight_key]).tofile('conv4_2_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv4_2_bs.npy')
                elif i == 21:
                    np.array(nn[i][weight_key]).tofile('conv4_3_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv4_3_bs.npy')
                elif i == 24:
                    np.array(nn[i][weight_key]).tofile('conv5_1_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv5_1_bs.npy')
                elif i == 26:
                    np.array(nn[i][weight_key]).tofile('conv5_2_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv5_2_bs.npy')
                elif i == 28:
                    np.array(nn[i][weight_key]).tofile('conv5_3_ws.npy')
                    np.array(nn[i][bias_key]).tofile('conv5_3_bs.npy')
                elif i == 32:
                    np.array(nn[i][weight_key]).tofile('fc6_ws.npy')
                    np.array(nn[i][bias_key]).tofile('fc6_bs.npy')
                elif i == 35:
                    np.array(nn[i][weight_key]).tofile('fc7_ws.npy')
                    np.array(nn[i][bias_key]).tofile('fc7_bs.npy')
                elif i == 38:
                    np.array(nn[i][weight_key]).tofile('fc8_ws.npy')
                    np.array(nn[i][bias_key]).tofile('fc8_bs.npy')


def get_mylist():
    """ Return manually selected information lists """
    return ['_type', 'nInputPlane', 'nOutputPlane', \
            'input_offset', 'groups', 'dH', 'dW', \
            'padH', 'padW', 'kH', 'kW', 'iSize', \
            'running_mean', 'running_var']


if __name__ == '__main__':
    # File loader
    t7_file = 'vgg_places.t7'
    o = torchfile.load(t7_file)

    # To show nn parameter
    print("this is show")
    show(o)
    print("after show")
    
    # To store as npy file
    param_list = []
    load(o, param_list)
    print(param_list)
    """""
    save_list = {}
    print("keys", keys)
    for i, k in enumerate(keys):
        print("i", i)
        print("k", k)
        save_list[k] = param_list[i]
    print("here")
    np.save('load_t7_vgg_places', save_list)
    print("save_list", save_list)
    """

