require 'torch'
require 'nn'
npy4th = require 'npy4th'

-- set good defaults
torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- load the network
local net = torch.load('vgg_places.t7')

print('Network:')
print(net)


local lyr_ind = 1
print(net.modules[lyr_ind].weight:size())
npy4th.savenpy('conv1_ws.npy', net.modules[lyr_ind].weight)
print(net.modules[lyr_ind].bias:size())
npy4th.savenpy('conv1_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 3
print(lyr_ind)
npy4th.savenpy('conv2_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv2_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 6
npy4th.savenpy('conv3_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv3_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 8
npy4th.savenpy('conv4_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv4_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 11
npy4th.savenpy('conv5_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv5_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 13
npy4th.savenpy('conv6_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv6_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 15
npy4th.savenpy('conv7_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv7_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 18
npy4th.savenpy('conv8_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv8_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 20
npy4th.savenpy('conv9_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv9_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 22
npy4th.savenpy('conv10_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv10_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 25
npy4th.savenpy('conv11_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv11_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 27
npy4th.savenpy('conv12_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv12_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 29
npy4th.savenpy('conv13_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('conv13_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 33
npy4th.savenpy('fc1_ws.npy', net.modules[lyr_ind].weight)
print(net.modules[lyr_ind].weight:size())
print(net.modules[lyr_ind].bias:size())
npy4th.savenpy('fc1_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 36
npy4th.savenpy('fc2_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('fc2_bs.npy', net.modules[lyr_ind].bias)

lyr_ind = 39
npy4th.savenpy('fc3_ws.npy', net.modules[lyr_ind].weight)
npy4th.savenpy('fc3_bs.npy', net.modules[lyr_ind].bias)

-- for i=1,24 do print(net.modules[i].output:size()) end

-- npy4th.savenpy('out.npy', preds[1])