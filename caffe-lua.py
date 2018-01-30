require 'loadcaffe';
model = loadcaffe.load('deploy_vgg16places2.prototxt', 'vgg16_places2.caffemodel');
torch.save("vgg_places.t7", model)
