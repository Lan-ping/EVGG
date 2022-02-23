
from network.enhanced_vgg import EVGG

model = EVGG(input_shape=(256, 256, 3), pooling='avg')
model.summary()