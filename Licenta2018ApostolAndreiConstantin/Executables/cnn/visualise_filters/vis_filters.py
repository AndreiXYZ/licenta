from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from keras.models import load_model
from matplotlib import pyplot as plt

model = load_model('best_model_acc_98_1.h5')
plt.rcParams['figure.figsize'] = (18, 6)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'activation_5')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

model.summary()
# This is the output node we want to maximize.
filter_idx = 1
img = visualize_activation(model, layer_idx, filter_indices=[0],
						   input_range=(0.,1.), verbose=True)
plt.imshow(img[..., 0])
plt.show()