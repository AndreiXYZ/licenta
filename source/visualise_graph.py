from keras.utils import plot_model
from keras.models import load_model

model = load_model('best_model_loss.h5')
plot_model(model, to_file='best_model_loss.png', show_shapes=True)