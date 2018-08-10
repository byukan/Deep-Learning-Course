# define different architectures for experimentation in this file

from trainer import Trainer
from keras.regularizers import l2


class MLRTrainer(Trainer):
    """Multi-Class Logistic Regression Classifier"""

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Activation

        model = Sequential()
        model.add(Flatten(input_shape=self.X[0].shape))
        model.add(Dense(units=self.Y[0].shape[0], W_regularizer=l2(self.C['reg'])))
        model.add(Activation('softmax'))
        self.model = model

class MLPTrainer(Trainer):
    """Multi-Layer Perceptron Classifier"""

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Activation

        model = Sequential()
        model.add(Flatten(input_shape=self.X[0].shape))
        model.add(Dense(units=self.C['hidden_size'], activation='relu'))
        model.add(Dense(units=self.Y[0].shape[0]))
        model.add(Activation('softmax'))
        self.model = model
