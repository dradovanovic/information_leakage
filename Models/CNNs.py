import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model

class Wang_CNN:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def prepare_hourly_data_for_wang(data):
        """This Method brings the 15-minute wide data format into
        hourly 7 x 24 per Household weekly snippet. So for one Household
        the shape is changed from 1 x 672 (15 minute intervals) to 7 x 24 (hourly intervals)"""
        # Build 4-wide bins along the columns; aggregate by SUM (good for kWh)
        bins = pd.Index(range(data.shape[1])) // 4
        df_hourly = pd.DataFrame(data).groupby(bins, axis=1).sum(min_count=1)
        reshaped_data = df_hourly.to_numpy(copy=False).reshape(df_hourly.shape[0], 7, 24)
        return reshaped_data

    @staticmethod
    def prepare_data_for_desired_resolution_for_wang(data, freq='15min'):
        time_resolution = int(freq[0:2])
        values_per_day = int(24 * (60/time_resolution))
        reshaped_data = data.reshape(data.shape[0], 7, values_per_day, 1)
        return reshaped_data

    @staticmethod
    def build_model(days=7, measurements=24, n_classes=2, dropout_rate=0.5):
        inp = L.Input(shape=(days, measurements, 1), name="Input")
        # C1: Conv2D -> 6x22x8
        x = L.Conv2D(filters=8, kernel_size=(2,3), activation=tf.nn.relu, name="C1")(inp)
        # C2: Conv2D -> 4x20x16
        x = L.Conv2D(filters=16, kernel_size=(3,3), activation=tf.nn.relu, name="C2")(x)
        # P1: MaxPool 2x2 -> 2x10x16
        x = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="P1")(x)
        # Dr1: Dropout
        x = L.Dropout(rate=dropout_rate, name="DR1")(x)
        # F1: Flatten -> 320
        x = L.Flatten(name="F1")(x)
        # D1: Dense 32
        x = L.Dense(32, activation="relu", name="D1")(x)
        # D2: final layer
        outputs = L.Dense(n_classes, activation="softmax", name="D2")(x)

        model = Model(inputs=inp, outputs=outputs, name="Wang_CNN_7x24x1")
        return model

    @staticmethod
    def build_model_with_aux(days=7,
                             measurements=96,
                             n_classes=2,
                             aux_logits=3,
                             dropout_rate=0,
                             model_name='base_wang_with_aux'):

        inp = L.Input(shape=(days, measurements, 1), name="Input")
        # C1: Conv2D -> 6x22x8
        x = L.Conv2D(filters=8, kernel_size=(2, 3), activation=tf.nn.relu, name="C1")(inp)
        # C2: Conv2D -> 4x20x16
        x = L.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu, name="C2")(x)
        # P1: MaxPool 2x2 -> 2x10x16
        x = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="P1")(x)
        # Dr1: Dropout
        x = L.Dropout(rate=dropout_rate, name="DR1")(x)
        # F1: Flatten -> 320
        x = L.Flatten(name="F1")(x)
        # D1: Dense 32
        x = L.Dense(32, activation="relu", name="D1")(x)

        # Real class logits
        real = L.Dense(n_classes, activation=None, name="real_logits")(x)

        # Auxiliary logits
        aux = L.Dense(aux_logits, name="aux_logit")(x)
        model = Model(inputs=inp, outputs=[real, aux], name=model_name)
        return model

class Ferner:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def prepare_15min_data_for_ferner(data):
        reshaped_data = data.reshape(data.shape[0], 96, 365)
        reshaped_data = reshaped_data[..., None]
        return reshaped_data

    @staticmethod
    def build_model(measurements=96, days=365, dropout_rate=0.5):
        inp = L.Input(shape=(measurements, days, 1), name="Input")
        x = L.Conv2D(filters=15, kernel_size=(5,5), strides=(1, 1), padding="same",
                     # activation='relu',
            name="C1")(inp)
        x = L.BatchNormalization(name="B1")(x)
        x = L.ReLU(name="ReLU1")(x)
        x = L.MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding="same", name="P1")(x)

        x = L.Conv2D(filters=15, kernel_size=(3,3), strides=(1, 1), padding="same",
                     # activation='relu',
                     name="C2")(x)
        x = L.BatchNormalization(name="B2")(x)
        x = L.ReLU(name="ReLU2")(x)
        x = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="P2")(x)

        x = L.Conv2D(filters=15, kernel_size=(3,3), strides=(1, 1), padding="same",
                     # activation='relu',
                     name="C3")(x)
        x = L.BatchNormalization(name="B3")(x)
        x = L.ReLU(name="ReLU3")(x)
        x = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="P3")(x)

        x = L.Conv2D(filters=15, kernel_size=(3,3), strides=(1, 1), padding="same",
                     # activation='relu',
                     name="C4")(x)
        x = L.BatchNormalization(name="B4")(x)
        x = L.ReLU(name="ReLU4")(x)
        x = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="P4")(x)

        x = L.Flatten(name="F1")(x)

        # D1: Dense 32
        x = L.Dense(32,
                    # activation='relu',
                    name="D1")(x)
        x = L.BatchNormalization(name="B5")(x)
        x = L.ReLU(name="ReLU5")(x)
        # Dr1: Dropout
        x = L.Dropout(rate=dropout_rate, name="DR1")(x)

        # D2: final layer
        # When working with Softmax, change units=2 and activation="softmax"
        outputs = L.Dense(2, activation="softmax", name="D2")(x)
        #outputs = L.Dense(1, activation="sigmoid", name="D2")(x)

        model = Model(inputs=inp, outputs=outputs, name="Ferner_CNN_96x7x1")
        return model