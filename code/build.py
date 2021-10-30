import time
import warnings

seed_value= 0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import keras
from numpy import newaxis
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential


warnings.filterwarnings("ignore")

#Load data from data file, and split the data into training, validation and test set
def load_data(filename, step):
    #load data from the data file
    day = step
    data = np.load(filename)
    data = data[:, :]
    gt_test = data[:,day:]
    #data normalization
    max_data = np.max(data, axis = 1)
    min_data = np.min(data, axis = 1)
    max_data = np.reshape(max_data, (max_data.shape[0],1))
    min_data = np.reshape(min_data, (min_data.shape[0],1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)
    #dataset split
    # train_split = round(0.8 * data.shape[1])
    train_split = 420 - day
    # train_split = 398 - day
    # val_split = round(0.9 * data.shape[1])
    val_split = 462 - day
    # val_split = 441 - day

    x_train = data[:,:train_split]
    y_train = data[:,day:train_split+day]
    x_val = data[:,:val_split]
    y_val = data[:,day:val_split+day]
    x_test = data[:,:-day]
    y_test = data[:,day:]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    return [x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data]


def load_tw_data(filename='../dataset/tw', step=1, tw_thres=5, emb_size=50):
    tw_masks = np.load(filename + '_mask.npy')
    tw_embs = np.load(filename + '_emb_' + str(tw_thres) + '_' + str(emb_size) + '.npy')

    day = step
    # train_split = 398 - day
    train_split = 420 - day
    # val_split = 441 - day
    val_split = 462 - day

    mask_train = tw_masks[:, :train_split]
    mask_val = tw_masks[:, :val_split]
    mask_test = tw_masks[:, :-day]

    mask_train = np.reshape(mask_train, (mask_train.shape[0], mask_train.shape[1], 1))
    mask_val = np.reshape(mask_val, (mask_val.shape[0], mask_val.shape[1], 1))
    mask_test = np.reshape(mask_test, (mask_test.shape[0], mask_test.shape[1], 1))

    emb_train = tw_embs[:, :train_split, :]
    emb_val = tw_embs[:, :val_split, :]
    emb_test = tw_embs[:, :-day, :]

    return [mask_train, emb_train, mask_val, emb_val, mask_test, emb_test]


#build the model
def build_model(layers, freq, learning_rate):
    # model = Sequential()
    # from itosfm import ITOSFM
    # model.add(ITOSFM(
    #     input_dim=layers[0],
    #     hidden_dim=layers[1],
    #     output_dim=layers[2],
    #     freq_dim = freq,
    #     return_sequences=True))

    from keras.layers import Input
    prices = Input(shape=(None, layers[0]), name='input_price')
    from itosfm import ITOSFM
    predictions = ITOSFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        freq_dim=freq,
        return_sequences=True)(prices)
    from keras.models import Model
    model = Model(input=prices, output=predictions)

    model.summary()
    start = time.time()
    
    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_coasfm(layers, freq, learning_rate):
    model = Sequential()
    from coasfm import COASFM
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from coasfm import COASFM
    predictions = COASFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_asfm(layers, freq, learning_rate):
    model = Sequential()
    from asfm import ASFM
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from asfm import ASFM
    predictions = ASFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_rasfm(layers, freq, learning_rate):
    model = Sequential()
    from asfm import ASFM
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from rasfm import RASFM
    predictions = RASFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_isfm(layers, freq, learning_rate):
    model = Sequential()
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from isfm import ISFM
    predictions = ISFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_fisfm(layers, freq, learning_rate):
    model = Sequential()
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from fisfm import FISFM
    predictions = FISFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_tisfm(layers, freq, learning_rate):
    model = Sequential()
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from tisfm import TISFM
    predictions = TISFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_tfisfm(layers, freq, learning_rate):
    model = Sequential()
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from tfisfm import TFISFM
    predictions = TFISFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build the model
def build_risfm(layers, freq, learning_rate):
    model = Sequential()
    from keras.layers import Input, merge
    prices = Input(shape=(None, layers[0]), name='input_price')
    masks = Input(shape=(None, layers[0]), name='input_mask')
    embs = Input(shape=(None, layers[3] * layers[4]), name='input_emb')

    # merged = merge([prices, prices, prices], mode='concat', concat_axis=-1)
    merged = merge([prices, masks, embs], mode='concat', concat_axis=-1)

    from risfm import RISFM
    predictions = RISFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        text_num=layers[3],
        text_dim=layers[4],
        att_dim=layers[5],
        freq_dim=freq,
        return_sequences=True)(merged)
    from keras.models import Model
    model = Model(input=[prices, masks, embs], output=predictions)

    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model


# build lstm model
def build_lstm(layers, learning_rate):
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    from keras.layers.core import TimeDistributedDense
    model.add(TimeDistributedDense(output_dim=1))
    # from keras.layers import TimeDistributed
    # model.add(TimeDistributed(Dense(1)))
    model.summary()
    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer='rmsprop', loss='mse')

    print "Compilation Time: ", time.time() - start
    return model