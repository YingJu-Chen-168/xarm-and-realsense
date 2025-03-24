from absl import app, flags, logging
from absl.flags import FLAGS
import os
import json
import tensorflow as tf
from unet_tf2.utils import build_data, iou_coef, dice_coef
from unet_tf2.models.unet import Unet
from keras.metrics import Precision, Recall
from keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping)

# from tensorflow.keras.metrics import Precision, Recall
# from tensorflow.keras.callbacks import (
#     ReduceLROnPlateau,
#     ModelCheckpoint,
#     EarlyStopping)


# flags.DEFINE_string('dataset', "C:\Ying-Ju Chen\Lab\robot xarm\GitHub\CO2Dnet\data", 'path to dataset')
# flags.DEFINE_string(
#     'weights_save', './checkpoints/unet_train.tf', 'path to save file')

# # 檢查是否有可用的 GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 為每個 GPU 啟用記憶體增長
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("記憶體增長已啟用")
#     except RuntimeError as e:
#         print(f"發生錯誤：{e}")
# else:
#     print("未檢測到可用的 GPU")

flags.DEFINE_string(
    'weights_save', 'my_model.h5', 'path to save file')
flags.DEFINE_integer('features', 24, 'features of Unet network')
flags.DEFINE_integer('levels', 5, 'levels of Unet network')
flags.DEFINE_integer('epochs', 100, 'epochs to train')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 2e-3, 'learning rate')
flags.DEFINE_enum('mode', 'resume', ['none', 'resume'], # 在這邊調有沒有預訓練
                  'none: no load weigths, '
                  'resume: resume pre-training')
# 9e-4

def main(_argv):

    x_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/JPEGImages'
    # os.path.join(FLAGS.dataset, 'JPEGImages')
    y_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/Mask'
    # os.path.join(FLAGS.dataset, 'Mask')
    train_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/ImageSets/Main/train.txt'
    # os.path.join(FLAGS.dataset, 'ImageSets', 'Main', 'train.txt')
    val_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/ImageSets/Main/val.txt'
    # os.path.join(FLAGS.dataset, 'ImageSets', 'Main', 'val.txt')

    Xtrain, ytrain = build_data(x_path, y_path, train_path)
    Xval, yval = build_data(x_path, y_path, val_path)

    input_shape = Xtrain.shape[1:]

    model = Unet(input_shape,  FLAGS.features, FLAGS.levels)

    model_name, _ = os.path.splitext(FLAGS.weights_save)
    config_path = model_name + '.txt'
    model_config = model.to_json()

    with open(config_path, "w") as text_file:
        text_file.write(model_config)

    callbacks = [
        ReduceLROnPlateau(verbose=1, patience=5,
                          factor=0.5, monitor='val_loss'),
        ModelCheckpoint(FLAGS.weights_save,
                        verbose=1, save_weights_only=True),
    ]

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)         
    # opt_mixed_precision = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    if FLAGS.mode == 'resume':
        model.load_weights("run=0.h5")
        # model.load_weights(FLAGS.weights_save)

    model.compile(optimizer=optimizer, loss=[
                  'binary_crossentropy'], metrics=[ dice_coef, iou_coef, Precision(), Recall()])

    model.summary()

    model.fit(x=Xtrain, y=ytrain, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
              validation_data=(Xval, yval), callbacks=callbacks)
    
if __name__ == '__main__':
    app.run(main)
