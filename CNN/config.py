import tensorflow as tf

# Parâmetros
IMAGE_SIZE = 400
NUM_CHANNELS = 3
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE # Otimização do TensorFlow
LAST_CONV_LAYER = 'resnet50v2'

class DIATOMS_CLASSES:
    def __init__(self):
        self.Diatoms_Classes_names = ['Encyonema', 'Eunotia', 'Gomphonema', 'Navicula', 'Pinnularia']
        self.class_to_int = {class_name: i for i, class_name in enumerate(self.Diatoms_Classes_names)}
        self.int_to_class = {i: class_name for i, class_name in enumerate(self.Diatoms_Classes_names)}