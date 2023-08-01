import tensorflow as tf
import segmentation_models as sm
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from pathlib import Path
from src.imageSegmentation.entity.config_entity import ModelTrainingConfig
from tensorflow.keras import backend as K



def bce_jaccard_loss(y_train, y_valid, smooth=1e-7):
    y_train_flat = K.flatten(y_train)
    y_valid_flat = K.flatten(y_valid)

    bce_loss = K.binary_crossentropy(y_train_flat, y_valid_flat)

    intersection = K.sum(y_train_flat * y_valid_flat)
    union = K.sum(y_train_flat) + K.sum(y_valid_flat) - intersection
    jaccard_loss = 1.0 - (intersection + smooth) / (union + smooth)
    total_loss = bce_loss + jaccard_loss
    return total_loss

def iou_score(y_train, y_valid, smooth=1e-7):

    y_train_flat = K.flatten(y_train)
    y_valid_flat = K.flatten(y_valid)

    intersection = K.sum(y_train_flat * y_valid_flat)
    union = K.sum(y_train_flat) + K.sum(y_valid_flat) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou
    


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def spliting_the_dataset(self, training, testing):
        x_train, x_valid , y_train, y_valid = train_test_split(training, testing, test_size= 0.15, random_state= 0)
        preprocess_input = sm.get_preprocessing(self.config.backbone)
        X_train = preprocess_input(x_train)
        X_valid = preprocess_input(x_valid)
        return X_train, X_valid , y_train, y_valid
    
    def model_build(self):

        input_shape = self.config.image_size 
        num_classes = self.config.number_of_classes

        inputs = Input(input_shape)
        # Downsampling path
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Upsampling path
        up6 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(drop5)
        up6 = concatenate([up6, drop4])
        conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

        up7 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
        up7 = concatenate([up7, conv3])
        conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

        up8 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
        up8 = concatenate([up8, conv2])
        conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
        up9 = concatenate([up9, conv1])
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

        outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        return model


    def compile_model(self, image_size: int, number_of_classes: int, learning_rate: float):
        model = self.model_build()  
        optimizer = Adam(learning_rate = learning_rate)
        # model.compile(optimizer= optimizer, loss= categorical_crossentropy, metrics= ['accuracy'])
                ## for custom loss and metrics
        model.compile(optimizer=optimizer, loss=bce_jaccard_loss, metrics=[iou_score])
        return model
    
    
    def fit_model(self, X_train, X_valid , y_train, y_valid, callback):
        model = self.compile_model(self.config.image_size, self.config.number_of_classes, self.config.learning_rate)
        model.fit(X_train, y_train, batch_size= self.config.batch_size, epochs= self.config.epochs, 
                  validation_data= (X_valid, y_valid), callbacks= callback)
        self.save_model(self.config.base_model, model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)