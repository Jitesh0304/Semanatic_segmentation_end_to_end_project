
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from pathlib import Path
from src.imageSegmentation.config.configuration import ModelCallbackConfig

class ModelCallback:
    def __init__(self, config: ModelCallbackConfig):
        self.config = config

    def earlyStop_data(self):
        earlystop = EarlyStopping(monitor = self.config.monitor, patience = self.config.patience, 
                                  verbose = 1, mode = self.config.mode, restore_best_weights = self.config.restore_best_weights)
        return earlystop
    
    def checkpoint_details(self):
        checkpoint = ModelCheckpoint(filepath =self.config.model_save_path, save_weights_only =False,
                             monitor = self.config.monitor, mode = self.config.mode, save_best_only =True)
        return checkpoint
    
    def learningrate_details(self):
        learningrate = ReduceLROnPlateau(monitor = self.config.monitor, mode= self.config.mode, min_delta= self.config.min_delta,
                                          patientce =3, factor = self.config.factor, min_lr = self.config.min_learningrate,
                                            verbose =1)
        return learningrate
    

    def get_call_back(self):
        callback = [self.earlyStop_data(), self.checkpoint_details(), self.learningrate_details()]
        return callback