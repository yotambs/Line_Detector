from tensorflow import keras
import segmentation_models as sm
#loss=sm.losses.bce_jaccard_loss,
#     metrics=[sm.metrics.iou_score
model = keras.models.load_model('/models/Scratch_detector_diceloss.h5',custom_objects=sm.losses.dice_loss())
print('End')