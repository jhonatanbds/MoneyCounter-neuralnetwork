model_name = "model_rpn"

from keras.models import load_model
model = load_model(f"models/{model_name}.h5")

import os
import tensorflow as tf
import keras
# Import the libraries needed for saving models
# Note that in some other tutorials these are framed as coming from tensorflow_serving_api which is no longer correct
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

# images will be the input key name
# scores will be the out key name
prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
    {
    "images": model.input
    }, {
    "scores": model.output
    })

# export_path is a directory in which the model will be created
export_path = os.path.join(
    tf.compat.as_bytes('models/export/{}'.format(model_name)),
    tf.compat.as_bytes('1'))

builder = saved_model_builder.SavedModelBuilder(export_path)

sess = keras.backend.get_session()

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
    sess, [tag_constants.SERVING],
    signature_def_map={
        'prediction': prediction_signature,
    })
# save the graph
builder.save()