#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from typing import Union, List
import numpy as np


class PredictionModel:
    """
    Helper class to load an exported model and apply it to image segments for transcription.

    :ivar session: ``tf.Session`` within which to run the loading process
    :vartype session: tf.Session
    :ivar model: loaded exported model
    :vartype model:

    :param model_dir: directory containing the saved model files.
    :param session: ``tf.Session`` to load the model
    :param signature: which signature to use to select the type of input :

            - predictions (default) : input a grayscale image
            - rgb_images : input a RGB image
            - filename : input the filename of the image segment
    """

    def __init__(self, model_dir: str, session: tf.Session=None, signature: str= 'predictions'):
        # Get session
        
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()

        # Load model
        self.model = tf.saved_model.loader.load(self.session, [tag_constants.SERVING], model_dir)

        # Gets the signature to be used
        if signature == 'predictions':
            self._input_dict, self._output_dict = _signature_def_to_tensors(self.model.signature_def['predictions'])
            input_dict_key = 'images'
        elif signature == 'rgb_images':
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(self.model.signature_def['input_rgb:{}'
                                          .format(DEFAULT_SERVING_SIGNATURE_DEF_KEY)])
            input_dict_key = 'rgb_images'
        elif signature == 'filename':
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(self.model.signature_def['input_filename:{}'
                                          .format(DEFAULT_SERVING_SIGNATURE_DEF_KEY)])
            input_dict_key = 'filename'
        elif signature == 'default':
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(self.model.signature_def[DEFAULT_SERVING_SIGNATURE_DEF_KEY])
            input_dict_key = 'images'
        else:
            raise NotImplementedError

        assert input_dict_key in self._input_dict.keys(), \
            'There is no "{}" key in input dictionnary. Try "{}"'.format(input_dict_key, self._input_dict.keys())

        self._input_tensor = self._input_dict[input_dict_key]

    def predict(self, input_to_predict: Union[np.ndarray, str]) -> dict:
        """
        Get transcription for input data.

        :param input_to_predict: input data of the format specified in `signature` when
            instantiating the object
        :return: a dictionary with the predictions
        """
        output = self._output_dict
        input_tensor = self._input_tensor
        return self.session.run(output, feed_dict={input_tensor: input_to_predict})


class PredictionModelBatch:
    """
    The same helper class as ``PredictionModel`` but for batch prediction.

    :ivar session: ``tf.Session``within which to run the loading process
    :vartype session: tf.Session
    :ivar model: loaded exported model
    :vartype model:
    """

    def __init__(self, model_dir: str, session: tf.Session = None, signature: str=DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        # Get session
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()

        # Load model
        self.model = tf.saved_model.loader.load(self.session, [tag_constants.SERVING], model_dir)

        # Gets the signature to be used
        if signature == DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            self._input_dict, self._output_dict = \
                _signature_def_to_tensors(
                    self.model.signature_def[DEFAULT_SERVING_SIGNATURE_DEF_KEY])
        else:
            raise NotImplementedError

        # assert input_dict_key in self._input_dict.keys(), \
        #     'There is no {} key in input dictionnary. Try {}'.format(input_dict_key, self._input_dict.keys())

        # Get init op for dataset
        g = tf.get_default_graph()
        self._init_op = g.get_operation_by_name('dataset_init')

    def predict(self, input_to_predict: List[str], batch_size: int=128) -> dict:
        # First run init op, then prediction
        _, predictions = self.session.run([self._init_op, self._output_dict],
                                          feed_dict={self._input_dict['list_filenames']: input_to_predict,
                                                     self._input_dict['batch_size']: batch_size})

        return predictions


def _signature_def_to_tensors(signature_def):  # from SeguinBe
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}