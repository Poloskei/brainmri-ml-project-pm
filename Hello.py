# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import tensorflow as tf
import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, array_to_img


LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Brains",
        page_icon="🧠",
    )

    st.write("brainss")



#new_model = tf.keras.models.load_model('my_model.keras')
def load_models():
  model_keras = load_model('models/keras_model.h5', compile=False)
  #model_auto = load_model('models/auto_model.h5', compile=False)
  return model_keras

def __load_and_preprocess_custom_image():
  img = st.file_uploader("upload an image of ur brain", type=['jpg','jpeg'], accept_multiple_files=False)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ) 
  img = cv2.resize(img, (75, 75), interpolation=cv2.INTER_AREA)
  img = img / 255.0
#  img = load_img(image_path, color_mode = 'grayscale', target_size = (700, 700))
  #img = img_to_array(img).astype('float32')/255
  return img

def __predict_score(image,model):
    image = __load_and_preprocess_custom_image(image)
    y_pred = model.predict(np.expand_dims(image, axis=0), verbose=1)[0] 
    y_pred_class = np.argmax(y_pred)
    y_pred_prob = y_pred[y_pred_class]*100 
    #score = __calculate_score(y_pred_class, y_pred_prob)
    return y_pred_class

if __name__ == "__main__":
    run()
    #st.sidebar.success("Select a demo above.")

    kerasmodel = load_models()
    img = __load_and_preprocess_custom_image()
    st.write("image uploaded")
    st.image(img)
    st.write("keras: " +__predict_score(img,kerasmodel))