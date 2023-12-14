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
from sklearn.ensemble import RandomForestClassifier
import joblib
import fastbook
from fastbook import *

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, array_to_img


LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Brains",
        page_icon="🧠",
    )

    st.write("brainss")



def load_models():
  model_keras = load_model('models/tumor87.h5', compile=False)
  loaded_rf = joblib.load("brainforest.joblib")
  learner = load_learner("fastai_export.pkl")
  return model_keras, loaded_rf, learner

def _process_image(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ) 
  img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
  img = img / 255.0
  #img = load_img(image_path, color_mode = 'grayscale', target_size = (700, 700))
  #img = img_to_array(img).astype('float32')/255
  return img
  


def __predict(image,model):
    y_pred = model.predict(np.expand_dims(image, axis=0), verbose=1)[0] 
    y_pred_class = np.argmax(y_pred)
    #y_pred_prob = y_pred[y_pred_class]*100 
    #score = __calculate_score(y_pred_class, y_pred_prob)
    return y_pred_class

if __name__ == "__main__":
    run()
    #st.sidebar.success("Select a demo above.")

    kerasmodel, brainforest, learner = load_models()
    img = cv2.imread('Y1.jpg')
    st.image(img)
    uplpoaded_img = st.file_uploader("upload an image of ur brain", type=['jpg','jpeg'], accept_multiple_files=False)
    if uplpoaded_img is not None:
       img = uplpoaded_img
       img = img_to_array(img).astype('float32')
       st.write("image uploaded")
       st.image(img)
    img = _process_image(img)
    st.write("keras: ",__predict(img,kerasmodel))
    st.write("random forest: ", brainforest.predict(img.reshape(1,-1)))
    st.write("fastai: ", learner.predict(img))