from flask import Flask, request, render_template
import numpy as np
from joblib import load
import keras
import cv2
import tensorflow as tf
app = Flask(__name__)



@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        image_file = request.files['ximage']
        image_path = "user_input_images/" + image_file.filename
        image_file.save(image_path)
        model_new = keras.models.load_model('covpredd.h5')
        pred = cov_or_not(model_new, image_path)
        predicted = str(pred)
        
        # {'Covid': 0, 'Normal': 1}

        if predicted == "[[0.]]":
            pred_str = 'Positive'
        else:
            pred_str = 'Negative'

        return render_template('index.html', prediction=pred_str)
        


def cov_or_not(model, image_path):
    # test_img1 = tf.keras.utils.load_img(image_path, target_size=(224,224))
    test_img1 = cv2.imread(image_path)
    test_img1 = cv2.resize(test_img1,(224,224))
    test_img1 = np.array(test_img1)
    test_img1 = np.expand_dims(test_img1, axis = 0) 
    predicted_op = model.predict(test_img1)
    return predicted_op