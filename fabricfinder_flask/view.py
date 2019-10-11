#!/usr/bin/python
# For website display
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename

# For object detection
import pandas as pd
import numpy as np
import gluoncv
from gluoncv import model_zoo
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

# For processing customer uploaded image and similarity measure
import keras
from PIL import ImageOps
from keras.preprocessing.image import img_to_array,array_to_img,load_img
from scipy.spatial import distance
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

# load VGG16 model
vgg_model = VGG16(include_top=True, weights='imagenet')
vgg_model.layers.pop()
vgg_model.outputs = [vgg_model.layers[-1].output]
vgg_model._make_predict_function()

# Load object detection (trained) model
net = gluoncv.model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=['pillow'], pretrained_base=False)
net.load_parameters('/home/ubuntu/fabricfinder_flask/ssd_512_resnet50_v1_fabricfinder.params')

# Load fabric embeddings
fabrics = pd.read_pickle('/home/ubuntu/fabricfinder_flask/fabric_embeddings.pickle')

# Warning for no pillow in image
warning = 'None'

# Process user-uploaded image
def process_image(image_path):
    ''' input: path to user_uploaded images
        output: 1. image_NDarray: mxnet NDArray as input to network
                2. image_NParray: numpy array of the input image'''
    image_NDarray, image_NParray = gluoncv.data.transforms.presets.ssd.load_test(image_path, short=512)
    return image_NDarray, image_NParray

# Perform object detection, get top category, score and bounding box
def object_detection(image_NDarray):
    ''' input: mxnet NDArray as input to network
        output: category, confidence score, bounding box from object detection
        confidence score is sorted as descending '''
    class_IDs, scores, bounding_boxes = net(image_NDarray)
    return scores, bounding_boxes

# Select top 1 object based on confidence score
def get_top_box(scores, bounding_boxes):
    ''' input: results from object detection (classIds is actually not needed)
        output: top 1 object/bbox from all results '''
    score = float(scores[0][0].asnumpy())
    if score < 0.3: warning = 'Yes'
    b_box = bounding_boxes[0][0].astype(int).asnumpy()
    b_box = [int(x) for x in b_box.tolist()]
    return b_box

# Crop bbox from image and extract feature vector
def get_cropped_image(image_NParray, b_box, warning):
    ''' input: 1. resized (short=512) np array of input image
               2. top 1 bounding box
        output: np array of resized (224,224) cropped pillow '''
    a,b,_ = image_NParray.shape
    x, y, w, h = b_box
    border = (x,y,b-w,a-h)
    cropped_img = ImageOps.crop(array_to_img(image_NParray), border)
    cropped_img_resized = cropped_img.resize((224,224))
    cropped_img_resized = img_to_array(cropped_img_resized)
    return cropped_img_resized, border

# Get feature vector from cropped image
def get_vector(cropped_img_resized):
    ''' input: np array of resized (224,224) & cropped pillow
        output: feature vector of input image '''
    cropped_img_resized = cropped_img_resized.reshape(1, cropped_img_resized.shape[0], cropped_img_resized.shape[1], cropped_img_resized.shape[2])
    cropped_img_resized = preprocess_input(cropped_img_resized)
    try:
        feature = vgg_model.predict(cropped_img_resized)
    except:
        keras.backend.clear_session()
        vgg_model = VGG16(include_top=True, weights='imagenet')
        vgg_model.layers.pop()
        vgg_model.outputs = [vgg_model.layers[-1].output]
        vgg_model._make_predict_function()
    feature_vect = feature.flatten()
    return feature_vect

# Measure similarity and return n top recommendations according to similarity
def measure_similarity(feature_vector):
    ''' input: user image embeddings, number of fabrics to return
        output: image links, product links for recommendation '''
    fabrics['vector_similarity'] = fabrics['vector'].apply(lambda x: 1-distance.cosine(x,feature_vector))
    image_link = list(fabrics.sort_values('vector_similarity',ascending=False)[:3]['image_link'])
    product_link = list(fabrics.sort_values('vector_similarity',ascending=False)[:3]['product_link'])
    try: fabrics.drop('vector_similarity',axis=1,inplace=True)
    except: pass
    return image_link, product_link

# Final recommendation function
def recommend(image_path):
    ''' input: filepath from flask app
        output: links for recommendation '''
    image_NDarray, image_NParray = process_image(image_path)
    scores, bounding_boxes = object_detection(image_NDarray)
    b_box = get_top_box(scores, bounding_boxes)
    cropped_img_resized, border = get_cropped_image(image_NParray, b_box, warning)
    feature_vector = get_vector(cropped_img_resized)
    image_link, product_link = measure_similarity(feature_vector)
    return image_link, product_link




# Define flask app
# app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
    UPLOAD_FOLDER = "/home/ubuntu/fabricfinder_flask/uploads",
    DATA_FOLDER = "/home/ubuntu/fabricfinder_flask",
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])))


# Test allowed file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# Define upload form and upload page
@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Save uploaded file, pass to model, return links and finally render result page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(filepath)
            image_link, product_link = recommend(filepath) # Use uploaded file to calculate
            redirect(send_file(filename))
            filename = 'http://18.190.119.234/uploads/' + filename
            return render_template('template.html',link1 = image_link,link2 = product_link,image=filename)
    return

#
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)
