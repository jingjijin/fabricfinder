
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
# this is model.py file
# change this later
# Import all packages first
# General packages
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Packages for object detection
import gluoncv
import mxnet as mx
from mxnet import autograd, gluon
from gluoncv import model_zoo, data, utils
from gluoncv.utils import download, viz, export_block
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

# Packagrs for processing customer uploaded image and similarity measure
import keras
from PIL import ImageOps
from keras.applications import inception_v3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from scipy.spatial import distance





# Load pre-trained SSD-ResNet50 model to perform object detection on
# user uploaded image
image = '/Users/Jingji/Desktop/FabricFinder/flask_app/uploads/test_dress1.jpg'
x, img = gluoncv.data.transforms.presets.ssd.load_test(image, short=512)



# here load model
net = gluoncv.model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=['pillow','clothing'], pretrained_base=False) # get the model with no pretraining weights and load trained weights to it
net.load_parameters('/Users/Jingji/Desktop/FabricFinder/ssd_512_resnet50_v1_fabricfinder.params') # weights is from above training



# Here perform object detection,
# get top category, score and bounding box from object detection
class_IDs, scores, bounding_boxes = net(x)

# generate cid, score, bbox
for i in range(len(scores[0])):
    cid = int(class_IDs[0][i].asnumpy())
    cname = net.classes[cid]
    score = float(scores[0][i].asnumpy())
    if score < 0.5:
        break
    x,y,w,h = bbox =  bounding_boxes[0][i].astype(int).asnumpy()



# load inception_v3 model
inception_model = inception_v3.InceptionV3(weights='imagenet',include_top=False, pooling='avg')



# process user uploaded image and extract feature vector
def user_image_feature_vector(image, bbox):
    # loade, crop and resize the image

    img = load_img(image)
    img = ImageOps.crop(img, bbox) # crop based on bbox
    img = img.resize((299,299)) # resize for feature vetor extraction
    # prep image for Inception v3
    numpy_image = img_to_array(img)
    image_batch = np.expand_dims(numpy_image, axis=0)
    # get avg pooling layer for this image
    feature_vect = inception_model.predict(image_batch)
    return feature_vect

feature_vect_user_image = user_image_feature_vector(image, bbox=(300, 370, 350, 600))





# Load pickle file of fabric feature vectors based on object type
# this need to be fixed
if cid == 1:
    fabrics_feature_vector = pd.read_pickle('/Users/Jingji/Desktop/FabricFinder/pillow_fabrics_feature_vector.pickle')
else: fabrics_feature_vector = pd.read_pickle('0000000000000') # here put clothing fabirc pickle file




# Measure similarity and return n top recommendations according to similarity
def measure_similarity(feature_vect_user_image, top_n_images=2):
    for i in range(len(fabrics_feature_vector)):
        # compute cosine distance between user image and all fabric images
        fabrics_feature_vector['similarity'] = fabrics_feature_vector['vector'].apply(lambda x: distance.cosine(x,feature_vect_user_image))
        # give recommendation based on similarity scores
        recommendation = list(fabrics_feature_vector.sort_values('similarity',ascending=True)[:top_n_images]['image_name'])

        return recommendation


# Load csv files that has fabrics info and return links based on fabric recommendations
recommendation = measure_similarity(feature_vect_user_image, top_n_images=1)
fabric_links = pd.read_csv('/Users/Jingji/Desktop/FabricFinder/pillow_fabrics_url.csv', index_col=0)
product_links = list(fabric_links[fabric_links['product_name'].isin(recommendation)]['product_link'])
fabric_images_links = list(fabric_links[fabric_links['product_name'].isin(recommendation)]['image_link'])








####################################################################
####################################################################
####################################################################
####################################################################
####################################################################











import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "/Users/Jingji/Desktop/FabricFinder/flask_app/uploads"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            links = product_links
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            redirect(send_file(filename))
            filename='http://127.0.0.1:5000/uploads/' + filename
            return render_template('template.html',links=links,image=filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run()
