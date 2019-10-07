# For website display
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from Fabric_Finder import model as model

# Define flask app
app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
    UPLOAD_FOLDER = "/Users/Jingji/Desktop/fabricfinder_flask/uploads",
    DATA_FOLDER = "/Users/Jingji/Desktop/fabricfinder_flask",
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
            image_link, product_link = model.recommend(filepath) # Use uploaded file to calculate
            redirect(send_file(filename))
            filename = 'http://127.0.0.1:5000/uploads/' + filename
            return render_template('template.html',link1 = image_link,link2 = product_link,image=filename)
    return



if __name__ == '__main__':
    app.run()
