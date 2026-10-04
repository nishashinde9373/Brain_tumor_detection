import os
import time
from flask import Flask, render_template, request, session, redirect, url_for
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change to a secure key

# Load model
model_path = os.path.join(os.path.dirname(__file__), "best_model.h5")
model = tf.keras.models.load_model(model_path)

# Use absolute path for upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'admin' and password == 'password':
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = "Invalid username or password"
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = None
    image_path = None

    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file:
            # Ensure folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file in uploads folder
            file.save(filepath)

            # Create URL for the saved image with cache buster
            image_path = url_for('static', filename='uploads/' + filename) + '?v=' + str(int(time.time()))

            # Predict
            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)
            class_index = np.argmax(preds, axis=1)[0]
            prediction = class_names[class_index]

    return render_template("index.html", prediction=prediction, image_path=image_path, username=session['username'])

if __name__ == "__main__":
    app.run(debug=True)
