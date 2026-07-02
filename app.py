from flask import Flask, render_template, request, redirect, url_for, session, flash
import cv2
import numpy as np
import os
import joblib
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.metrics import AUC # type: ignore

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management

# ---------- Load ML Model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, "ecg_rf_model.pkl")
# model = joblib.load(model_path)

model = None

def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model("winoecg.h5")
    return model
categories = ['Abnormal_Heartbeat', 'History_of_MI', 'Myocardial_Infarction', 'Normal']

verbose_name = {
0: 'Abnormal_Heartbeat',
1: 'History_of_MI',
2: 'Myocardial_Infarction',
3: 'Normal',
           }

# ---------- Database Helper ----------
def get_db_connection():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'users.db'))
    conn.row_factory = sqlite3.Row
    return conn

# ---------- Image Preprocessing ----------
def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, -1)
    return img

# ---------- Routes ----------

# Default Route → Show Register Page First
@app.route('/')
def register_page():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/chart')
def chart():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login_page'))
    return render_template('chart.html')


@app.route('/perf')
def perf():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login_page'))
    return render_template('perf.html')

@app.route('/pred')
def pred():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login_page'))
    return render_template('pred.html')

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login_page'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Email already registered!")

    return render_template('register.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_email'] = email
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid email or password")

    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_email', None)
    return redirect(url_for('login_page'))

# Home Page
@app.route('/home')
def home():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login_page'))
    return render_template('index.html')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(150,150))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 150,150,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	print(classes_x)
	return verbose_name[classes_x[0]]

# ECG Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login_page'))

    if 'ecgFile' not in request.files:
        return "No file uploaded", 400

    file = request.files['ecgFile']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)
    predict_result = predict_label(file_path)
    print(predict_result)

    #img = preprocess_image(file_path)
    #prediction = model.predict(img)[0]
   # result = categories[predict_result]

    return render_template('result.html', prediction=predict_result)

# ---------- Run App ----------
if __name__ == '__main__':
    app.run(debug=True)
