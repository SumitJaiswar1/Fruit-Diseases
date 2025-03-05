from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
import mysql.connector
import os
import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load Hugging Face API token from environment variable
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# Configure MySQL Database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345",
    database="login_users",
    autocommit=True  # ✅ Ensures automatic commit
)
cursor = db.cursor()

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model paths and loading
model_paths = {
    'tomato': 'models/tomato_disease_model.h5',
    'mango': 'models/mango_disease_model.h5',
    'apple': 'models/apple_disease_model.h5'
}
models = {fruit: tf.keras.models.load_model(path) for fruit, path in model_paths.items() if os.path.exists(path)}

# Class names
class_names = {
    'tomato': ['tomato_reject', 'tomato_ripe', 'tomato_unripe'],
    'mango': ['mango_Alternaria', 'mango_anthracnose', 'mango_BlackMouldRot', 'mango_healthy', 'mango_stem&rot'],
    'apple': ['apple_BOTCH', 'apple_NORMAL', 'apple_ROT', 'apple_SCAB']
}
IMAGE_SIZES = {'tomato': (224, 224), 'mango': (224, 224), 'apple': (224, 224)}

@app.route('/')
def home():
    return render_template('home.html', logged_in=('user_id' in session))

@app.route('/menu')
def menu():
    return render_template('menu.html', logged_in=('user_id' in session))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        flash("A user is already logged in!", "warning")
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password!", "danger")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        flash("You are already logged in!", "warning")
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password)

        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            flash("Username or Email already exists!", "error")
            return redirect(url_for('register'))
        
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        flash("Registration successful!", "success")
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/upload/<fruit>', methods=['GET', 'POST'])
def upload(fruit):
    if 'user_id' not in session:  # Ensure the user is logged in
        flash("You must be logged in to upload images.", "warning")
        return redirect(url_for('login'))
    
    if fruit.lower() not in models:
        flash('Invalid fruit type selected!')
        return redirect(url_for('menu'))

    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class, predicted_probability = make_prediction(fruit, file_path)

            try:
                user_id = session.get('user_id')  # Ensure user_id is retrieved
                if not user_id:
                    flash("Error: User ID not found in session!", "danger")
                    return redirect(url_for('login'))

                # Insert data into user_history
                cursor.execute(
                    "INSERT INTO user_history (user_id, image_path, predicted_class, upload_time) VALUES (%s, %s, %s, NOW())",
                    (user_id, file_path, predicted_class)
                )
                db.commit()  # ✅ Ensures the data is saved to MySQL

                return render_template(
                    'result.html',
                    fruit=fruit,
                    filename=filename,
                    predicted_class=predicted_class,
                    predicted_probability=predicted_probability
                )
            except mysql.connector.Error as err:
                db.rollback()  # Rollback in case of an error
                flash(f"Database Error: {err}", "danger")
                print("Database Error:", err)

    return render_template('upload.html', fruit=fruit)


def make_prediction(fruit, image_path):
    target_size = IMAGE_SIZES.get(fruit.lower(), (224, 224))
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    model = models[fruit.lower()]
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_names[fruit.lower()][predicted_class_index], predictions[0][predicted_class_index] * 100

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("You must be logged in to view history.", "warning")
        return redirect(url_for('login'))

    user_id = session['user_id']
    try:
        cursor.execute("SELECT id, image_path, predicted_class, upload_time FROM user_history WHERE user_id = %s ORDER BY upload_time DESC", (user_id,))
        history_data = cursor.fetchall()

        if not history_data:
            flash("No history found!", "info")

        return render_template('history.html', history_data=history_data, logged_in=True)
    except mysql.connector.Error as err:
        flash(f"Database Error: {err}", "danger")
        print("Database Error:", err)
        return redirect(url_for('home'))


@app.route('/delete_history/<int:entry_id>', methods=['POST'])
def delete_history(entry_id):
    if 'user_id' not in session:
        flash("You must be logged in to delete history.", "warning")
        return redirect(url_for('login'))

    # Fetch image path before deleting entry
    cursor.execute("SELECT image_path FROM user_history WHERE id = %s", (entry_id,))
    result = cursor.fetchone()

    if result:
        image_path = result[0]
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete the selected history entry
    cursor.execute("DELETE FROM user_history WHERE id = %s", (entry_id,))
    db.commit()

    # Reset ID order to remove gaps
    cursor.execute("SET @new_id = 0;")
    cursor.execute("UPDATE user_history SET id = (@new_id := @new_id + 1) ORDER BY upload_time;")
    cursor.execute("ALTER TABLE user_history AUTO_INCREMENT = 1;")
    db.commit()

    flash("History entry deleted and IDs reordered successfully!", "success")
    return redirect(url_for('history'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Zephyr AI Route
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    if not HF_API_TOKEN:
        return jsonify({"error": "Hugging Face API token not set"}), 500

    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": user_query, "parameters": {"max_new_tokens": 150}}

    response = requests.post("https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta", headers=headers, json=payload)
    
    if response.status_code == 200:
        return jsonify({"response": response.json()[0].get("generated_text", "No response generated.")})
    return jsonify({"error": "Failed to fetch response from AI"}), 500

@app.route('/chat')
def ai_chat():
    return render_template('ai_chat.html')

if __name__ == '__main__':
    app.run(debug=True)
