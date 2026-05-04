import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = 'iot_club_heart_project_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Load the Model ---
# Ensure heart_model.pkl is in the same directory
model = joblib.load('heart_model.pkl')

# --- Database Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Authentication Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = generate_password_hash(request.form.get('password'))
        new_user = User(username=request.form.get('username'), password=hashed_pw)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login.')
            return redirect(url_for('login'))
        except:
            flash('Username already exists')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Main App Routes ---

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # 1. Capture Inputs & Convert Age (Years to Days)
        age_years = float(request.form.get('age'))
        age = age_years * 365.25  # Conversion for AI model accuracy
        
        gender = int(request.form.get('gender'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        ap_hi = int(request.form.get('ap_hi'))
        ap_lo = int(request.form.get('ap_lo'))
        chol = int(request.form.get('cholesterol'))
        gluc = int(request.form.get('gluc'))
        smoke = int(request.form.get('smoke'))
        alco = int(request.form.get('alco'))
        active = int(request.form.get('active'))

        # 2. Heuristic Warnings (Clinical Observations)
        warnings = []
        bmi = weight / ((height / 100) ** 2)
        if bmi >= 25: warnings.append(f"High BMI ({round(bmi, 1)})")
        if chol >= 2: warnings.append("Elevated Cholesterol")
        if gluc >= 2: warnings.append("Elevated Glucose")
        if smoke == 1: warnings.append("Smoking Habit")
        if active == 0: warnings.append("Sedentary Lifestyle")
        if ap_hi >= 130 or ap_lo >= 85: warnings.append("Elevated Blood Pressure")

        # 3. Model Prediction
        feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        input_values = [age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active]
        df_input = pd.DataFrame([input_values], columns=feature_names)
        prediction = model.predict(df_input)[0]

        # 4. Result Formatting & Nuanced Messaging
        status_class = "high-risk" if prediction == 1 else "low-risk"
        
        if prediction == 0 and warnings:
            # The "Nuanced Low Risk" logic
            safety_note = "Your overall risk is low, but keep in mind the specific clinical observations listed below."
        elif prediction == 0:
            safety_note = "Your biometric markers are within standard healthy ranges."
        else:
            safety_note = "High risk detected. Please consult a healthcare professional regarding the observations below."

        return render_template('result.html', 
                               prediction=int(prediction), 
                               status_class=status_class, 
                               safety_note=safety_note,
                               warnings=warnings)

    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)