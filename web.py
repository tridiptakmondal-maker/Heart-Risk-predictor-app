import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- Absolute Path Configuration for Render ---
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'iot_club_heart_project_2026')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Load the Model using Absolute Path ---
model_path = os.path.join(basedir, 'heart_model.pkl')
model = joblib.load(model_path)

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
            flash('Account created successfully! Please login.')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Username already exists or database error.')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Main Logic Routes ---

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # 1. Capture Inputs & Convert Years to Days
        age_years = float(request.form.get('age'))
        age_days = age_years * 365.25
        
        gender = int(request.form.get('gender'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        
        # --- CALIBRATION LAYER: Linear Shift ---
        bp_shift = 8  
        
        raw_ap_hi = int(request.form.get('ap_hi'))
        raw_ap_lo = int(request.form.get('ap_lo'))
        
        # Shifted values for the AI model
        ap_hi = raw_ap_hi - bp_shift
        ap_lo = raw_ap_lo - bp_shift
        # ---------------------------------------

        chol = int(request.form.get('cholesterol'))
        gluc = int(request.form.get('gluc'))
        smoke = int(request.form.get('smoke'))
        alco = int(request.form.get('alco'))
        active = int(request.form.get('active'))

        # 2. Heuristic Clinical Observations (Using RAW values for accuracy)
        warnings = []
        bmi = weight / ((height / 100) ** 2)
        if bmi >= 25: warnings.append(f"High BMI ({round(bmi, 1)})")
        if chol >= 2: warnings.append("Elevated Cholesterol")
        if gluc >= 2: warnings.append("Elevated Glucose")
        if smoke == 1: warnings.append("Smoking Habit")
        if active == 0: warnings.append("Sedentary Lifestyle")
        
        # to make a well rounded prediction
        if raw_ap_hi >= 135 or raw_ap_lo >= 84: 
            warnings.append("Elevated Blood Pressure")

        # 3. Model Prediction (Using SHIFTED values)
        feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        input_data = pd.DataFrame([[age_days, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active]], columns=feature_names)
        
        # Get the prediction from the 70k-patient model
        prediction = model.predict(input_data)[0]

        # 4. Result Formatting
        status_class = "high-risk" if prediction == 1 else "low-risk"
        
        if prediction == 0 and warnings:
            safety_note = "Your overall risk is low based on the model, but keep in mind the specific clinical observations listed below."
        elif prediction == 0:
            safety_note = "All measured biometric markers are within standard ranges."
        else:
            safety_note = "High risk detected. Please consult a healthcare professional regarding these results."

        return render_template('result.html', 
                               prediction=int(prediction), 
                               status_class=status_class, 
                               safety_note=safety_note,
                               warnings=warnings)

    except Exception as e:
        return f"System Error: {str(e)}"

# --- Ensure Database Initialization ---
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False)