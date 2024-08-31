#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data = pd.read_csv(r'C:\Users\97150\Downloads\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv')

# Drop columns that are not needed
data = data.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 
                           'examide', 'citoglipton', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
                           'metformin-pioglitazone', 'max_glu_serum', 'A1Cresult'])

# Handle missing values
data = data.replace('?', pd.NA)
data = data.dropna()

# Define features and target
features = data.drop(columns=['readmitted'])
target = data['readmitted']

# Convert target to binary (0: No readmission, 1: Readmission within 30 days)
target = target.apply(lambda x: 1 if x == '<30' else 0)

# Define numerical and categorical columns
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = features.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=10, validation_split=0.2)

# Evaluate the model
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Print metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Precision: {precision_score(y_test, y_pred):.2f}')
print(f'Recall: {recall_score(y_test, y_pred):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.2f}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba):.2f}')

# Save preprocessor and model
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
model.save('diabetes_readmission_model.h5')


# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import Scrollbar
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define mock preprocessor and model
def create_mock_preprocessor():
    categorical_features = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 
                            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                            'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
                            'glyburide-metformin', 'glipizide-metformin', 'change', 'diabetesMed']
    
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return preprocessor

def create_mock_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize and fit preprocessor and model once
preprocessor = create_mock_preprocessor()

mock_data = pd.DataFrame({
    'admission_type_id': [1],
    'discharge_disposition_id': [1],
    'admission_source_id': [7],
    'time_in_hospital': [2],
    'num_lab_procedures': [10],
    'num_procedures': [0],
    'num_medications': [1],
    'number_outpatient': [0],
    'number_emergency': [0],
    'number_inpatient': [0],
    'number_diagnoses': [2],
    'race': ['Caucasian'],
    'gender': ['Male'],
    'age': ['[40-50)'],
    'diag_1': ['250.00'],
    'diag_2': ['None'],
    'diag_3': ['None'],
    'metformin': ['Yes'],
    'repaglinide': ['No'],
    'nateglinide': ['No'],
    'chlorpropamide': ['No'],
    'glimepiride': ['No'],
    'acetohexamide': ['No'],
    'glipizide': ['No'],
    'glyburide': ['No'],
    'tolbutamide': ['No'],
    'pioglitazone': ['No'],
    'rosiglitazone': ['No'],
    'acarbose': ['No'],
    'miglitol': ['No'],
    'troglitazone': ['No'],
    'tolazamide': ['No'],
    'insulin': ['No'],
    'glyburide-metformin': ['No'],
    'glipizide-metformin': ['No'],
    'change': ['No'],
    'diabetesMed': ['Yes']
})

X = mock_data
preprocessor.fit(X)

y = np.array([0])  # Mock target variable

X_transformed = preprocessor.transform(X)
input_dim = X_transformed.shape[1]

model = create_mock_model(input_dim)
model.fit(X_transformed, y, epochs=5)

def predict():
    try:
        input_data = {
            'admission_type_id': [admission_type_id_var.get()],
            'discharge_disposition_id': [discharge_disposition_id_var.get()],
            'admission_source_id': [admission_source_id_var.get()],
            'time_in_hospital': [time_in_hospital_var.get()],
            'num_lab_procedures': [num_lab_procedures_var.get()],
            'num_procedures': [num_procedures_var.get()],
            'num_medications': [num_medications_var.get()],
            'number_outpatient': [number_outpatient_var.get()],
            'number_emergency': [number_emergency_var.get()],
            'number_inpatient': [number_inpatient_var.get()],
            'number_diagnoses': [number_diagnoses_var.get()],
            'race': [race_var.get()],
            'gender': [gender_var.get()],
            'age': [age_var.get()],
            'diag_1': [diag_1_var.get()],
            'diag_2': [diag_2_var.get()],
            'diag_3': [diag_3_var.get()],
            'metformin': [metformin_var.get()],
            'repaglinide': [repaglinide_var.get()],
            'nateglinide': [nateglinide_var.get()],
            'chlorpropamide': [chlorpropamide_var.get()],
            'glimepiride': [glimepiride_var.get()],
            'acetohexamide': [acetohexamide_var.get()],
            'glipizide': [glipizide_var.get()],
            'glyburide': [glyburide_var.get()],
            'tolbutamide': [tolbutamide_var.get()],
            'pioglitazone': [pioglitazone_var.get()],
            'rosiglitazone': [rosiglitazone_var.get()],
            'acarbose': [acarbose_var.get()],
            'miglitol': [miglitol_var.get()],
            'troglitazone': [troglitazone_var.get()],
            'tolazamide': [tolazamide_var.get()],
            'insulin': [insulin_var.get()],
            'glyburide-metformin': [glyburide_metformin_var.get()],
            'glipizide-metformin': [glipizide_metformin_var.get()],
            'change': [change_var.get()],
            'diabetesMed': [diabetesMed_var.get()]
        }

        input_df = pd.DataFrame(input_data)

        preprocessed_input = preprocessor.transform(input_df)

        prediction_proba = model.predict(preprocessed_input).flatten()
        prediction = (prediction_proba > 0.5).astype(int)

        # Create a new window for the result
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Result")
        result_window.geometry("300x150")
        result_window.config(bg='#f0f0f0')

        # Add some padding and styling
        result_label = tk.Label(result_window, text=f'Prediction Probability: {prediction_proba[0]:.2f}', 
                                font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_label.pack(pady=10)

        result_text = "Readmission within 30 days" if prediction[0] == 1 else "No readmission"
        result_label_text = tk.Label(result_window, text=f'Prediction: {result_text}', 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_label_text.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")
        print(f"Prediction Error: {e}")

# Create the main window
root = tk.Tk()
root.title("Diabetes Readmission Prediction")
root.geometry("700x800")
root.config(bg='#e0e0e0')

# Create a frame for the form with scrollbar
form_frame = tk.Frame(root, padx=20, pady=20, bg='#e0e0e0')
form_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a canvas and a scrollbar
canvas = tk.Canvas(form_frame, bg='#e0e0e0')
scrollbar = tk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

# Create a frame to hold the form, which will be placed inside the canvas
form_container = tk.Frame(canvas, bg='#e0e0e0')
form_container.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=form_container, anchor="nw")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add the input fields
admission_type_id_var = tk.StringVar(value='1')
discharge_disposition_id_var = tk.StringVar(value='1')
admission_source_id_var = tk.StringVar(value='7')
time_in_hospital_var = tk.StringVar(value='3')
num_lab_procedures_var = tk.StringVar(value='41')
num_procedures_var = tk.StringVar(value='0')
num_medications_var = tk.StringVar(value='1')
number_outpatient_var = tk.StringVar(value='0')
number_emergency_var = tk.StringVar(value='0')
number_inpatient_var = tk.StringVar(value='0')
number_diagnoses_var = tk.StringVar(value='9')
race_var = tk.StringVar(value='Caucasian')
gender_var = tk.StringVar(value='Female')
age_var = tk.StringVar(value='[50-60)')
diag_1_var = tk.StringVar(value='250.83')
diag_2_var = tk.StringVar(value='276')
diag_3_var = tk.StringVar(value='250.13')
metformin_var = tk.StringVar(value='No')
repaglinide_var = tk.StringVar(value='No')
nateglinide_var = tk.StringVar(value='No')
chlorpropamide_var = tk.StringVar(value='No')
glimepiride_var = tk.StringVar(value='No')
acetohexamide_var = tk.StringVar(value='No')
glipizide_var = tk.StringVar(value='No')
glyburide_var = tk.StringVar(value='No')
tolbutamide_var = tk.StringVar(value='No')
pioglitazone_var = tk.StringVar(value='No')
rosiglitazone_var = tk.StringVar(value='No')
acarbose_var = tk.StringVar(value='No')
miglitol_var = tk.StringVar(value='No')
troglitazone_var = tk.StringVar(value='No')
tolazamide_var = tk.StringVar(value='No')
insulin_var = tk.StringVar(value='No')
glyburide_metformin_var = tk.StringVar(value='No')
glipizide_metformin_var = tk.StringVar(value='No')
change_var = tk.StringVar(value='No')
diabetesMed_var = tk.StringVar(value='Yes')

input_fields = {
    "Admission Type ID": admission_type_id_var,
    "Discharge Disposition ID": discharge_disposition_id_var,
    "Admission Source ID": admission_source_id_var,
    "Time in Hospital": time_in_hospital_var,
    "Number of Lab Procedures": num_lab_procedures_var,
    "Number of Procedures": num_procedures_var,
    "Number of Medications": num_medications_var,
    "Number of Outpatient": number_outpatient_var,
    "Number of Emergency": number_emergency_var,
    "Number of Inpatient": number_inpatient_var,
    "Number of Diagnoses": number_diagnoses_var,
    "Race": race_var,
    "Gender": gender_var,
    "Age": age_var,
    "Diagnosis 1": diag_1_var,
    "Diagnosis 2": diag_2_var,
    "Diagnosis 3": diag_3_var,
    "Metformin": metformin_var,
    "Repaglinide": repaglinide_var,
    "Nateglinide": nateglinide_var,
    "Chlorpropamide": chlorpropamide_var,
    "Glimepiride": glimepiride_var,
    "Acetohexamide": acetohexamide_var,
    "Glipizide": glipizide_var,
    "Glyburide": glyburide_var,
    "Tolbutamide": tolbutamide_var,
    "Pioglitazone": pioglitazone_var,
    "Rosiglitazone": rosiglitazone_var,
    "Acarbose": acarbose_var,
    "Miglitol": miglitol_var,
    "Troglitazone": troglitazone_var,
    "Tolazamide": tolazamide_var,
    "Insulin": insulin_var,
    "Glyburide Metformin": glyburide_metformin_var,
    "Glipizide Metformin": glipizide_metformin_var,
    "Change": change_var,
    "Diabetes Med": diabetesMed_var
}

for idx, (label_text, var) in enumerate(input_fields.items()):
    tk.Label(form_container, text=label_text, font=("Arial", 12), bg='#e0e0e0').grid(row=idx, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(form_container, textvariable=var, font=("Arial", 12)).grid(row=idx, column=1, padx=10, pady=5)

# Button to make prediction
predict_button = tk.Button(form_container, text="Predict", command=predict, font=("Arial", 12, "bold"), bg='#4CAF50', fg='white')
predict_button.grid(row=len(input_fields) + 1, column=0, columnspan=2, pady=20)

root.mainloop()


# In[7]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import Scrollbar
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define mock preprocessor and model
def create_mock_preprocessor():
    categorical_features = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 
                            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                            'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
                            'glyburide-metformin', 'glipizide-metformin', 'change', 'diabetesMed']
    
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return preprocessor

def create_mock_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # Adjusted input shape
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def mock_fit(preprocessor, model):
    mock_data = pd.DataFrame({
        'admission_type_id': [1],
        'discharge_disposition_id': [1],
        'admission_source_id': [7],
        'time_in_hospital': [2],
        'num_lab_procedures': [30],
        'num_procedures': [0],
        'num_medications': [1],
        'number_outpatient': [0],
        'number_emergency': [0],
        'number_inpatient': [0],
        'number_diagnoses': [1],
        'race': ['Caucasian'],
        'gender': ['Female'],
        'age': ['[50-60)'],
        'diag_1': ['250.00'],
        'diag_2': ['None'],
        'diag_3': ['None'],
        'metformin': ['No'],
        'repaglinide': ['No'],
        'nateglinide': ['No'],
        'chlorpropamide': ['No'],
        'glimepiride': ['No'],
        'acetohexamide': ['No'],
        'glipizide': ['No'],
        'glyburide': ['No'],
        'tolbutamide': ['No'],
        'pioglitazone': ['No'],
        'rosiglitazone': ['No'],
        'acarbose': ['No'],
        'miglitol': ['No'],
        'troglitazone': ['No'],
        'tolazamide': ['No'],
        'insulin': ['No'],
        'glyburide-metformin': ['No'],
        'glipizide-metformin': ['No'],
        'change': ['No'],
        'diabetesMed': ['No']
    })
    
    X = mock_data
    preprocessor.fit(X)
    
    y = np.array([0])  # Mock target variable
    
    X_transformed = preprocessor.transform(X)
    input_dim = X_transformed.shape[1]  # Get number of features after transformation
    
    model = create_mock_model(input_dim)
    model.fit(X_transformed, y)

def predict():
    try:
        input_data = {
            'admission_type_id': [admission_type_id_var.get()],
            'discharge_disposition_id': [discharge_disposition_id_var.get()],
            'admission_source_id': [admission_source_id_var.get()],
            'time_in_hospital': [time_in_hospital_var.get()],
            'num_lab_procedures': [num_lab_procedures_var.get()],
            'num_procedures': [num_procedures_var.get()],
            'num_medications': [num_medications_var.get()],
            'number_outpatient': [number_outpatient_var.get()],
            'number_emergency': [number_emergency_var.get()],
            'number_inpatient': [number_inpatient_var.get()],
            'number_diagnoses': [number_diagnoses_var.get()],
            'race': [race_var.get()],
            'gender': [gender_var.get()],
            'age': [age_var.get()],
            'diag_1': [diag_1_var.get()],
            'diag_2': [diag_2_var.get()],
            'diag_3': [diag_3_var.get()],
            'metformin': [metformin_var.get()],
            'repaglinide': [repaglinide_var.get()],
            'nateglinide': [nateglinide_var.get()],
            'chlorpropamide': [chlorpropamide_var.get()],
            'glimepiride': [glimepiride_var.get()],
            'acetohexamide': [acetohexamide_var.get()],
            'glipizide': [glipizide_var.get()],
            'glyburide': [glyburide_var.get()],
            'tolbutamide': [tolbutamide_var.get()],
            'pioglitazone': [pioglitazone_var.get()],
            'rosiglitazone': [rosiglitazone_var.get()],
            'acarbose': [acarbose_var.get()],
            'miglitol': [miglitol_var.get()],
            'troglitazone': [troglitazone_var.get()],
            'tolazamide': [tolazamide_var.get()],
            'insulin': [insulin_var.get()],
            'glyburide-metformin': [glyburide_metformin_var.get()],
            'glipizide-metformin': [glipizide_metformin_var.get()],
            'change': [change_var.get()],
            'diabetesMed': [diabetesMed_var.get()]
        }

        input_df = pd.DataFrame(input_data)
        
        preprocessed_input = preprocessor.transform(input_df)
        
        input_dim = preprocessed_input.shape[1]
        model = create_mock_model(input_dim)
        mock_fit(preprocessor, model)
        
        prediction_proba = model.predict(preprocessed_input).flatten()
        prediction = (prediction_proba > 0.5).astype(int)
        
        # Create a new window for the result
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Result")
        result_window.geometry("300x150")
        result_window.config(bg='#f0f0f0')
        
        # Add some padding and styling
        result_label = tk.Label(result_window, text=f'Prediction Probability: {prediction_proba[0]:.2f}', 
                                font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_label.pack(pady=10)
        
        result_text = "Readmission within 30 days" if prediction[0] == 1 else "No readmission"
        result_label_text = tk.Label(result_window, text=f'Prediction: {result_text}', 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_label_text.pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")
        print(f"Prediction Error: {e}")

# Create the main window
root = tk.Tk()
root.title("Diabetes Readmission Prediction")
root.geometry("700x800")
root.config(bg='#e0e0e0')

# Create a frame for the form with scrollbar
form_frame = tk.Frame(root, padx=20, pady=20, bg='#e0e0e0')
form_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a canvas and a scrollbar
canvas = tk.Canvas(form_frame, bg='#e0e0e0')
scrollbar = tk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

# Create a frame to hold the form, which will be placed inside the canvas
form_container = tk.Frame(canvas, bg='#e0e0e0')
form_container.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=form_container, anchor="nw")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add the input fields
admission_type_id_var = tk.StringVar(value='1')
discharge_disposition_id_var = tk.StringVar(value='1')
admission_source_id_var = tk.StringVar(value='7')
time_in_hospital_var = tk.StringVar(value='2')
num_lab_procedures_var = tk.StringVar(value='30')
num_procedures_var = tk.StringVar(value='0')
num_medications_var = tk.StringVar(value='1')
number_outpatient_var = tk.StringVar(value='0')
number_emergency_var = tk.StringVar(value='0')
number_inpatient_var = tk.StringVar(value='0')
number_diagnoses_var = tk.StringVar(value='1')
race_var = tk.StringVar(value='Caucasian')
gender_var = tk.StringVar(value='Female')
age_var = tk.StringVar(value='[50-60)')
diag_1_var = tk.StringVar(value='250.00')
diag_2_var = tk.StringVar(value='None')
diag_3_var = tk.StringVar(value='None')
metformin_var = tk.StringVar(value='No')
repaglinide_var = tk.StringVar(value='No')
nateglinide_var = tk.StringVar(value='No')
chlorpropamide_var = tk.StringVar(value='No')
glimepiride_var = tk.StringVar(value='No')
acetohexamide_var = tk.StringVar(value='No')
glipizide_var = tk.StringVar(value='No')
glyburide_var = tk.StringVar(value='No')
tolbutamide_var = tk.StringVar(value='No')
pioglitazone_var = tk.StringVar(value='No')
rosiglitazone_var = tk.StringVar(value='No')
acarbose_var = tk.StringVar(value='No')
miglitol_var = tk.StringVar(value='No')
troglitazone_var = tk.StringVar(value='No')
tolazamide_var = tk.StringVar(value='No')
insulin_var = tk.StringVar(value='No')
glyburide_metformin_var = tk.StringVar(value='No')
glipizide_metformin_var = tk.StringVar(value='No')
change_var = tk.StringVar(value='No')
diabetesMed_var = tk.StringVar(value='No')

# Define and place labels and entry fields
fields = [
    'Admission Type ID', 'Discharge Disposition ID', 'Admission Source ID',
    'Time in Hospital', 'Number of Lab Procedures', 'Number of Procedures',
    'Number of Medications', 'Number of Outpatient Visits', 'Number of Emergency Visits',
    'Number of Inpatient Visits', 'Number of Diagnoses', 'Race', 'Gender', 'Age',
    'Diagnosis 1', 'Diagnosis 2', 'Diagnosis 3', 'Metformin', 'Repaglinide',
    'Nateglinide', 'Chlorpropamide', 'Glimepiride', 'Acetohexamide', 'Glipizide',
    'Glyburide', 'Tolbutamide', 'Pioglitazone', 'Rosiglitazone', 'Acarbose',
    'Miglitol', 'Troglitazone', 'Tolazamide', 'Insulin', 'Glyburide-Metformin',
    'Glipizide-Metformin', 'Change', 'Diabetes Medication'
]

vars = [
    admission_type_id_var, discharge_disposition_id_var, admission_source_id_var,
    time_in_hospital_var, num_lab_procedures_var, num_procedures_var,
    num_medications_var, number_outpatient_var, number_emergency_var,
    number_inpatient_var, number_diagnoses_var, race_var, gender_var, age_var,
    diag_1_var, diag_2_var, diag_3_var, metformin_var, repaglinide_var,
    nateglinide_var, chlorpropamide_var, glimepiride_var, acetohexamide_var,
    glipizide_var, glyburide_var, tolbutamide_var, pioglitazone_var, rosiglitazone_var,
    acarbose_var, miglitol_var, troglitazone_var, tolazamide_var, insulin_var,
    glyburide_metformin_var, glipizide_metformin_var, change_var, diabetesMed_var
]

for i, field in enumerate(fields):
    label = tk.Label(form_container, text=field, bg='#e0e0e0', anchor='w')
    label.grid(row=i, column=0, padx=5, pady=5, sticky='w')
    entry = tk.Entry(form_container, textvariable=vars[i])
    entry.grid(row=i, column=1, padx=5, pady=5, sticky='ew')

# Add a button to trigger prediction
predict_button = tk.Button(form_container, text="Predict", command=predict, bg='#4CAF50', fg='white')
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

root.mainloop()


# In[6]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import Scrollbar
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define mock preprocessor and model
def create_mock_preprocessor():
    categorical_features = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 
                            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                            'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
                            'glyburide-metformin', 'glipizide-metformin', 'change', 'diabetesMed']
    
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return preprocessor

def create_mock_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # Adjusted input shape
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def mock_fit(preprocessor, model):
    mock_data = pd.DataFrame({
        'admission_type_id': [1],
        'discharge_disposition_id': [1],
        'admission_source_id': [7],
        'time_in_hospital': [2],
        'num_lab_procedures': [30],
        'num_procedures': [0],
        'num_medications': [1],
        'number_outpatient': [0],
        'number_emergency': [0],
        'number_inpatient': [0],
        'number_diagnoses': [1],
        'race': ['Caucasian'],
        'gender': ['Female'],
        'age': ['[50-60)'],
        'diag_1': ['250.00'],
        'diag_2': ['None'],
        'diag_3': ['None'],
        'metformin': ['No'],
        'repaglinide': ['No'],
        'nateglinide': ['No'],
        'chlorpropamide': ['No'],
        'glimepiride': ['No'],
        'acetohexamide': ['No'],
        'glipizide': ['No'],
        'glyburide': ['No'],
        'tolbutamide': ['No'],
        'pioglitazone': ['No'],
        'rosiglitazone': ['No'],
        'acarbose': ['No'],
        'miglitol': ['No'],
        'troglitazone': ['No'],
        'tolazamide': ['No'],
        'insulin': ['No'],
        'glyburide-metformin': ['No'],
        'glipizide-metformin': ['No'],
        'change': ['No'],
        'diabetesMed': ['No']
    })
    
    X = mock_data
    preprocessor.fit(X)
    
    y = np.array([0])  # Mock target variable
    
    X_transformed = preprocessor.transform(X)
    input_dim = X_transformed.shape[1]  # Get number of features after transformation
    
    model = create_mock_model(input_dim)
    model.fit(X_transformed, y)

def predict():
    try:
        input_data = {
            'admission_type_id': [admission_type_id_var.get()],
            'discharge_disposition_id': [discharge_disposition_id_var.get()],
            'admission_source_id': [admission_source_id_var.get()],
            'time_in_hospital': [time_in_hospital_var.get()],
            'num_lab_procedures': [num_lab_procedures_var.get()],
            'num_procedures': [num_procedures_var.get()],
            'num_medications': [num_medications_var.get()],
            'number_outpatient': [number_outpatient_var.get()],
            'number_emergency': [number_emergency_var.get()],
            'number_inpatient': [number_inpatient_var.get()],
            'number_diagnoses': [number_diagnoses_var.get()],
            'race': [race_var.get()],
            'gender': [gender_var.get()],
            'age': [age_var.get()],
            'diag_1': [diag_1_var.get()],
            'diag_2': [diag_2_var.get()],
            'diag_3': [diag_3_var.get()],
            'metformin': [metformin_var.get()],
            'repaglinide': [repaglinide_var.get()],
            'nateglinide': [nateglinide_var.get()],
            'chlorpropamide': [chlorpropamide_var.get()],
            'glimepiride': [glimepiride_var.get()],
            'acetohexamide': [acetohexamide_var.get()],
            'glipizide': [glipizide_var.get()],
            'glyburide': [glyburide_var.get()],
            'tolbutamide': [tolbutamide_var.get()],
            'pioglitazone': [pioglitazone_var.get()],
            'rosiglitazone': [rosiglitazone_var.get()],
            'acarbose': [acarbose_var.get()],
            'miglitol': [miglitol_var.get()],
            'troglitazone': [troglitazone_var.get()],
            'tolazamide': [tolazamide_var.get()],
            'insulin': [insulin_var.get()],
            'glyburide-metformin': [glyburide_metformin_var.get()],
            'glipizide-metformin': [glipizide_metformin_var.get()],
            'change': [change_var.get()],
            'diabetesMed': [diabetesMed_var.get()]
        }

        input_df = pd.DataFrame(input_data)
        
        preprocessed_input = preprocessor.transform(input_df)
        
        input_dim = preprocessed_input.shape[1]
        model = create_mock_model(input_dim)
        mock_fit(preprocessor, model)
        
        prediction_proba = model.predict(preprocessed_input).flatten()
        prediction = (prediction_proba > 0.5).astype(int)
        
        # Create a new window for the result
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Result")
        result_window.geometry("300x150")
        result_window.config(bg='#f0f0f0')
        
        # Add some padding and styling
        result_label = tk.Label(result_window, text=f'Prediction Probability: {prediction_proba[0]:.2f}', 
                                font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_label.pack(pady=10)
        
        result_text = "Readmission within 30 days" if prediction[0] == 1 else "No readmission"
        result_label_text = tk.Label(result_window, text=f'Prediction: {result_text}', 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0')
        result_label_text.pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")
        print(f"Prediction Error: {e}")

# Create the main window
root = tk.Tk()
root.title("Diabetes Readmission Prediction")
root.geometry("700x800")
root.config(bg='#e0e0e0')

# Create a frame for the form with scrollbar
form_frame = tk.Frame(root, padx=20, pady=20, bg='#e0e0e0')
form_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a canvas and a scrollbar
canvas = tk.Canvas(form_frame, bg='#e0e0e0')
scrollbar = tk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

# Create a frame to hold the form, which will be placed inside the canvas
form_container = tk.Frame(canvas, bg='#e0e0e0')
form_container.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=form_container, anchor="nw")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add the input fields
admission_type_id_var = tk.StringVar(value='1')
discharge_disposition_id_var = tk.StringVar(value='1')
admission_source_id_var = tk.StringVar(value='7')
time_in_hospital_var = tk.StringVar(value='2')
num_lab_procedures_var = tk.StringVar(value='30')
num_procedures_var = tk.StringVar(value='0')
num_medications_var = tk.StringVar(value='1')
number_outpatient_var = tk.StringVar(value='0')
number_emergency_var = tk.StringVar(value='0')
number_inpatient_var = tk.StringVar(value='0')
number_diagnoses_var = tk.StringVar(value='1')
race_var = tk.StringVar(value='Caucasian')
gender_var = tk.StringVar(value='Female')
age_var = tk.StringVar(value='[50-60)')
diag_1_var = tk.StringVar(value='250.00')
diag_2_var = tk.StringVar(value='None')
diag_3_var = tk.StringVar(value='None')
metformin_var = tk.StringVar(value='No')
repaglinide_var = tk.StringVar(value='No')
nateglinide_var = tk.StringVar(value='No')
chlorpropamide_var = tk.StringVar(value='No')
glimepiride_var = tk.StringVar(value='No')
acetohexamide_var = tk.StringVar(value='No')
glipizide_var = tk.StringVar(value='No')
glyburide_var = tk.StringVar(value='No')
tolbutamide_var = tk.StringVar(value='No')
pioglitazone_var = tk.StringVar(value='No')
rosiglitazone_var = tk.StringVar(value='No')
acarbose_var = tk.StringVar(value='No')
miglitol_var = tk.StringVar(value='No')
troglitazone_var = tk.StringVar(value='No')
tolazamide_var = tk.StringVar(value='No')
insulin_var = tk.StringVar(value='No')
glyburide_metformin_var = tk.StringVar(value='No')
glipizide_metformin_var = tk.StringVar(value='No')
change_var = tk.StringVar(value='No')
diabetesMed_var = tk.StringVar(value='No')

# Define and place labels and entry fields
fields = [
    'Admission Type ID', 'Discharge Disposition ID', 'Admission Source ID',
    'Time in Hospital', 'Number of Lab Procedures', 'Number of Procedures',
    'Number of Medications', 'Number of Outpatient Visits', 'Number of Emergency Visits',
    'Number of Inpatient Visits', 'Number of Diagnoses', 'Race', 'Gender', 'Age',
    'Diagnosis 1', 'Diagnosis 2', 'Diagnosis 3', 'Metformin', 'Repaglinide',
    'Nateglinide', 'Chlorpropamide', 'Glimepiride', 'Acetohexamide', 'Glipizide',
    'Glyburide', 'Tolbutamide', 'Pioglitazone', 'Rosiglitazone', 'Acarbose',
    'Miglitol', 'Troglitazone', 'Tolazamide', 'Insulin', 'Glyburide-Metformin',
    'Glipizide-Metformin', 'Change', 'Diabetes Medication'
]

vars = [
    admission_type_id_var, discharge_disposition_id_var, admission_source_id_var,
    time_in_hospital_var, num_lab_procedures_var, num_procedures_var,
    num_medications_var, number_outpatient_var, number_emergency_var,
    number_inpatient_var, number_diagnoses_var, race_var, gender_var, age_var,
    diag_1_var, diag_2_var, diag_3_var, metformin_var, repaglinide_var,
    nateglinide_var, chlorpropamide_var, glimepiride_var, acetohexamide_var,
    glipizide_var, glyburide_var, tolbutamide_var, pioglitazone_var, rosiglitazone_var,
    acarbose_var, miglitol_var, troglitazone_var, tolazamide_var, insulin_var,
    glyburide_metformin_var, glipizide_metformin_var, change_var, diabetesMed_var
]

for i, field in enumerate(fields):
    label = tk.Label(form_container, text=field, bg='#e0e0e0', anchor='w')
    label.grid(row=i, column=0, padx=5, pady=5, sticky='w')
    entry = tk.Entry(form_container, textvariable=vars[i])
    entry.grid(row=i, column=1, padx=5, pady=5, sticky='ew')

# Add a button to trigger prediction
predict_button = tk.Button(form_container, text="Predict", command=predict, bg='#4CAF50', fg='white')
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

root.mainloop()

