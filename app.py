from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin #CORS: Allows frontend apps (e.g., React) to talk to your Flask backend.
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

import pymongo
from pymongo import MongoClient
from flask_pymongo import PyMongo
import json
from werkzeug.utils import secure_filename
import base64
import datetime
from flask import redirect, url_for

client = MongoClient("mongodb://localhost:27017")
mydb = client["Kidney_tumor"]

collection = mydb.Personal_details

os.putenv('LANG', 'en_US.UTF-8') #These ensure app handles characters and files properly in most systems
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app) #For frontend backend communication


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg" # where uploaded image will be saved.
        self.classifier = PredictionPipeline(self.filename) # The model pipeline object
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('home.html')

@app.route("/details", methods=['GET'])
@cross_origin()
def ind():
    return render_template('index.html')

@app.route("/search", methods=['GET', 'POST'])
@cross_origin()
def search():
    if request.method == 'POST':
        fname = request.form.get('fname', '').strip()
        lname = request.form.get('lname', '').strip()
        
        query = {}
        if fname:
            query['fname'] = {'$regex': fname, '$options': 'i'}
        if lname:
            query['lname'] = {'$regex': lname, '$options': 'i'}
        
        patients = list(collection.find(query))
        return render_template('search.html', patients=patients) #left side patients is variable name for Jinja 2 HTML and right side patients is variable name for Python where mongo db data is found.
    
    return render_template('search.html')
@app.route("/delete/<string:patient_id>", methods=['POST'])
@cross_origin()
def delete_patient(patient_id):
    # Convert string ID to MongoDB ObjectId
    from bson.objectid import ObjectId
    collection.delete_one({'_id': ObjectId(patient_id)})
    return redirect(url_for('search'))  # Redirect back to search page


@app.route("/create", methods=['POST'])
@cross_origin()
def create():
    # Get form data with default "No record" for empty fields
    fname = request.form.get('fname', 'No record').strip() or 'No record'
    lname = request.form.get('lname', 'No record').strip() or 'No record'
    age = request.form.get('age', 'No record')
    gender = request.form.get('gender', 'No record').strip() or 'No record'
    contact = request.form.get('contact', 'No record').strip() or 'No record'
    weight = request.form.get('weight', 'No record')
    symptoms = request.form.get('symptoms', 'No record').strip() or 'No record'
    diagnosis = request.form.get('diagnosis', 'No record').strip() or 'No record'
    classification = request.form.get('classification', 'No record').strip() or 'No record'

    # Handle numeric fields specially
    try:
        age = int(age) if age != 'No record' else 'No record'
    except ValueError:
        age = 'No record'
        
    try:
        weight = int(weight) if weight != 'No record' else 'No record'
    except ValueError:
        weight = 'No record'

    # Handle image upload
    image_data = 'No record'
    if 'upload' in request.files:
        file = request.files['upload']
        if file.filename != '':
            image_data = base64.b64encode(file.read()).decode('utf-8')

    # Insert into database
    collection.insert_one({
        "fname": fname,
        "lname": lname,
        "age": age,
        "gender": gender,
        "contact": contact,
        "weight": weight,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "image_data": image_data,
        "created_at": datetime.datetime.utcnow(),
        "classification": classification
    })
    
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__": # It runs only when we run this file directly. If we import this file from any other file it won't run because the name would not be __main__. This is to prevenmty accidents
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080,debug=True) #for AWS

