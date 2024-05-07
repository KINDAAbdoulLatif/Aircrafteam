from flask import Flask, render_template, request,url_for
from werkzeug.utils import secure_filename
import os
from fct import *
from fnct import *
# from fl import *



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/image')
def image():
    return render_template("image.html")

@app.route("/traitement", methods=["POST", "GET"])
def traitement():
    if request.method=="POST":
        file = request.files['image']
        

        file_name = secure_filename(filename=file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        image_path  = 'static/img/' + file_name
        # Load the YOLOv5 model
        

        class_name = predict_image(image_path=image_path)
        modele_name = predict_res(image_path=image_path)
        
        # result = predict_yolov8_classification(image_path=image_path)
        c = classify_image(image_path)
        # prediction = predict_aircraft(image_path)
        return render_template("traitement.html", image_path=image_path, class_name=class_name, modele_name= modele_name,  c=c)
    
if __name__ == '__main__':
    app.run(debug=True)