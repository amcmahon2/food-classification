from flask import Flask, request, jsonify, render_template_string
from predict import estimate_nutrition, predict_cnn, get_calories
import os
import json
import pickle
from torchvision import transforms
import torch
import time
from PIL import Image

app = Flask(__name__)

#load USDA data
with open('../data/USDA_db.json') as f:
    usda_data = json.load(f)

#load substitution data
with open('../data/substitutions.json') as f:
    substitutions = json.load(f)

#load ORB feature database
# with open('../models/feature_db.pkl', 'rb') as f:
#     feature_db = pickle.load(f)

#load CNN model
from torchvision.models import resnet18
model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 101)  # adjust if needed
model.load_state_dict(torch.load('../models/food_classifier.pth', map_location='cpu'))
model.eval()

# Transform for CNN model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filepath = os.path.join('uploads', "test_image.png")
            image.save(filepath)
            try:
                # start = time.time()
                # orb_prediction = predict_orb(filepath, feature_db)
                # print("ORB time to compute:", time.time() - start)

                start = time.time()
                cnn_prediction, cnn_prediction_top5 = predict_cnn(filepath, model, transform)
                print("CNN time to compute:", time.time() - start)

                #get calorie info from USDA JSON
                calories = get_calories(usda_data, cnn_prediction)

                #suggest substitutes
                substitutions_map = substitutions["General Substitutions"][0]
                subs = substitutions_map.get(cnn_prediction.lower(), ["none"])
                print(f"Substitutes for {cnn_prediction}: {subs}")
                
                #if no calorie data is present for CNN prediction, check if subsitutes ar present in USDA db
                if(calories == 0):
                    calories_sub0 = get_calories(usda_data, subs[0])
                    calories_sub1 = get_calories(usda_data, subs[1])
                    calories_sub2 = get_calories(usda_data, subs[2])
                    if(calories_sub0 != 0):
                        calories = calories_sub0
                    elif(calories_sub1 != 0):
                        calories = calories_sub1
                    elif(calories_sub2 != 0):
                        calories = calories_sub2

                #extract calorie amount from main food (or one of 3 subs if needed)
                nutrient_info = estimate_nutrition("uploads/test_image.png", cnn_prediction, calories, 0.5)

            except Exception as e: #if an error is thrown, give pop-up bx and forc user back to homepage
                return render_template_string(f''' <script>alert("Error: {str(e)}"); window.location.href = "/";</script>''')
            return f'''
                <html>
                <head>
                    <title>Food Image Classification with Nutrition Analysis and Ingredient Substitution</title>
                    <style>
                        body {{
                            background-color: #e6ffe6;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            padding: 40px;
                        }}
                        .container {{
                            background-color: #ffffff;
                            border-radius: 10px;
                            padding: 30px;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                            max-width: 800px;
                            margin: auto;
                        }}
                        h1 {{
                            text-align: center;
                            color: #2e7d32;
                            margin-bottom: 30px;
                        }}
                        .result {{
                            display: flex;
                            gap: 20px;
                            align-items: flex-start;
                            justify-content: center;
                        }}
                        .nutrition {{
                            background-color: #f0fff0;
                            padding: 15px;
                            border-left: 5px solid #4caf50;
                            border-radius: 5px;
                            flex: 1;
                        }}
                        .nutrition table {{
                            width: 100%;
                            border-collapse: collapse;
                        }}
                        .nutrition table, .nutrition th, .nutrition td {{
                            border: 1px solid #ccc;
                        }}
                        .nutrition th, .nutrition td {{
                            padding: 8px 12px;
                            text-align: left;
                        }}
                        .preview {{
                            flex: 1;
                        }}
                        .preview img {{
                            width: 100%;
                            max-width: 350px;
                            border-radius: 8px;
                            border: 1px solid #ccc;
                        }}
                        .btn {{
                            display: inline-block;
                            padding: 10px 20px;
                            margin-top: 30px;
                            background-color: #4caf50;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                            text-align: center;
                        }}
                        .footer {{
                            margin-top: 40px;
                            text-align: center;
                            font-size: 0.9em;
                            color: #555;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Prediction, Nutrition, and Substitutes</h1>
                        <h2>Prediction: {cnn_prediction}</h2>
                        <div class="result">
                            <div class="nutrition">
                                <h3>Nutrition Info:</h3>
                                {nutrient_info}
                                <h3 style="margin-top: 20px;">Substitutes:</h3>
                                <ul>
                                    <li>{subs[0]}</li>
                                    <li>{subs[1]}</li>
                                    <li>{subs[2]}</li>
                                </ul>
                            </div>
                            <div class="preview">
                                <img src="/static/output.png" alt="Detected Image">
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <a class="btn" href="/">Try another image</a>
                        </div>
                    </div>
                    <div class="footer">
                        Created by Andrew McMahon • CPR E 575 - ISU • Spring 2025
                    </div>
                </body>
                </html>
            '''
    return '''
        <html>
        <head>
            <title>Upload Food Image</title>
            <style>
                body {
                    background-color: #e6ffe6;
                    font-family: Arial, sans-serif;
                    padding: 40px;
                }
                .container {
                    background-color: #ffffff;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    max-width: 600px;
                    margin: auto;
                    text-align: center;
                }
                h1 {
                    color: #2e7d32;
                }
                input[type="file"] {
                    padding: 10px;
                    margin: 15px 0;
                }
                input[type="submit"] {
                    padding: 10px 20px;
                    background-color: #4caf50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                .footer {
                    margin-top: 40px;
                    text-align: center;
                    font-size: 0.9em;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Food Image Classification</h1>
                <form method="post" enctype="multipart/form-data">
                    <p><b>Upload a food image to classify:</b></p>
                    <input type="file" name="image"><br>
                    <input type="submit" value="Classify Food">
                </form>
            </div>
            <div class="footer">
                Created by Andrew McMahon • HCI 575 @ ISU • Spring 2025
            </div>
        </body>
        </html>
    '''

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)