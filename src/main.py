from flask import Flask, request, jsonify
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

            # start = time.time()
            # orb_prediction = predict_orb(filepath, feature_db)
            # print("ORB time to compute:", time.time() - start)

            start = time.time()
            cnn_prediction, cnn_prediction_top5 = predict_cnn(filepath, model, transform)
            print("CNN time to compute:", time.time() - start)

            #get calorie info from USDA JSON
            calories = get_calories(usda_data, cnn_prediction)

            #extract calorie amount if available
            nutrient_info = estimate_nutrition("uploads/test_image.png", cnn_prediction, calories, 0.5)

            #suggest substitutes
            substitutions_map = substitutions["General Substitutions"][0]
            subs = substitutions_map.get(cnn_prediction.lower(), ["none"])
            print(f"Substitutes for {cnn_prediction}: {subs}")

            return f'''
            <h2>CNN Prediction: {cnn_prediction}</h2>
            <h3>CNN Nutrition info: {nutrient_info}</h3>
            <h3>CNN Substitutes: {subs[0], subs[1], subs[2]}</h3>
            <br>
            <a href="/">Try another image</a>
            '''
    return '''
    <form method="post" enctype="multipart/form-data">
      Upload a food image: <input type="file" name="image"><br><br>
      <input type="submit" value="Classify Food">
    </form>
    '''

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)