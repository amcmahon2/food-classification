import cv2
import os
import pickle
import json
import torch
import numpy as np
from PIL import Image   
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import difflib  # for optional fuzzy matching
from flask import Flask, request, jsonify
import os
import json
import pickle
from torchvision import transforms
import torch
import time
from PIL import Image

with open('../models/classes.txt') as f:
    model_classes = [line.strip() for line in f]

def get_calories(usda_data, cnn_prediction):
    for item in usda_data.get("FoundationFoods", []):

        #if detected food is plural, chop the "s" off
        if len(cnn_prediction) >= 1:
            temp = cnn_prediction[0:len(cnn_prediction)-1]
        if (cnn_prediction.lower() in item.get("description", "").lower()) or (temp.lower() in item.get("description", "").lower()):
            for nutrient in item.get("foodNutrients", []):
                if (
                    nutrient.get("nutrient", {}).get("name") == "Energy" and
                    nutrient.get("nutrient", {}).get("unitName") == "kcal"
                ):
                    return nutrient.get("amount", 0.0)
    return 0.0  # fallback if nothing matches


def predict_cnn(img_path, model, transform):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        top5 = torch.topk(probs, k=5)
        predicted_indices = top5.indices[0].tolist()
        predicted_labels = [model_classes[i] for i in predicted_indices]
    return predicted_labels[0], predicted_labels #return top-1 label s well as top 5

def estimate_nutrition(image_path, food_label="unknown", calories_per_100g=250, density_g_cm3=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found")

    #resize large images to max 1024px for consistent processing
    MAX_DIM = 1024
    h, w = image.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 7)

    #Hough Circle Detection
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=100, param2=30, minRadius=10, maxRadius=100
    )

    if circles is None:
        raise ValueError("No circular reference (quarter) found!")

    #convert (x, y, r) to integers
    circles = np.uint16(np.around(circles))
    ref_circle = max(circles[0], key=lambda c: c[2])  # largest circle

    #use diameter in pixels to get conversion ratio
    ref_radius_px = ref_circle[2]
    ref_diameter_px = 2 * ref_radius_px
    REF_DIAMETER_CM = 2.426
    px_to_cm = REF_DIAMETER_CM / ref_diameter_px

    #canny + contour for food region
    edges = cv2.Canny(gray_blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        raise ValueError("No food contour found")

    #combine all contours excluding the quarter area
    food_contours = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        if abs(radius - ref_radius_px) > 10:
            food_contours.append(c)
    if len(food_contours) == 0:
        raise ValueError("No valid food region found")
    combined = np.vstack(food_contours)
    food_contour = cv2.convexHull(combined)
    x, y, w_box, h_box = cv2.boundingRect(food_contour)
    food_area_px = cv2.contourArea(food_contour)
    food_area_cm2 = food_area_px * (px_to_cm ** 2)
    food_height_cm = h_box * px_to_cm
    food_volume_cm3 = food_area_cm2 * food_height_cm
    food_mass_g = food_volume_cm3 * density_g_cm3
    calories = (food_mass_g / 100) * calories_per_100g

    #visualization
    vis = image.copy()
    cv2.drawContours(vis, [food_contour], -1, (0, 0, 204), 2)
    cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (255, 102, 178), 2)
    #cv2.circle(vis, (ref_circle[0], ref_circle[1]), ref_radius_px, (0, 255, 255), 2)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis_rgb)
    plt.title("Red: Food, Blue: Bounding box")
    plt.axis("off")
    plt.tight_layout()
    #plt.show()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/output.png")
    plt.close()


    return f"""
    <table border="1" cellpadding="8" cellspacing="0">
    <tr><th>Measurement</th><th>Value</th></tr>
    <tr><td>Pixel-to-CM Ratio</td><td>{round(px_to_cm, 4)} px/cm</td></tr>
    <tr><td>Bounding Box Height</td><td>{round(food_height_cm, 2)} cm</td></tr>
    <tr><td>Estimated Area</td><td>{round(food_area_cm2, 2)} cm²</td></tr>
    <tr><td>Estimated Volume</td><td>{round(food_volume_cm3, 2)} cm³</td></tr>
    <tr><td>Estimated Mass</td><td>{round(food_mass_g, 2)} g</td></tr>
    <tr><td>Estimated Calories</td><td>{round(calories, 2)} kcal</td></tr>
    </table>
    """




#old code which uses ORB

# #build feature database
# def build_featureDB(image_dir, path):
#     print("Building feature DB at path" + path)
#     orb = cv2.ORB_create()
#     db = []
#     for class_name in os.listdir(image_dir):
#         class_path = os.path.join(image_dir, class_name)
#         print(class_name)
#         if not os.path.isdir(class_path):
#             continue
#         for img_name in os.listdir(class_path):
#             img_path = os.path.join(class_path, img_name)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 continue
#             kp, des = orb.detectAndCompute(img, None)
#             db.append((class_name, des))
#     with open(path, "wb") as f: #cache the feature DB
#         pickle.dump(db, f)
#     return db
# if __name__ == "__main__":
#    build_featureDB('../data/split/train', '../models/feature_db.pkl')

# #compute x and y Sobel gradients (secondary check for ORB-based matching) 
# def compute_gradient_magnitude(image):
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     grad_mag = cv2.magnitude(grad_x, grad_y)
#     return grad_mag

# #match uploaded image using openCV ORB matching
# def predict_orb(img_path, db):
#     orb = cv2.ORB_create(nfeatures=300)
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     def process(img):
#         kp, des = orb.detectAndCompute(img, None)
#         return kp, des, compute_gradient_magnitude(img)
    
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (256, 256))
#     kp1, des1, grad1 = process(img)
#     flipped_img = cv2.flip(img, 1)
#     kp1f, des1f, grad1f = process(flipped_img)
#     best_match = None
#     best_score = -1
#     for label, des2 in db:
#         if des1 is None or des2 is None:
#             continue
#         matches = bf.match(des1, des2)
#         matches_flipped = bf.match(des1f, des2)

#         #filter by distance
#         good = [m for m in matches if m.distance < 50]
#         good_f = [m for m in matches_flipped if m.distance < 50]
#         match_score = len(good) + 0.8 * len(good_f)

#         #gradient similarity comparison
#         db_img_path = f'../data/split/train/{label}/{os.listdir(f"../data/split/train/{label}")[0]}'
#         db_img = cv2.imread(db_img_path, cv2.IMREAD_GRAYSCALE)
#         db_img = cv2.resize(db_img, (256, 256))
#         grad2 = compute_gradient_magnitude(db_img)
#         grad_diff = np.mean(cv2.absdiff(grad1, grad2))

#         #lower gradient difference is better, subtract to reward close gradients
#         final_score = match_score - 0.01 * grad_diff
#         if final_score > best_score:
#             best_score = final_score
#             best_match = label
#     return best_match