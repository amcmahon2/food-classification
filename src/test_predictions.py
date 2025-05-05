import os
import cv2
import torch
import pickle
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from predict import predict_cnn, predict_orb

#paths
DATASET_DIR = '../data/split/val'
FEATURE_DB_PATH = '../models/feature_db.pkl'
MODEL_PATH = '../models/food_classifier.pth'
CLASS_NAMES_PATH = '../models/classes.txt'

#transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#load class names
with open(CLASS_NAMES_PATH) as f:
    class_names = [line.strip() for line in f]

#load ORB feature DB
#with open(FEATURE_DB_PATH, 'rb') as f:
#   feature_db = pickle.load(f)

#load CNN model
model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

#collect test samples (limit to 100 total)
test_samples = []
for cls in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(cls_path):
        images = os.listdir(cls_path)
        selected = random.sample(images, min(2, len(images)))  # max 2 per class
        for img in selected:
            test_samples.append((os.path.join(cls_path, img), cls))
    if len(test_samples) >= 100:
        break

random.shuffle(test_samples)

#run predictions
cnn_top1_preds, cnn_top5_preds, orb_preds, truths = [], [], [], []

print(f"Running predictions on {len(test_samples)} images...\n")
for idx, (img_path, label) in enumerate(test_samples):
    print(f"[{idx+1}/{len(test_samples)}] Processing CNN for {os.path.basename(img_path)}")
    
    start_time = time.time()
    # CNN
    top5 = predict_cnn(img_path, MODEL_PATH, transform)
    cnn_top1_preds.append(top5[0])
    cnn_top5_preds.append(top5)
    end_time = time.time()
    print("CNN took " + f"{end_time - start_time}")

    #print(f"Processing ORB for {os.path.basename(img_path)}")
    #start_time2 = time.time()
    # ORB
    #orb_label = predict_orb(img_path, feature_db)
    #orb_preds.append(orb_label)
    #end_time2 = time.time()
    #print("ORB took " + f"{end_time2 - start_time2}")


    #ground truth
    truths.append(label)

#metrics
cnn_top1_acc = accuracy_score(truths, cnn_top1_preds)
cnn_top5_acc = np.mean([truth in top5 for truth, top5 in zip(truths, cnn_top5_preds)])
#orb_acc = accuracy_score(truths, orb_preds)

print(f"\nCNN Top-1 Accuracy: {cnn_top1_acc*100:.2f}%")
print(f"CNN Top-5 Accuracy: {cnn_top5_acc*100:.2f}%")
#print(f"ORB Accuracy: {orb_acc*15:.2f}%")

print("\nClassification Report (CNN):")
print(classification_report(truths, cnn_top1_preds, zero_division=0))

#print("\nClassification Report (ORB):")
#print(classification_report(truths, orb_preds, zero_division=0))

#confusion Matrices
cm_cnn = confusion_matrix(truths, cnn_top1_preds, labels=class_names)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=class_names)
disp_cnn.plot(include_values=False, xticks_rotation='vertical', cmap='Blues')
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.show()

#cm_orb = confusion_matrix(truths, orb_preds, labels=class_names)
#disp_orb = ConfusionMatrixDisplay(confusion_matrix=cm_orb, display_labels=class_names)
#disp_orb.plot(include_values=False, xticks_rotation='vertical', cmap='Oranges')
#plt.title("ORB Confusion Matrix")
#plt.tight_layout()
#plt.show()