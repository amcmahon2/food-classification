import json
import shutil
import os
import re
from bing_image_downloader import downloader
IMAGES_PER_TEMPLATE = 5
DATA_DIR = '../data/images'
JSON_PATH = '../data/nutrition/USDA_db.json'

#remove old images
if os.path.exists(DATA_DIR):
    for dir in os.listdir(DATA_DIR):
        shutil.rmtree(os.path.join(DATA_DIR, dir))

#load descriptions from usda database
with open(JSON_PATH) as f:
    data = json.load(f)
foods = [item["description"] for item in data["FoundationFoods"]]

#clean the food name: remove forbidden Windows filename characters
def clean_filename(text):
    text = text.replace(',', '')
    text = text.replace('/', '-')
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = text.strip()
    return text

#query templates for better image results
query_templates = [
    "{} food isolated",
    "{} on white plate",
    "plain {} dish photo",
    "single {} meal top view"
]

#download images via bing image downloader
for food in foods:
    clean_food = clean_filename(food)
    print(f"\nSearching images for: {food}")
    for template in query_templates:
        search_term = template.format(clean_food)
        downloader.download(
            search_term,
            limit=IMAGES_PER_TEMPLATE,
            output_dir=DATA_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=30
        )
print("\nAll images downloaded.")