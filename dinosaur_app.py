from flask import Flask, request, render_template, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import io
import os
import random
import torchvision.models as models

app = Flask(__name__)

# Configuration for uploads and puzzle storage
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['PUZZLE_FOLDER'] = './static/puzzle_pieces'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PUZZLE_FOLDER'], exist_ok=True)

# Load AlexNet Model for Dinosaur Classification
def load_alexnet_model():
    model = models.alexnet(weights=None)
    num_classes = 43
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    model.load_state_dict(torch.load(r'T:\pro\dinopedia\alexnet_dino_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_alexnet_model()

# Class mappings
class_to_dino = {
    0: "Allosaurus", 1: "Ankylosaurus", 2: "Apatosaurus", 3: "Baryonyx", 4: "Brachiosaurus",
    # Add the remaining classes as in your original code
    41: "Tyrannosaurus rex", 42: "Velociraptor"
}

# Load Flan-T5 model
def load_flan_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

flan_model, flan_tokenizer = load_flan_model()

# Image preprocessing for classification
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Classify the dinosaur using the AlexNet model
def classify_dinosaur(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        return class_to_dino.get(predicted_class.item(), "Unknown Species")

# Function to simplify text using Flan model
def simplify_text(text):
    prompt = f"Simplify and explain this text for kids: {text}"
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = flan_model.generate(inputs["input_ids"], max_length=150, temperature=0.7)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fetch Dinosaur Information from Wikipedia
def get_dino_details(species_name):
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(species_name)
    if not page.exists():
        return {"error": "No details available"}
    
    summary = simplify_text(page.summary)
    return {"title": page.title, "description": summary}

# Route for index page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        species_name = classify_dinosaur(image)
        
        dino_info = get_dino_details(species_name)
        if "error" in dino_info:
            return render_template("index.html", error=dino_info["error"])

        return render_template("index.html", species_name=species_name, dino_info=dino_info)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
