from flask import Flask, request, render_template, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import io
import os
import random
from PIL import Image
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/puzzle_pieces'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load AlexNet Model for Dinosaur Classification
def load_alexnet_model():
    model = models.alexnet(weights=None)
    num_classes = 43  # Set this to the number of classes in your fine-tuned model
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    model.load_state_dict(torch.load(r'T:\pro\dinopedia\alexnet_dino_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

    
class_to_dino = {
    0: "Allosaurus", 1: "Ankylosaurus", 2: "Apatosaurus", 3: "Baryonyx", 4: "Brachiosaurus",
    5: "Carnotaurus", 6: "Ceratosaurus", 7: "Compsognathus", 8: "Corythosaurus", 9: "Dilophosaurus",
    10: "Dimetrodon", 11: "Dimorphodon", 12: "Dreadnoughtus", 13: "Gallimimus", 14: "Giganotosaurus",
    15: "Iguanodon", 16: "Kentrosaurus", 17: "Lystrosaurus", 18: "Mamenchisaurus", 19: "Microceratus",
    20: "Monolophosaurus", 21: "Mosasaurus", 22: "Nasutoceratops", 23: "Nothosaurus", 24: "Ouranosaurus",
    25: "Oviraptor", 26: "Pachycephalosaurus", 27: "Pachyrhinosaurus", 28: "Parasaurolophus",
    29: "Pteranodon", 30: "Pyroraptor", 31: "Quetzalcoatlus", 32: "Sinoceratops", 33: "Smilodon",
    34: "Spinosaurus", 35: "Stegosaurus", 36: "Stygimoloch", 37: "Suchomimus", 38: "Tarbosaurus",
    39: "Therizinosaurus", 40: "Triceratops", 41: "Tyrannosaurus rex", 42: "Velociraptor"
}

model = load_alexnet_model()

# Load Flan-T5 Model for Simplifying Language
def load_simplify_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

simplify_model, simplify_tokenizer = load_simplify_model()





# Preprocess and Classify Image
def classify_dinosaur(image):
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_to_dino.get(predicted.item(), "Unknown Dinosaur")

# Simplify Language for Kids
def simplify_text(text):
    prompt = f"Simplify this text for kids: {text}"
    inputs = simplify_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = simplify_model.generate(inputs["input_ids"], max_length=100, temperature=0.7)
    return simplify_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fetch and Simplify Specific Dinosaur Sections from Wikipedia
def get_dino_details(species_name):
    wiki = wikipediaapi.Wikipedia(
    
    language='en',
        user_agent="DinopediaApp/1.0 (Contact: tabran1023@gmail.com)"
    )
    page = wiki.page(species_name)
    if not page.exists():
        return {"error": "No details available"}
    
    # Retrieve specific sections
    details = {
        "title": page.title,
        "summary": simplify_text(page.summary),
        "history": "",
        "description": {
            "sizes": "",
            "skull": "",
            "skeleton": "",
            "feeding_strategies": ""
        },
        "fossil": "",
        "footprint": ""
    }

    # Parse sections and subsections for the desired information
    for section in page.sections:
        if section.title.lower() == "history":
            details["history"] = simplify_text(section.text)
        elif section.title.lower() == "description":
            for subsection in section.sections:
                if "size" in subsection.title.lower():
                    details["description"]["sizes"] = simplify_text(subsection.text)
                elif "skull" in subsection.title.lower():
                    details["description"]["skull"] = simplify_text(subsection.text)
                elif "skeleton" in subsection.title.lower():
                    details["description"]["skeleton"] = simplify_text(subsection.text)
                elif "feeding" in subsection.title.lower():
                    details["description"]["feeding_strategies"] = simplify_text(subsection.text)
        elif section.title.lower() == "fossil":
            details["fossil"] = simplify_text(section.text)
        elif section.title.lower() == "footprint":
            details["footprint"] = simplify_text(section.text)

    return details

# Create a 9-Grid Puzzle for the Uploaded Image
def create_puzzle(image, species_name):
    width, height = image.size
    tile_width, tile_height = width // 3, height // 3
    tiles = []
    for row in range(3):
        for col in range(3):
            tile = image.crop((col * tile_width, row * tile_height, (col + 1) * tile_width, (row + 1) * tile_height))
            tile_filename = f"{species_name}_tile_{row}_{col}.png"
            tile_path = os.path.join(app.config['UPLOAD_FOLDER'], tile_filename)
            tile.save(tile_path)
            tiles.append(url_for('static', filename=f"puzzle_pieces/{tile_filename}"))
    random.shuffle(tiles)
    return tiles

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        image = Image.open(io.BytesIO(file.read()))

        # Classify Dinosaur
        species_name = classify_dinosaur(image)

        # Retrieve Simplified Dinosaur Info
        dino_info = get_dino_details(species_name)
        if "error" in dino_info:
            return render_template("index.html", error=dino_info["error"])

        # Create 9-Grid Puzzle
        puzzle_tiles = create_puzzle(image, species_name)

        return render_template("index.html", species_name=species_name, dino_info=dino_info,
                               puzzle_tiles=puzzle_tiles, uploaded_image=url_for('static', filename=f"{file.filename}"))

    return render_template("index1.html")

if __name__ == "__main__":
    app.run(debug=True)
