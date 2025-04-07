from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
from args import get_parser
import pickle
import os

from utils.output_utils import prepare_output

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS

args = get_parser()

with open("ingr_vocab.pkl", "rb") as f:
    ingr_vocab = pickle.load(f)

with open("instr_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(args, 1488, 23231)
model.load_state_dict(torch.load("modelbest.ckpt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route("/")
def home():
    return render_template("index.html")  # Serve index.html

def decode_sequence(vocab, sequence):
    """Convert token indices back into words."""
    words = []
    for idx in sequence:
        if idx == vocab["<eos>"]:  # Stop at end token
            break
        words.append(vocab.get(idx, "<unk>"))  # Convert index to word
    return " ".join(words)


def predict_recipe(image):
    """Predict ingredients and recipe from an image."""
    image = transform(image).unsqueeze(0).to(device)  # Move to GPU/CPU

    with torch.no_grad():
        outputs = model.sample(image, greedy=True, temperature=1.0)

    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    recipe_ids = outputs['recipe_ids'].cpu().numpy()


    outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, vocab)
    # Decode ingredient IDs
    # ingredients = []
    # for idx in ingr_ids:
    #     for key, value in ingr_vocab.items():
    #         if value == idx:
    #             ingredients.append(key)

    # # Decode recipe instructions

    #ingredients = [ingr_vocab[idx] for idx in ingr_ids if idx < len(ingr_vocab)]

    # Decode recipe instructions
    #recipe_steps = decode_sequence(instr_vocab, recipe_ids)


    # recipe_steps = decode_sequence(instr_vocab, recipe_ids)

    # return ingredients, recipe_steps

    print(outs)
    return outs if valid['is_valid'] else {"title": "Not a valid recipe!", "reason": valid['reason']}



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file).convert("RGB")
        result = predict_recipe(image)
        
        ingredients = result["ingrs"]
        recepie = result["recipe"]
        title = result["title"]

        return jsonify({
            "ingredients": ingredients,
            "recipe": recepie,
            "title": title
        })
    

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
