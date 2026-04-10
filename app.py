# ============================================================
# app.py — Hugging Face Spaces | CIFAR-10 Classifier
# ============================================================

import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ── Load model + config ──────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model('cifar10_model.keras')

with open('model_config.json') as f:
    config = json.load(f)

CLASS_NAMES = config['classes']
MEAN = np.array(config['normalization']['mean'])
STD  = np.array(config['normalization']['std'])

print(f"Model loaded! Classes: {CLASS_NAMES}")

# ── Preprocessing ─────────────────────────────────────────────
def preprocess_image(pil_image):
    """
    User ki image → model ke liye ready
    """
    # Resize to 32x32 (CIFAR-10 size)
    img = pil_image.resize((32, 32), Image.LANCZOS)
    
    # RGB ensure karo (PNG mein RGBA hoti hai)
    img = img.convert('RGB')
    
    # NumPy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize: 0-255 → 0-1
    img_array = img_array / 255.0
    
    # Per-channel normalization (same as training)
    img_array = (img_array - MEAN) / STD
    
    # Batch dimension add karo: (32,32,3) → (1,32,32,3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ── Prediction Function ───────────────────────────────────────
def predict(image):
    """
    Gradio yeh function call karta hai
    Input: PIL Image
    Output: Dict of {class: confidence}
    """
    if image is None:
        return {}
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed, verbose=0)[0]
    
    # Dict banao: {class_name: probability}
    results = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }
    
    return results


# ── Example Images (optional) ────────────────────────────────
# Kuch example images add karo taaki users test kar sakein
examples = [
    # URL ya local path de sakte ho
    # "examples/cat.jpg",
    # "examples/airplane.jpg",
]


# ── Gradio Interface ──────────────────────────────────────────
with gr.Blocks(
    title="CIFAR-10 Image Classifier",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🖼️ CIFAR-10 Image Classifier
    
    **ResNet-style CNN** trained on CIFAR-10 dataset (~85% accuracy)
    
    Upload any image and the model will classify it into one of 10 categories:
    **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=300
            )
            submit_btn = gr.Button(
                "🔍 Classify Image",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            # Output
            label_output = gr.Label(
                num_top_classes=5,
                label="Predictions (Top 5)"
            )
    
    # Warning note
    gr.Markdown("""
    > ⚠️ **Note:** CIFAR-10 is trained on 32×32 images of specific objects.
    > For best results, upload clear images of:
    > animals (cat, dog, bird, deer, frog, horse) or
    > vehicles (airplane, automobile, ship, truck)
    """)
    
    # Model info accordion
    with gr.Accordion("ℹ️ Model Details", open=False):
        gr.Markdown(f"""
        | Detail | Value |
        |--------|-------|
        | Architecture | ResNet-style CNN |
        | Dataset | CIFAR-10 (60,000 images) |
        | Test Accuracy | ~85% |
        | Input Size | 32×32×3 |
        | Parameters | ~1.2M |
        | Framework | TensorFlow 2.x |
        """)
    
    # Connect button to function
    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=label_output
    )
    
    # Auto-predict on image upload bhi
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=label_output
    )

# Launch!
demo.launch()
