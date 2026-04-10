from __future__ import annotations

try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:  # pragma: no cover - compatibility shim for newer hub versions
            @staticmethod
            def get_token():
                return None

            @staticmethod
            def save_token(token):
                return None

            @staticmethod
            def delete_token():
                return None

        huggingface_hub.HfFolder = HfFolder
except ImportError:
    pass

import gradio as gr

from cifar10_inference import CIFAR10ResNet


MODEL = CIFAR10ResNet()


def predict(image):
    if image is None:
        return {}

    probabilities = MODEL.predict(image)
    return {
        MODEL.class_names[index]: float(probabilities[index])
        for index in range(len(MODEL.class_names))
    }


with gr.Blocks(title="CIFAR-10 Image Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CIFAR-10 Image Classifier

        Upload an image and the model will classify it into one of 10 CIFAR-10 classes.
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload image", height=320)
        label_output = gr.Label(num_top_classes=5, label="Predictions")

    image_input.change(fn=predict, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()
