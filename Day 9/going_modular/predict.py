from argparse import ArgumentParser

import torch
import torchvision
from torchvision import transforms

from typing import List
import matplotlib.pyplot as plt

from model_builder import TinyVGG

device = "cuda" if torch.cuda.is_available() else "cpu"

## To get the argument from the command line as "predict.py  --image image/path"
parser = ArgumentParser()

parser.add_argument("--image", type=str, default=None, help="Path of the inage to be tested")

args = parser.parse_args()

## Function to predict and plot the image and its probability
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    target_image_uint8 = torchvision.io.read_image(str(image_path))

    target_image = target_image_uint8 / 255

    if transform:
        target_image = transform(target_image)

    model.eval()

    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image)

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    print(f"The predicted class is {class_names[target_image_pred_label.cpu()]} with Probability: {target_image_pred_probs.max().cpu():.3f}")

def main():
    if not args.image:
        print("Didn't get any image paths to predict.")
        return

    else:
        IMAGE_PATH = "data/pizza_steak_sushi/test/pizza/194643.jpg"
        print(IMAGE_PATH)
        MODEL_PATH = "models/05_going_modular_tingvgg_model.pth"
        model = TinyVGG(
            input_shape=3,
            hidden_units=20,
            output_shape=3
        ).to(device)
        try: 
            model.load_state_dict(torch.load(MODEL_PATH))
        except Exception as e:
            print("Model not found ", e)
            user_input = input("Would you like to train it now? Y/n").lower()
            if user_input == "y":
                import subprocess

                script_name = "going_modular/train.py"
                arguments_dict = {
                    "batch_size": 32,
                    "learning_rate": 0.02,
                    "epochs": 5
                }
                arguments = [f"--{key}={value}" for key, value in arguments_dict.items()]

                subprocess.call(["python", script_name]+arguments)
            else:
                return
            try:
                model.load_state_dict(torch.load(MODEL_PATH))
            except Exception as e:
                print("Something went very wrong. Maybe you need to train the model again", e)
                return

        class_names = ["pizza", "steak", "sushi"]
        test_data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
        ])

        pred_and_plot_image(
            model=model,
            image_path=IMAGE_PATH,
            class_names=class_names,
            transform=test_data_transform,
            device=device
        )

if __name__ == "__main__":
    main()


