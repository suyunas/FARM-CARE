import os
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# ----------- USER CONFIGURATION -----------

EXCEL_FILES = [
     r'J:\Batch_1_YF12749_0_YF12811_1.xlsx',
     r'J:\Batch_2_YF12892_0_OF174_1.xlsx',
     r'J:\Batch_3_YF12825B_0_OF62_1.xlsx',
     r'J:\Batch_4_YF12876B_0_YF13921_1.xlsx',
     r'J:\Batch_5_YF12447_0_YF13922_1.xlsx',
     r'J:\Batch_6_YF12612_0_YF12746_1.xlsx',
     r'J:\Batch_7_YF12752_0_YF12750_1.xlsx',
     r'J:\Batch_8_YF13770_0_YF13825_1.xlsx',
     r'J:\Batch_9_YF12697_0_PF25_1.xlsx'
]

MODEL_PATHS = [
    r'J:\old_results\models\convnext\batch_1\convnext_best.pth',
    r'J:\old_results\models\efficientnet_v2\batch_1\efficientnet_v2_best.pth',
    r'J:\old_results\models\mobilenet_v3\batch_1\mobilenet_v3_best.pth',
    r'J:\old_results\models\regnet\batch_1\regnet_best.pth',
    r'J:\old_results\models\vit\batch_1\vit_best.pth'
]

# Add base model names here to skip them (e.g. 'efficientnet', 'vit')
SKIP_MODELS = []
SKIP_MODELS = ['efficientnet', 'mobilenet', 'regnet', 'vit']

RESULTS_BASE_DIR = Path(r"J:\old_results\gradcam results")  # Main Grad-CAM results folder
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- MODEL INITIALIZATION -----------

def initialize_model(model_name):
    base = model_name.split("_")[0]

    if base == 'convnext':
        model = models.convnext_base(weights='DEFAULT')
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    elif base == 'efficientnet':
        model = models.efficientnet_v2_s(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif base == 'mobilenet':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    elif base == 'regnet':
        model = models.regnet_y_400mf(weights='DEFAULT')
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif base == 'vit':
        model = models.vit_b_16(weights='DEFAULT')
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

def get_target_layer(model, model_name):
    base = model_name.split("_")[0]

    if base == 'convnext':
        # get convnext features.7 module
        return dict(model.named_modules())["features.7"]
    elif base == 'efficientnet':
        # efficientnet features.6.1 module
        return dict(model.named_modules())["features.6.1"]
    elif base == 'mobilenet':
        # mobilenet features.15 module
        return dict(model.named_modules())["features.15"]
    elif base == 'regnet':
        # regnet trunk_output attribute
        return getattr(model, "trunk_output")
    elif base == 'vit':
        # Use conv_proj conv layer for ViT GradCAM
        return model.conv_proj
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ----------- TRANSFORMS -----------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------- MAIN PROCESSING FUNCTION -----------

def apply_gradcam_from_excel(excel_path, model_name, model_path):
    df = pd.read_excel(excel_path)
    image_paths = df['Path'].tolist()
    conditions = df['Condition'].tolist()

    excel_base_name = Path(excel_path).stem
    model_folder = RESULTS_BASE_DIR / excel_base_name / model_name
    unstressed_dir = model_folder / 'Unstressed'
    stressed_dir = model_folder / 'Stressed'

    unstressed_dir.mkdir(parents=True, exist_ok=True)
    stressed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Running model: {model_name} on file: {excel_base_name}")

    model = initialize_model(model_name).to(DEVICE)
    model.eval()

    if not Path(model_path).exists():
        print(f"❌ Missing weights: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    try:
        target_layer = get_target_layer(model, model_name)
        cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    except Exception as e:
        print(f"⚠️ Error initializing CAM for {model_name}: {e}")
        return

    for img_path_str, cond in zip(image_paths, conditions):
        try:
            img_path = Path(img_path_str)
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)

            output = model(input_tensor)
            pred_class = output.argmax().item()

            cams = cam_extractor(pred_class, output)
            cam = cams[0].cpu()
            result = overlay_mask(img, to_pil_image(cam, mode='F'), alpha=0.5)

            out_dir = unstressed_dir if cond == 0 else stressed_dir
            result.save(out_dir / img_path.name)

        except Exception as e:
            print(f"⚠️ Failed processing image {img_path_str}: {e}")

    print(f"✅ Finished {model_name} for {excel_base_name}")

# ----------- RUN PIPELINE -----------

if __name__ == "__main__":
    for excel_path in EXCEL_FILES:
        for model_path_str in MODEL_PATHS:
            model_path = Path(model_path_str)
            model_name = model_path.stem  # e.g., 'convnext_best'
            base_model_name = model_name.split("_")[0]  # get base model name like 'convnext'

            if base_model_name in SKIP_MODELS:
                print(f"⏭️ Skipping model: {model_name}")
                continue

            apply_gradcam_from_excel(excel_path, model_name, model_path)

    print("\n🎉 All Grad-CAM processing complete.")
