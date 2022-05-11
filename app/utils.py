import torchvision.transforms as T, os, numpy as np, torch, streamlit as st
from PIL import Image
from config import *
from models import AlexNet

@st.cache()
def transform_data(file):
    img = Image.open(file)
    transform_val  = T.Compose([
                    T.Resize((227,227)),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    img_transformed = transform_val(img)
    img_transformed = img_transformed.unsqueeze(0)
    return img_transformed

@st.cache()
def load_model():
    f = open(os.path.join(root_path, best_model_path), 'r')
    lines = f.read().splitlines()
    f.close()
    # print(lines)
    model_path = lines[1].split('\t')[0]
    checkpoint = torch.load(os.path.join(root_path,model_path))
    model = AlexNet()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

@st.cache()
def predict_proba(img_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = transform_data(img_file)
    img = img.to(device)
    model = load_model()
    model = model.to(device)
    model.eval()
    output_tensor = model(img)
    prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
    top_k = torch.topk(prob_tensor, len(class_names), dim=1)
    probabilites = top_k.values.detach().numpy().flatten()
    indices = top_k.indices.detach().numpy().flatten()
    formatted_predictions = []

    for pred_prob, pred_idx in zip(probabilites, indices):
        predicted_label = class_names[pred_idx]
        predicted_perc = pred_prob * 100
        formatted_predictions.append((predicted_label, f"{predicted_perc:.3f}%"))

    return formatted_predictions

