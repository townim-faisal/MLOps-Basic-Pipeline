# from re import M
import torch, sys, os
import yaml
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
print(os.getcwd())
sys.path.append('.')
from model.models import AlexNet
from model.augment import transform_test

fig = plt.figure()

# Configuration
config_file = open("./model/params/config.yaml", "r")
config = yaml.safe_load(config_file)
config_file.close()

# with open("./app/custom.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Cat Dog Classifier')

def main():
    file_uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                # time.sleep(1)
                # st.success('Classified')
                st.write(predictions)
                # st.pyplot(fig)
            
def predict(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model_path = "./storage/artifact/Experiment 1/Version 1/best_model.pth"
    model = AlexNet(num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load(base_model_path)['model'])
    image = Image.open(image).convert('RGB')
    imagetensor = transform_test(image)
    ps=torch.exp(model(imagetensor))
    topconf, topclass = ps.topk(1, dim=1)
    if topclass.item() == 1:
        return {'class':'dog','confidence':str(topconf.item())}
    else:
        return {'class':'cat','confidence':str(topconf.item())}

if __name__ == "__main__":
    main()