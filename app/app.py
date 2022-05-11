import torch, sys, os, pandas as pd, numpy as np
import streamlit as st
from PIL import Image
from utils import predict_proba


def main():
    st.title('Cat Dog Classifier')
    file_uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")

    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image.resize((300, 300)), caption='Uploaded Image')

    if class_btn and file_uploaded:
        predictions = predict_proba(file_uploaded)
        print(predictions)
        st.title(f"THe prediction is: {predictions[0][0]}")
        df = pd.DataFrame(data=np.zeros((len(predictions), 2)),
                        columns=['Species', 'Confidence Level'],
                        index=np.linspace(1, len(predictions), len(predictions), dtype=int))
        for idx, p in enumerate(predictions):
            df.iloc[idx,0] = p[0]
            df.iloc[idx, 1] = p[1]
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

    
            


if __name__ == "__main__":
    main()