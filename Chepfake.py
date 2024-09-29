#pip install torch torchvision transformers sentence-transformers Pillow streamlit  
import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_model = SentenceTransformer("all-MiniLM-L6-v2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image):
    if image is not None:
        try:
            i_image = Image.open(image)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [pred.strip() for pred in preds]
    return None


st.title("Image Captioning and Similarity Checker")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


user_caption = st.text_input("Enter a caption to compare:")

if st.button("Generate Caption and Similarity"):
    if uploaded_file and user_caption:
       
        generated_caption = predict_step(uploaded_file)
        
        if generated_caption:
            st.write("Generated Caption:", generated_caption[0])

            
            embeddings1 = caption_model.encode(generated_caption)
            embeddings2 = caption_model.encode([user_caption])
            
            
            similarities = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings1), torch.tensor(embeddings2)
            ).item()
            if similarities > 0.4:
                st.write("Non Out of Context Media")
            else:
                st.write("Out of Context Media")
            
        else:
            st.write("No captions generated.")
    else:
        st.warning("Please upload an image and enter a caption.")
