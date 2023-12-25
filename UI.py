import gradio as gr
import numpy as np
from inference import *

def sepia(input_img, model):
    return_str = "UI demo"
    if model == "model1":
        model_path = "./model_1/best_flickr8k.ckpt"
    elif model == "model2":
        model_path = "./model_2/best_flickr8k.ckpt"
    return_str = inference(model_path=model_path, original_image=Image.fromarray(input_img))
    # return_str = inference("./model/ARCTIC/best_flickr8k.ckpt", image_path=str(input_img.name))
    return return_str

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():

            model = gr.Radio(["model1", "model2"], label="Model Choose") #单选

            image = gr.Image()
            greet_btn = gr.Button()
        output = gr.Textbox()
    greet_btn.click(fn=sepia, inputs=[image, model], outputs=output)

# demo = gr.Interface(sepia, gr.Image(), gr.Textbox())
demo.launch()
