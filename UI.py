import gradio as gr
import numpy as np
from inference import *

def sepia(input_img):
    return_str = "UI demo"
    print(input_img.name)
    return_str = inference("./model/ARCTIC/best_flickr8k.ckpt", input_img.name)
    return return_str

demo = gr.Interface(sepia, gr.File(), gr.Textbox())
demo.launch()
