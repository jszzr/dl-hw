import gradio as gr
import numpy as np

def sepia(input_img):
    return_str = "UI demo"
    return return_str

demo = gr.Interface(sepia, gr.Image(), gr.Textbox())
demo.launch()