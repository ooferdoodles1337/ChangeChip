import gradio as gr
from changechip import *

def pipeline(input):
    return None


demo = gr.Interface(pipeline, ["image", "image"], "image")

if __name__ == "__main__":
    demo.launch()
