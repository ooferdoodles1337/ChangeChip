import gradio as gr
from changechip import *


def process(input_image, reference_image, resize_factor):
    return pipeline((input_image, reference_image), resize_factor=resize_factor)


demo = gr.Interface(
    fn=process,
    inputs=["image", "image", gr.Slider(0.1, 1, 0.5, step=0.1)],
    outputs=["image"],
    title="ChangeChip",
)


if __name__ == "__main__":
    demo.launch()
