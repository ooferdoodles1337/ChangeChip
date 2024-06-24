import os
import gradio as gr

from changechip import *

app_port = os.getenv("APP_PORT", "7860")


def process(input_image, reference_image, resize_factor, output_alpha):
    return pipeline(
        (input_image, reference_image),
        resize_factor=resize_factor,
        output_alpha=output_alpha,
    )


with gr.Blocks() as demo:
    gr.Markdown("# ChangeChip")
    gr.Markdown(
        'Please input a "golden sample" PCB image as the reference image and compare it to the input image to highlight any defects.'
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image")
            reference_image = gr.Image(label="Reference Image")
            with gr.Accordion(label="Other Options", open=False):
                resize_factor = gr.Slider(0.1, 1, 0.5, step=0.1, label="Resize Factor")
                output_alpha = gr.Slider(0, 255, 50, step=1, label="Output Alpha")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Output Image", scale=9)
            btn = gr.Button("Run", scale=1)

    btn.click(
        fn=process,
        inputs=[input_image, reference_image, resize_factor, output_alpha],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
