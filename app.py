import gradio as gr
from changechip import *

def process(input_image, reference_image, resize_factor, output_alpha):
    return pipeline(
        (input_image, reference_image),
        resize_factor=resize_factor,
        output_alpha=output_alpha,
    )

with gr.Blocks() as demo:
    gr.Markdown("# ChangeChip")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image")
            reference_image = gr.Image(label="Reference Image")
            resize_factor = gr.Slider(0.1, 1, 0.5, step=0.1, label="Resize Factor")
            output_alpha = gr.Slider(0, 255, 50, step=1, label="Output Alpha")
        
        with gr.Column(scale=2):
            output_image = gr.Image(label="Output Image")
            btn = gr.Button("Run")
    
    
    
    btn.click(fn=process, inputs=[input_image, reference_image, resize_factor, output_alpha], outputs=output_image)

if __name__ == "__main__":
    demo.launch()
