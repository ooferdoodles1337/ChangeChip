import gradio as gr

from changechip import *


def process(
    input_image,
    reference_image,
    resize_factor,
    output_alpha,
    window_size,
    clusters,
    pca_dim_gray,
    pca_dim_rgb,
):
    return pipeline(
        (input_image, reference_image),
        resize_factor=resize_factor,
        output_alpha=output_alpha,
        window_size=window_size,
        clusters=clusters,
        pca_dim_gray=pca_dim_gray,
        pca_dim_rgb=pca_dim_rgb,
    )


with gr.Blocks() as demo:
    gr.Markdown("# ChangeChip")
    gr.Markdown(
        """
        Welcome to ChangeChip! This tool allows you to detect defects on printed circuit boards (PCBs) by comparing an input image with a reference "golden sample" image. 
        Simply upload your images, adjust the settings if needed, and click "Run" to highlight any discrepancies.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image")
            reference_image = gr.Image(label="Reference Image")
            with gr.Accordion(label="Other Options", open=False):
                resize_factor = gr.Slider(0.1, 1, 0.5, step=0.1, label="Resize Factor")
                output_alpha = gr.Slider(0, 255, 50, step=1, label="Output Alpha")
                window_size = gr.Slider(0, 10, 5, step=1, label="Window Size")
                clusters = gr.Slider(0, 32, 16, step=1, label="Clusters")
                pca_dim_gray = gr.Slider(0, 9, 3, step=1, label="PCA Dim Gray")
                pca_dim_rgb = gr.Slider(0, 27, 9, step=1, label="PCA Dim RGB")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Output Image", scale=9)
            btn = gr.Button("Run", scale=1)

    btn.click(
        fn=process,
        inputs=[
            input_image,
            reference_image,
            resize_factor,
            output_alpha,
            window_size,
            clusters,
            pca_dim_gray,
            pca_dim_rgb,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
