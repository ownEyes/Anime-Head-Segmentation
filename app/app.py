import gradio as gr
from pathlib import Path
import secrets
import shutil


def upload_file(files):
    file_paths = [file.name for file in files]
    print(f"Uploaded files: {file_paths}")
    return file_paths


def Download_file(files):
    file_paths = [file.name for file in files]
    print(f"Downloaded files: {file_paths}")
    return file_paths


with gr.Blocks(theme=gr.themes.Monochrome(), delete_cache=(60, 3600)) as demo:
    gr.Markdown("""# Satellite Image Roofs Auto Annotation
                Powered by fine-tuned RT-DETR model and Fast-SAM model.
               📤 Upload an image or a folder containing images.
               🖼️ Images are saved in a user-specific directory and deleted when the users closes the page.
            """)
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                interactive=True,
                sources=["upload", "clipboard"],
                show_share_button=True,
                type='filepath',
                label="Upload a single image",
            )
            upload_button = gr.UploadButton("Upload a folder", file_count="directory", file_types=["image"])
            batch_slider = gr.Slider(1, 6, step=1, label="Batch size", interactive=True, value=4, visible=False)
        with gr.Column(scale=2):
            img_output = gr.AnnotatedImage(label="Annotated Image", visible=False)
            download_button = gr.DownloadButton("Download annotation results", label="Download", visible=False)
    with gr.Row():

        examples = gr.Examples(
            examples=[
                ["./img/example.jpg"],
            ],
            inputs=[img_input],
        )

demo.launch()
