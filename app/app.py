import gradio as gr
from pathlib import Path
import secrets
import shutil
from inference import detector, detector_processor, segment_predictor, ModelInference

current_dir = Path(__file__).parent


def process_images(image_path, files, slider_value, request: gr.Request):

    user_dir: Path = current_dir / str(request.session_hash)
    user_dir.mkdir(exist_ok=True)

    annotation_path = user_dir / f"{secrets.token_hex(nbytes=8)}_annotations.coco.json"
    class_names = list(inferencer.id2label.values())

    if image_path:
        print(f"Processing image: {image_path}")
        seg_detections, annotated_frame = inferencer.predict_one(image_path)

        inferencer.save_annotations([image_path], [seg_detections], class_names, annotation_path)
    elif files:
        print(f"Processing files: {files}")
        print(f"Batch size: {slider_value}")
        all_image_paths, all_results, annotated_frame,  detector_failed_list, segmentor_failed_list = inferencer.predict_folder(files, slider_value)

        print(f"Detector failed list: {detector_failed_list}")
        print(f"Segmentor failed list: {segmentor_failed_list}")

        inferencer.save_annotations(all_image_paths, all_results, class_names, annotation_path)

    return [
        gr.UploadButton(visible=False),
        gr.Button("Run", visible=False),
        gr.DownloadButton("Download annotation results", value=annotation_path, label="Download", visible=True),
        gr.Image(value=annotated_frame, label="Annotated Image", visible=True),
    ]


def upload_file():

    return [
        None,
        gr.UploadButton(visible=False),
        gr.Slider(1, 6, step=1, label="Batch size", interactive=True, value=4, visible=True),
        gr.Button("Run", visible=True),
        gr.DownloadButton(visible=False),
        gr.Image(value=None, label="Annotated Image", visible=True),
    ]


def upload_image(imge_path):

    return [
        gr.UploadButton(visible=False),
        gr.Slider(1, 6, step=1, label="Batch size", interactive=True, value=4, visible=False),
        gr.Button("Run", visible=True),
        gr.DownloadButton(visible=False),
        gr.Image(value=None, label="Annotated Image", visible=True),
    ]


def download_file():
    return [
        gr.Image(value=None),
        gr.UploadButton(visible=True),
        gr.Slider(1, 6, step=1, label="Batch size", interactive=True, value=4, visible=False),
        gr.Button("Run", visible=False),
        gr.DownloadButton(visible=True),
        gr.Image(value=None, visible=False),
    ]


def delete_directory(request: gr.Request):
    """Delete the user-specific directory when the user's session ends."""
    user_dir = current_dir / str(request.session_hash)
    if user_dir.exists():
        shutil.rmtree(user_dir)


def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Monochrome(), delete_cache=(60, 3600)) as demo:
        gr.HTML("""
                <div style="text-align: center;">
                <h1>Satellite Image Roofs Auto Annotation</h1>
                <p>Powered by a <a href="https://huggingface.co/Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection" target="_blank">fine-tuned RT-DETR model</a> and Fast-SAM model.</p>
                <p>📤 Upload an image or a folder containing images.</p>
                <p>🖼️ Images are saved in a user-specific directory and deleted when the user closes the page.</p>
                <p>⚙️ Each user can upload files with a maximum file size of 200 MB.</p>
                </div>
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
                upload_button = gr.UploadButton("Upload a folder", file_count="directory")
                batch_slider = gr.Slider(1, 6, step=1, label="Batch size", interactive=True, value=4, visible=False)
                run_button = gr.Button("Run", visible=False)
            with gr.Column(scale=1):
                img_output = gr.Image(label="Annotated Image", visible=False)
                download_button = gr.DownloadButton("Download annotation results", label="Download", visible=False)

        with gr.Row():
            examples = gr.Examples(
                examples=[["./img/example.jpg"]],
                inputs=[img_input],
                outputs=[upload_button, batch_slider, run_button, download_button, img_output],
                fn=upload_image,
                run_on_click=True,
            )

        upload_button.upload(upload_file, None, [img_input, upload_button, batch_slider, run_button, download_button, img_output])

        download_button.click(download_file, None, [img_input, upload_button, batch_slider, run_button, download_button, img_output])

        run_button.click(process_images,
                         [img_input, upload_button, batch_slider],
                         [upload_button, run_button, download_button, img_output])

        img_input.upload(upload_image, img_input, [upload_button, batch_slider, run_button, download_button, img_output])

        demo.unload(delete_directory)

    return demo


def inferencer_init():
    id2label = {0: 'building'}
    CONFIDENCE_TRESHOLD = 0.5
    return ModelInference(detector, detector_processor, segment_predictor, id2label, CONFIDENCE_TRESHOLD)


inferencer = inferencer_init()

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(max_file_size=200 * gr.FileSize.MB)
