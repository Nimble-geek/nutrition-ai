import os
import gradio as gr

def hello(name):
    return f"Hello {name}"

demo = gr.Interface(hello, "text", "text")

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)