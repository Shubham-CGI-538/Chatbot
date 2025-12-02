import gradio as gr
from chatbot import chatbot_full  

APP_TITLE = "Learnings Search"
APP_DESC = (
    "Type your query and see results from SQL FTS (Exact/Partial) plus Semantic tier. "
    "Entries show the full content of each match."
)

def run_chat(query: str):
    if not query or not query.strip():
        return "Please enter a query."
    try:
        md = chatbot_full(query.strip())
        return md
    except Exception as e:
        return f"**Error:** {e}"

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(f"# {APP_TITLE}\n{APP_DESC}")

    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder='E.g., "root cause analysis" compressor failure',
            lines=2
        )

    with gr.Row():
        run_btn = gr.Button("Search", variant="primary")
        clear_btn = gr.Button("Clear")

    output = gr.Markdown(label="Results", value="_Results will appear here..._")

    def _on_clear():
        return "", "_Results will appear here..._"

    run_btn.click(run_chat, inputs=[query_box], outputs=[output])
    clear_btn.click(_on_clear, inputs=None, outputs=[query_box, output])

if __name__ == "__main__":
    # Launch on localhost; set share=True if you want a public link (not recommended on corp networks)
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
