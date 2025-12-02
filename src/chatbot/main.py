import os
import gradio as gr
import json
import re
from crew import Chatbot

PORT = int(os.environ.get("PORT", 8000))   # Azure will set PORT; fallback to 8000
HOST = "0.0.0.0"

crew_instance = Chatbot()

def process_query(user_query: str) -> str:
    inputs = {"user_query": user_query}
    result = crew_instance.crew().kickoff(inputs=inputs)

    # Convert CrewOutput to string
    result_str = str(result)

    # Extract JSON block from ```json ... ```
    match = re.search(r"```json\s*(\{.*\})\s*```", result_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return "Failed to parse JSON from Crew output."

        # Build Markdown from structured data
        answer = data.get("answer", "")
        sources = ", ".join(map(str, data.get("sources", [])))
        notes = data.get("notes", "")

        md = "### Answer\n" + answer + "\n\n"
        if sources:
            md += f"**Sources:** {sources}\n\n"
        if notes:
            md += f"**Notes:** {notes}\n"
        return md

    # If no JSON found, fallback to raw string
    return result_str

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# CrewAI Chatbot - Markdown Output")
    user_input = gr.Textbox(label="Enter your query")
    output = gr.Markdown(label="Response")
    submit_btn = gr.Button("Submit")

    submit_btn.click(process_query, inputs=user_input, outputs=output)

if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT, show_error=True, share=False, prevent_thread_lock=True)
