import gradio as gr
import os

from chroma import ChromaVectorStore
from reasoning import GroqReasoningModel
from groq import Groq
import dotenv

db_path = "./chroma_db"

vector_store = ChromaVectorStore(db_path=db_path)

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = GroqReasoningModel(groq)


# -----------------------
# Core pipeline function
# -----------------------

def nutrition_agent(query):

    results = vector_store.query(
        query_text=query,
        n_results=2
    )

    response = llm.generate(results, query)

    return response


# -----------------------
# Gradio Interface
# -----------------------

demo = gr.Interface(
    fn=nutrition_agent,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Example: Give me a high protein breakfast and nutrient breakdown"
    ),
    outputs=gr.Markdown(),
    title="AI Nutrition Planner",
    description="Ask for meals, calories, and nutrient breakdowns."
)


# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    demo.launch(share=True)