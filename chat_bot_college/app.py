import gradio as gr
from back import retriever

retriever_db = retriever()

def response(text,retriever=retriever_db):
    texto = retriever.run(text)
    return texto

with gr.Blocks() as app:
    gr.Markdown("# CHATBOT UNIESP, BEM VINDO!")
    with gr.Row(equal_height=True):
        texto= gr.Textbox(label="faça sua pergunta")
        button = gr.Button("Enviar",variant="primary")
    
    area = gr.TextArea(label="resposta")
    
    button.click(
        fn=response,
        inputs=[texto],
        outputs=[area]
    )

if __name__ == "__main__":
    app.launch()
