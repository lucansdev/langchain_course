from transformers import pipeline

image_to_text = pipeline("image-to-text",model="nlpconnect/vit-gpt2-image-captioning")
text = image_to_text("huggingface/segmentacao_imagem/imagens/image4.jpg")
print(text)