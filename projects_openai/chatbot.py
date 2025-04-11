import openai
import dotenv
import os

dotenv.load_dotenv()

cliente = openai.Client(api_key=os.environ["openaiKey"])

def geracao_texto(mensagens):
    resposta = cliente.chat.completions.create(messages=mensagens,model="gpt-3.5-turbo-0125",max_tokens=1000,temperature=0,stream=True)
    print("bot:",end="")
    texto_completo = ""
    for resposta_stream in resposta:
        texto = resposta_stream.choices[0].delta.content
        if texto:
            print(texto,end="")
            texto_completo += texto
    print()

    mensagens.append({"role":"assistant","content":texto_completo})
    return mensagens


if __name__ == "__main__":
    print("bem vindo ao chatbot")
    mensagens:list = []
    while True:
        in_user = input("user:")
        mensagens.append({"role":"user","content":in_user})
        mensagens = geracao_texto(mensagens)