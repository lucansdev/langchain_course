import openai
from playsound import playsound #type:ignore
import speech_recognition as sr #type:ignore
from pathlib import Path
from io import BytesIO
import os
import dotenv

dotenv.load_dotenv()

client = openai.Client(api_key=os.getenv("openaiKey"))

arquivo_audio = "hello.mp3"

recognizer = sr.Recognizer()

def gravar_audio():
    with sr.Microphone(0) as source:
        print("ouvindo...")
        recognizer.adjust_for_ambient_noise(source,duration=1)
        audio = recognizer.listen(source)

    return audio


def transcricao_audio(audio):
    try:
        wav_data = BytesIO(audio.get_wav_data())
        wav_data.name = "audio.wav"
        transcricao = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_data
        )
        return transcricao.text
    except Exception as e:
        print(f"erro na transcrição do audio {e}")
        return ""
    
def completa_texto(mensagens):
    try:
        resposta = client.chat.completions.create(
            messages=mensagens,
            model="gpt-3.5-turbo-0125",
            max_tokens=1000,
            temperature=0
        )
        return resposta.choices[0].message.content
    except Exception as e:
        print(f"erro na geracao de resposta {e}")
        return "desculpe,nao consegui entender"
    
def cria_audio(texto):
    if Path(arquivo_audio).exists():
        Path(arquivo_audio).unlink()
    
    try:
        resposta  = client.audio.speech.create(model="tts-1",voice="echo",input=texto)
        resposta.write_to_file(arquivo_audio)
    except Exception as e:
        print(f"erro na criacao do audio")


def roda_audio():
    if Path(arquivo_audio).exists():
        playsound(arquivo_audio)
    else:
        print("erro: o arquivo nao foi encontrado")



def main():
    mensagens = []
    while True:
        audio = gravar_audio()
        transcricao = transcricao_audio(audio)

        if not transcricao:
            print("nao foi possivel transcrever o audio")
            continue
        
        mensagens.append({"role":"user","content":transcricao})

        #print(f"user:{mensagens[-1]["content"]}")

        resposta_texto = completa_texto(mensagens)
        mensagens.append({"role":"assistant","content":resposta_texto})

        #print(f"assistant: {mensagens[-1]["content"]}")

        cria_audio(resposta_texto)
        roda_audio()

        


if __name__ == "__main__":
    main()