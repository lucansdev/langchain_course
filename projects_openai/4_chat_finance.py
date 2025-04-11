import yfinance as yf
import openai
import json
import dotenv
import os

dotenv.load_dotenv()

client = openai.Client(api_key=os.getenv("openaiKey"))

def retorna_cotacao(ticker,period="1mo"):
    ticker_obj = yf.Ticker(f"{ticker}.SA")
    hist = ticker_obj.history(period=period)["Close"]
    hist.index = hist.index.strftime("%Y-%m-%d")
    hist = round(hist,2)

    if len(hist)> 30:
        slice_size = int(len(hist)/ 30)
        hist = hist.iloc[::slice_size][::-1]

    return hist.to_json()


tools = [{
    "type":"function",
    "function":{
        "name":"retorna_cotacao",
        "description":"Retorna a cotação de ações da Ibovespa",
        "parameters":{
            "type":"object",
            "properties":{
                "ticker":{
                    "type":"string",
                    "description":"o ticker da ação. EX:BBAS3,BBDC4,etc"
                },
                "period":{
                    "type":"string",
                    "description":"Periodo retornado dos dados histŕicos da cotação \
                        sendo '1mo' equivale a um mês,'1d' equivale a 1 dia \
                            e '1y' equivale a um ano e 'ytd' equivale a todos os tempos",
                    "enum":["1mo","1d","5d","6mo","1y","ytd"]
                }
            }
        }
    }
}]


funcao_disponivel = {"retorna_cotacao":retorna_cotacao}

mensagens = [{"role":"user","content":"qual e a cotação do banco do brasil no ultimo ano?"}]


resposta = client.chat.completions.create(model="gpt-3.5-turbo-0125",messages=mensagens,tools=tools,tool_choice='auto')

tools = resposta.choices[0].message.tool_calls

if tools:
    mensagens.append(resposta.choices[0].message)
    for tool_call in tools:
        function_name = tool_call.function.name
        function_to_call = funcao_disponivel[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_return = function_to_call(**function_args)

        mensagens.append({
            "tool_call_id":tool_call.id,
            "role":"tool",
            "name":function_name,
            "content":function_return
        })

        segunda_resposta = client.chat.completions.create(model="gpt-3.5-turbo-0125",messages=mensagens)
        mensagens.append(segunda_resposta.choices[0].message)
        print(segunda_resposta.choices[0].message.content)

