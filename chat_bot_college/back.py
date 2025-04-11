from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_openai import ChatOpenAI
import os
import dotenv
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA,BaseRetrievalQA
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

api_key = os.getenv("openaiKey")

chat = ChatOpenAI(model="gpt-4o",api_key=api_key)

def processing_files() -> str:
    arquivos = ["https://www.iesp.edu.br/institucional/a-faculdade"
                ,"https://www.iesp.edu.br/polos"
                ,"https://www.iesp.edu.br/institucional/biblioteca"
                ,"https://www.iesp.edu.br/institucional/coopere",
                "https://www.iesp.edu.br/institucional/cpa",
                "https://www.iesp.edu.br/institucional/ceua",
                "https://www.iesp.edu.br/institucional/enade",
                "https://www.iesp.edu.br/institucional/comite-de-etica",
                "https://www.iesp.edu.br/institucional/dce-e-das",
                "https://www.iesp.edu.br/institucional/dirigentes",
                "https://www.iesp.edu.br/institucional/setores-e-telefones",
                "https://www.iesp.edu.br/institucional/dpo-lgpd",
                "https://www.iesp.edu.br/ingresso/inscreva-se",
                "https://www.iesp.edu.br/cursos/graduacao/modalidade/presencial",
                "https://www.iesp.edu.br/cursos/graduacao/administracao",
                "https://www.iesp.edu.br/cursos/graduacao/arquitetura-e-urbanismo",
                "https://www.iesp.edu.br/cursos/graduacao/biomedicina",
                "https://www.iesp.edu.br/cursos/graduacao/ciencias-contabeis",
                "https://www.iesp.edu.br/cursos/graduacao/ciencias-da-computacao",
                "https://www.iesp.edu.br/cursos/graduacao/design-de-interiores",
                "https://www.iesp.edu.br/cursos/graduacao/design-grafico",
                "https://www.iesp.edu.br/cursos/graduacao/direito",
                "https://www.iesp.edu.br/cursos/graduacao/educacao-fisica",
                "https://www.iesp.edu.br/cursos/graduacao/enfermagem",
                "https://www.iesp.edu.br/cursos/graduacao/engenharia-civil",
                "https://www.iesp.edu.br/cursos/graduacao/estetica-e-cosmetica",
                "https://www.iesp.edu.br/cursos/graduacao/farmacia",
                "https://www.iesp.edu.br/cursos/graduacao/fisioterapia",
                "https://www.iesp.edu.br/cursos/graduacao/gestao-comercial",
                "https://www.iesp.edu.br/cursos/graduacao/gestao-de-recursos-humanos",
                "https://www.iesp.edu.br/cursos/graduacao/gestao-financeira",
                "https://www.iesp.edu.br/cursos/graduacao/medicina-veterinaria",
                "https://www.iesp.edu.br/cursos/graduacao/nutricao",
                "https://www.iesp.edu.br/cursos/graduacao/odontologia",
                "https://www.iesp.edu.br/cursos/graduacao/producao-publicitaria",
                "https://www.iesp.edu.br/cursos/graduacao/psicologia",
                "https://www.iesp.edu.br/cursos/graduacao/publicidade-e-propaganda",
                "https://www.iesp.edu.br/cursos/graduacao/sistemas-para-internet",
                "https://www.iesp.edu.br/cursos/graduacao/modalidade/semipresencial",
                "https://www.iesp.edu.br/cursos/graduacao/licenciatura-em-pedagogia-semipresencial-",
                "https://www.iesp.edu.br/cursos/graduacao/modalidade/ead",
                "https://www.iesp.edu.br/cursos/graduacao/analise-e-desenvolvimento-de-sistemas-ead-",
                "https://www.iesp.edu.br/cursos/graduacao/gestao-da-tecnologia-da-informacao-ead-",
                "https://www.iesp.edu.br/cursos/graduacao/logistica-ead-",
                "https://www.iesp.edu.br/cursos/graduacao/marketing-ead-",
                "https://www.iesp.edu.br/cursos/pos-graduacao/modalidade/presencial",
                "https://www.iesp.edu.br/cursos/pos-graduacao/modalidade/ead",
                "https://www.iesp.edu.br/cursos/cursos-livres",
                "https://www.iesp.edu.br/cursos/cursos-livres",
                "https://www.iesp.edu.br/servicos/internacionalizacao",
                "https://www.iesp.edu.br/servicos/nucleo-de-carreiras",
                "https://www.iesp.edu.br/servicos/estagios",
                "https://www.iesp.edu.br/servicos/nups",
                "https://www.iesp.edu.br/servicos/clinicaescola",
                "https://www.iesp.edu.br/servicos/area-do-egresso"]

    arquivos_lidos = ""

    for arquivo in arquivos:
        loader = WebBaseLoader(arquivo)
        docs = loader.load()
        arquivos_lidos += docs[0].page_content

    
    return arquivos_lidos


def split_text() -> list[str]:

    arquivos = processing_files()
    split = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap = 50,
        separators=["\n\n","\n","","."," "]
    )

    text = split.split_text(text=arquivos)

    return text

def embedding_text()-> FAISS:

    text = split_text()

    embedding = HuggingFaceHubEmbeddings(huggingfacehub_api_token=os.getenv("huggin"),model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vector_db = FAISS.from_texts(texts=text,embedding=embedding)

    return vector_db



def retriever() ->BaseRetrievalQA:
    RAG_PROMPT_TEMPLATE = """Use o contexto abaixo **E** seu conhecimento geral para responder:
            
            {context}
            
            Pergunta: {question}

            Caso a questão não seja baseada no contexto, utilize seus conhecimento gerais para responder a pergunta do usuário.


            args:
                    institucional:str = Field(description="retorna link sobre instituicional. link:https://www.iesp.edu.br/institucional/. use os examples para saber o que colocar apos o ultimo /",
                                examples=["centro universitário:no link inves de centro universitario coloque a-faculdade,biblioteca,estrutura,polos,coopere,publicacoes,cpa(comissão Própria de avaliação),ceua(comitê de ética na utilização animal),enade,dce-e-das,dirigentes,setores-e-telefones,dpo-lgpd(oficial de proteção de dados),calendarios,projetos,noticias"])
                    cursos_presencial:str = Field(description="retorna link sobre cursos ou cursos presencias. link: https://www.iesp.edu.br/cursos/graduacao/modalidade/presencial")
                    cursos_semipresencial:str = Field(description="retorna link sobre cursos semipresenciais. link:https://www.iesp.edu.br/cursos/graduacao/modalidade/semipresencial")
                    cursos_ead:str = Field(description="retorna link sobre os cursos ead. link: https://www.iesp.edu.br/cursos/graduacao/modalidade/ead")
                    cursos_posgraducao_presencial:str = Field(description="retorna link sobre cursos de pos graduação ou pos graduação presencial. link: https://www.iesp.edu.br/cursos/pos-graduacao/modalidade/presencial")
                    cursos_posgraduacao_ead:str = Field(description="retorna link sobre cursos pos graduação ead. link:https://www.iesp.edu.br/cursos/pos-graduacao/modalidade/ead")
                    servicos:str = Field(description="retorna link sobre os serviços do site. link:https://www.iesp.edu.br/servicos/. use os examples para saber o que colocar apos o ultimo /",
                            examples=["educação-corporativa,internacionalizacao,nucleo-de-carreiras,estagios,nups,clinicaescola"])
                    inscrevase:str = Field(description="retorna link sobre increvase. link:https://www.iesp.edu.br/ingresso/inscreva-se. use os examples para saber quais são os assunstos/ tipos de inscrição.",
                            examples=["matricula,enem,vestibular online,2 graduação,vestibular online,retorno ao curso,bolsas e financiamentos,consulte seu resultado"])

            IMPORTANTE: utilize os args para responder a pergunta do usuário quando achar necessário fornecer estes links para o usuário.
            CUIDADO: Não forneça sites que nao sejam sobre o assunto perguntado pelo usuário e formate adequadamente a resposta. exemplo: [link](url) nao deixe '\n\n' no final dos links por exemplo.     

            Resposta (seja natural e combine informações quando necessário):"""

    prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    vector_db = embedding_text()

    retriever = RetrievalQA.from_chain_type(
                llm=chat,
                retriever=vector_db.as_retriever(search_type="mmr"),
                chain_type="stuff",
                chain_type_kwargs={"prompt":prompt_template},
                verbose=True
            )
    
    return retriever