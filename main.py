from typing import Union
from fastapi import FastAPI, HTTPException
from src.Utils.Embeddings import Embeddings
from src.Utils.ConsultaGpt import ConsultaGpt
from src.Utils.extraer_datos_pinecone import extraer_datos_pinecone
import json
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/embedding/")
async def generar_embedding():
    try:
        with open("./src/Resources/propiedades.json") as archivo:
            texto = json.load(archivo)
        vector = await Embeddings().procesar_lista_embeddings()
        return {"texto": texto, "embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/consultas-api-gpt5-nano")
async def consulta_gpt():
    try:
        response = await ConsultaGpt().Consulta_Gpt5_nano()
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/consultas-api-gpt5-nano/prompt")
async def consulta_gpt_prompt():
    try:
        datos = {
            "edad": 35,
            "estadoCivil": "casado",
            "hijos": 2,
            "presupuesto": "medio",
            "intereses": "deporte, naturaleza y tranquilidad"
        }
        prompt = ConsultaGpt.redactar_prompt(datos)
        response = await ConsultaGpt().Sugerencias_propiedades(promp_de_busqueda=prompt)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/pinecone/guardar-datos")
async def pinecone_guardar_datos():
    try:
        response = await Embeddings().guardar_datos()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pinecone/extraer-datos")
async def pinecone_extraer_datos(promp_de_busqueda: str = "una casa ubicada en ciudad con un máximo de dos habitaciones"):
    try:
        response = await extraer_datos_pinecone().extraer_datos(promp_de_busqueda)

        # fuerza conversión a JSON-friendly
        import json
        response = json.loads(json.dumps(response, default=str))

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
