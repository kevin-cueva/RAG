from typing import Union
from fastapi import FastAPI, HTTPException
from src.Utils.Embeddings import Embeddings
from src.Utils.ConsultaGpt import ConsultaGpt
from src.Utils.extraer_datos_pinecone import extraer_datos_pinecone
from src.Utils.pinecone_class import pinecone_clase
from src.Domain.datos_usuario import datos_usuario

import ast
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
@app.post("/busqueda-rag")
async def pinecone_busqueda_chatgpt(datos: datos_usuario):
    """
Endpoint para realizar una búsqueda utilizando RAG (Retrieval-Augmented Generation).
Este servicio expuesto recibe datos del usuario, genera un prompt para ChatGPT, obtiene sugerencias de propiedades,
y realiza una búsqueda vectorial en la base de datos utilizando Pinecone. Retorna tanto la respuesta sugerida por ChatGPT
como los resultados de la búsqueda vectorial.
Args:
    datos (datos_usuario): Datos enviados por el usuario para realizar la consulta.
Returns:
    dict: Un diccionario con dos claves:
        - "respuesta_sugerida_chatGPT": Respuesta generada por ChatGPT basada en el prompt.
        - "response_bd_vectorial": Resultados obtenidos de la búsqueda vectorial en la base de datos.
Raises:
    HTTPException: Si ocurre algún error durante el procesamiento, retorna un error 500 con el detalle del mismo.
"""
    try:
        prompt = ConsultaGpt.redactar_prompt(datos.dict())
        respuesta_sugerida_chatGPT = await ConsultaGpt().Sugerencias_propiedades(promp_de_busqueda=prompt)
        
        diccionario_sugerencia_chatGPT = json.loads(respuesta_sugerida_chatGPT)
        prompt_busqueda = pinecone_clase.prompt_busqueda(diccionario_sugerencia_chatGPT)
        
        generar_embedding = await Embeddings().generador_embeddings(prompt_busqueda)
        buscador_datos_DB_vectorial = await pinecone_clase().buscar_datos(query_formato_embedding=generar_embedding, top_k=3)
        
        response_bd_vectorial = json.loads(json.dumps(buscador_datos_DB_vectorial, default=str))
        vector_search_results = [ast.literal_eval(item) for item in response_bd_vectorial]
        response = {
            "respuesta_sugerida_chatGPT": diccionario_sugerencia_chatGPT,
            "response_bd_vectorial": vector_search_results
        }
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))