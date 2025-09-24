import json
import aiohttp
import os
from .pinecone_class import pinecone_clase
from dotenv import load_dotenv

class Embeddings:
    def __init__(self):
        # Cargar variables desde .env
        load_dotenv()
        
    
    async def generador_embeddings(self, texto):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "text-embedding-3-small",
                "input": texto
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    return embedding
                
        except Exception as e:
            print(f"Error al generar embedding: {e}")
            return None
        
    async def procesar_lista_embeddings(self):
        # Carpeta actual de este archivo (scrip.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir un nivel (de Utils a src) y luego entrar en Resources
        ruta_json = os.path.join(base_dir, "..", "Resources", "propiedades.json")
        # Normalizar la ruta
        ruta_json = os.path.normpath(ruta_json)

        with open(ruta_json, encoding="utf-8") as archivo:
            propiedades = json.load(archivo)
            
        for propiedad in propiedades:
            texto = (
                f"{propiedad['titulo']}, ubicación en la {propiedad['ubicacion']} "  # Cambiado a propiedad
                f"con {propiedad['habitaciones']} habitaciones y {propiedad['banos']} baños. "  # Cambiado a propiedad
                f"Tiene un tamaño {propiedad['tamano']} de {propiedad['metrosCuadrados']} metros cuadrados "  # Cambiado a propiedad
                f"y un precio de {propiedad['precio']} euros."  # Cambiado a propiedad
            )
            
            embedding = await self.generador_embeddings(texto)
            propiedad["values"] = embedding

        productos_formato_embedding = [
            {
                "id": producto["id"],
                "values": producto["values"],
                "metadata": {"titulo": producto["titulo"]},
            }
            for producto in propiedades
        ]

        return productos_formato_embedding
    
    async def guardar_datos(self):
        datos_procesados = await self.procesar_lista_embeddings()
        pinecone_clase().almacenar_datos(datos_procesados)