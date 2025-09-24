from pinecone import Pinecone
from dotenv import load_dotenv
import os
class pinecone_clase:
    def __init__(self):
        # Cargar variables desde .env
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("NAME_INDEX_PINECONE")
        # Inicializar conexión con Pinecone
        self.pc = Pinecone(api_key=api_key)
         # Conectar al índice
        try:
            self.index = self.pc.Index(index_name)
        except Exception as e:
            raise RuntimeError(f"Error conectando al índice {index_name}: {e}")
        
    def almacenar_datos(self, data: list):
        """Insertar o actualizar embeddings en el índice"""
        try:
            self.index.upsert(vectors=data)
            print('Datos almacenados en Pinecone')
        except Exception as e:
            raise RuntimeError('Error al almacenar los embeddings en Pinecone', e)
    async def buscar_datos(self, query_formato_embedding, top_k: int = 3):
        """Buscar los embeddings más similares en el índice"""
        try:
            response = self.index.query(
                vector=query_formato_embedding, # Asumiendo que query_formato_embedding es un vector
                top_k=top_k, # Número de resultados similares a devolver
                include_metadata=True, # Incluir metadatos en la respuesta 
                include_values=False # Incluir valores de los vectores en la respuesta (opcional)
            )
            return response['matches'] # Resultado mas similares
        except Exception as e:
            raise RuntimeError('Error al buscar los embeddings en Pinecone', e)
         

