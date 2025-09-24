from pinecone import Pinecone
from .pinecone_class import pinecone_clase
from .Embeddings import Embeddings

class extraer_datos_pinecone:
    def __init__(self):
        self.pinecone_class = pinecone_clase()
        self.embeddings = Embeddings()
    
    async def extraer_datos(self, promp_de_busqueda: str = "una casa ubicada en ciudad con un máximo de dos habitaciones"):
        # Generar embedding para el prompt de búsqueda
        busqueda_formato_embeddings = await self.embeddings.generador_embeddings(promp_de_busqueda)
        
        # Realizar la consulta en Pinecone
       
        propiedades_encontradas = await self.pinecone_class.buscar_datos(query_formato_embedding = busqueda_formato_embeddings, top_k=3)
        return propiedades_encontradas



