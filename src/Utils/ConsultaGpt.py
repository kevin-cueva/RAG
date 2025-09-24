from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

class ConsultaGpt:
    def __init__(self):
        load_dotenv()
        
        
    async def Consulta_Gpt5_nano(self):
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
            model="gpt-5-nano",
                #temperature=0.3,
                #max_completion_tokens=100,
                messages=
                [
                    {"role": "system",
                    "content": """Hablas en español, todas tus respuestas
                    estan en rima continua, crea respuestas breves de dos parrafos"""},

                    {"role": "user",
                    "content": "Cuentame como prepararme un huevo"}
                ]
            )
            return response.choices[0].message

        finally:
            print()

    async def Sugerencias_propiedades(self, promp_de_busqueda: str):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            client = AsyncOpenAI(api_key= api_key)
            response = await client.chat.completions.create(
            model="gpt-5-nano",
                #temperature=0.3,
                messages=
                [
                    {"role": "system",
                    "content": """Eres un sistema de asesoría de bienes raíces, analizas las 
                    características de una persona y sugieres la propiedad que mejor se ajuste a sus 
                    necesidades, respondes con 4 parámetros:

                    - ubicacion: con una sugerencia entre montaña, playa o ciudad
                    - tamano: con una sugerencia del tamaño de la propiedad entre grande, mediano o pequeño
                    - precio: con una sugerencia de precio en euros
                    - sugerencia: un breve texto con los detalles de la recomendación enfocados en los intereses del usuario.
                    Usa un formato similar a "Una propiedad cerca de [ubicacion] de tamaño [tamano] puede interesarte por [beneficios]".

                    Devuelve una respuesta en formato JSON con los 4 parámetros solicitados
                    """},

                    {"role": "user",
                    "content": promp_de_busqueda}
                ],
                response_format={"type": "json_object"},
                #max_completion_tokens=220,#No funciona
            )
            return response.choices[0].message.content

        finally:
            print()    
    
    # En condiciones normales debe hacer una limpieza y validación de los datos de entrada
    # para evitar inyecciones de prompt, pero se omite por simplicidad
    def redactar_prompt(datos_usuario):
        return (
            f"Analiza qué tipo de propiedad se ajusta a las necesidades de una persona "
            f"de {datos_usuario['edad']} años, {datos_usuario['estadoCivil']}, "
            f"con {datos_usuario['hijos']} hijos, "
            f"con un presupuesto {datos_usuario['presupuesto']} "
            f"y cuyos intereses son: {datos_usuario['intereses']}"
        )
