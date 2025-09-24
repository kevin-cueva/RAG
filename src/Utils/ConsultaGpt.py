from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

class ConsultaGpt:
    def __init__(self):
        load_dotenv()
    async def Consulta_Gpt5_nano(self):
        api_key = os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)
        try:
            response = await client.chat.completions.create(
            model="gpt-5-nano",
                #temperature=0.3,
                #max_tokens=100,
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
