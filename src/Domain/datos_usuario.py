from pydantic import BaseModel

class datos_usuario(BaseModel):
    edad: int
    estadoCivil: str
    hijos: int
    presupuesto: str
    intereses: str