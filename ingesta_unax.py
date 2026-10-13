import sys                               # leer el argumento de la terminal
import re                                # detectar patrones de texto (titulos, fechas, etc)
import pdfplumber                        # leer texto y tablas de PDFs
import pytesseract                       # OCR (imagen -> texto)
from pdf2image import convert_from_path  # convierte paginas PDF en imagenes
import numpy as np                       # calculo de vectores y matrices

# Convierte una tabla en texto
def tabla_a_texto(tabla):
    filas = []
    for fila in tabla:
        celdas = [celda if celda else "" for celda in fila]
        filas.append(" | ".join(celdas))
    return "\n".join(filas)

# Hace OCR a una pagína escaneada y devuelve el texto
def ocr_paginas(ruta_pdf, numero_pagina):
    imagenes = convert_from_path(ruta_pdf, first_page=numero_pagina, last_page=numero_pagina)
    return pytesseract.image_to_string(imagenes[0], lang="spa")

# Ingesta PDF, Recorre todas las paginas y decide como leerla
def ingesta_pdf(ruta_pdf):
    texto_total= ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for i, pagina in enumerate(pdf.pages, start=1):
            # si es none, usa texto vacio
            texto = pagina.extract_text() or ""
            if len(texto.strip()) < 20:
                print(f"Pagina {i}: escaneada usando OCR")
                texto = ocr_paginas(ruta_pdf, i)
            else:
                print(f"Pagina {i}: texto simple")
                
            # Es lo mismo que texto_total = texto_total + ...
            texto_total += texto + "\n\n"
            
            # extraer tablas con texto real(simple)
            # los "\n\n" son  para hacer 2 saltos de linea
            for tabla in pagina.extract_tables():
                texto_total += "Tabla:\n" + tabla_a_texto(tabla) + "\n\n"
                
    return texto_total
# CHUNKING

# Fixed, cortar n caracteres, sin mirar el contenido
def chunking_fixed(texto, tamano=500):
    chunks = []
    for inicio in range(0, len(texto), tamano):
        chunks.append(texto[inicio: inicio + tamano])
    return chunks

# Overlap, cada chunk repite el final del anterior
def chunking_overlap(texto, tamano=500, solape=100):
    chunks=[]
    paso = tamano - solape
    for inicio in range(0, len(texto), paso):
        chunks.append(texto[inicio:inicio + tamano])
    return chunks

# Recursive, separador mas "natural" 
def chunking_recursive(texto, tamano=500, separadores=None):
    if separadores is None:
        separadores= ["\n\n", "\n", ". ", " "]
        
    # Caso base El texto entra en 1 Chunk
    if len(texto) <= tamano:
        return [texto]
    
    # Caso Base No quedan separadores, se corta
    if not separadores:
        return chunking_fixed(texto, tamano)
    
    separador = separadores[0]
    partes = texto.split(separador)
    
    chunks = []
    actual = ""
    for parte in partes:
        # Si al anadir la parte nos pasamos, guardamos lo que tenemos
        if len(actual) +  len(parte) + len(separador) > tamano and actual:
            chunks.append(actual)
            actual = ""
        # Si la parte es demasiado grande, la cortamos con el separador
        if len(parte) > tamano:
            chunks.extend(chunking_recursive(parte, tamano, separadores[1 :]))
        else:
            actual += parte + separador
            
    if actual.strip():
        chunks.append(actual)
            
    return chunks

# Por Structura, detecta las diferentes secciones de los textos(titulos + contenido)
def es_titulo(linea):
    linea = linea.strip()
    # Rechaza lineas vacias o con > 80 caracteres
    if not linea or len(linea) > 80:
        return False
    # Acepta numeraciones tipo 1. ,1.2 ...
    if re.match(r"^(\d+\.)+\s*\w+", linea) or re.match(r"^\d+\s+[A-Z]", linea) :
        return True
    # Acepta lineas que empiezan por Capitulo, Aticulo ...
    if linea.lower().startswith(("capitulo", "capítulo", "apartad", "articulo", "artículo")):
        return True
    # acepta lineas escritas en mayusculas
    return linea.isupper()

# Por estructura, un chunk por cada seccion (titulo + contenido)
def chunking_estructura(texto):
    chunks = []
    actual = ""
    en_tabla = False
    
    for linea in texto.split("\n"):
        # cierra el chunk anterior y empieza la tabla
        if linea.startswith("Tabla:"):
            if actual.strip():
                chunks.append(actual.strip())
            actual = linea + "\n"
            en_tabla = True
        # linea vacia para terminar la tabla 
        elif en_tabla and linea.strip() == "":
            chunks.append(actual.strip())
            actual = ""
            en_tabla = False
        # titulo empieza nueva seccion
        elif not en_tabla and es_titulo(linea):
            if actual.strip():
                chunks.append(actual.strip())
            actual = linea + "\n"
        else:
            # cualquier otra cosa se añade al chunk actual
            actual += linea + "\n"
    if actual.strip():
        chunks.append(actual.strip())
    return chunks

#Semantico, cortar cuando cambia de tema, no cuando alcanza un tamaño
def chunking_semantico(texto, umbral=0.5):
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Dividir en frases por . y salto de linea
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+|\n+", texto) if f.strip()]
    if not frases:
        return[]
    
    # Un Vector por frase
    vectores = modelo.encode(frases, normalize_embeddings=True)
    
    chunks = []
    actual = [frases[0]]
    for i in range(1, len(frases)):
        parecido = float(np.dot(vectores[i - 1], vectores[i]))
        if parecido < umbral:
            chunks.append(" ".join(actual))
            actual = []
        actual.append(frases[i])
    chunks.append(" ".join(actual))
    return chunks
    
    # impreme cantidad de chunks salieron y ejemplo
def mostrar(nombre, chunks):
    print(f"\n=== {nombre}: {len(chunks)} chunks ===")
    if chunks:
        print("Primer Chunk:", chunks[0] [:200].replace("\n", " "), "...")
            
def main ():
    if len(sys.argv) < 2:
        print ("Uso: python ingesta_pdf.py documento.pdf")
        return
    ruta_pdf = sys.argv[1]
    
    # Ingesta
    texto = ingesta_pdf(ruta_pdf)
    print(f"\nTexto extraido: {len(texto)} caracteres")
    
    # chunks con las 5 estrategias
    mostrar("Fixed", chunking_fixed(texto))
    mostrar("Overlap", chunking_overlap(texto))
    mostrar("Recursive", chunking_recursive(texto))
    mostrar("Estructura", chunking_estructura(texto))
    mostrar("Semantico", chunking_semantico(texto))
    
    # para ver cuantas tablas ha encontrado en el pdf
    print("Tablas detectadas:", texto.count("Tabla:"))
    
    # Ver una tabla
    # print(texto[texto.find("Tabla:"):][:400])


if __name__ == "__main__":
    main()