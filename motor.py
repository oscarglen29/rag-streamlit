"""Motor aislado: se ejecuta con el Python del entorno de Jupyter."""
import contextlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELOS = {}
CLIENTES = {}

def leer(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def buscar(p):
    import chromadb
    from sentence_transformers import SentenceTransformer
    version = p['version']
    if version not in ('v5', 'v6'):
        raise ValueError('Versión desconocida.')
    carpeta = ROOT / 'indices' / version
    manifiesto = leer(carpeta / 'manifiesto.jsonl')[0]
    config = manifiesto['configuracion']
    llave = (config['modelo'], config['revision'])
    if llave not in MODELOS:
        MODELOS[llave] = SentenceTransformer(llave[0], revision=llave[1], device='cpu', local_files_only=os.environ.get('OEFA_OFFLINE', 'false').lower() == 'true')
    modelo = MODELOS[llave]
    if len(modelo.tokenizer.encode(p['pregunta'])) > modelo.max_seq_length:
        raise ValueError('La pregunta supera el límite del modelo de embeddings. Acórtala.')
    if version not in CLIENTES:
        CLIENTES[version] = chromadb.PersistentClient(path=str(carpeta / 'chroma'))
    coleccion = CLIENTES[version].get_collection(manifiesto['coleccion'], embedding_function=None)
    filtro = {} if p.get('filtro_id') is None else {'where': {'id_resolucion': int(p['filtro_id'])}}
    disponibles = len(coleccion.get(include=[], **filtro)['ids'])
    if not disponibles:
        raise ValueError('No hay fragmentos para la resolución seleccionada.')
    vector = modelo.encode([p['pregunta']], normalize_embeddings=True)
    datos = coleccion.query(query_embeddings=vector.tolist(), n_results=min(p['top_k'],disponibles),
                            include=['documents','metadatas','distances'], **filtro)
    hechos = {h['codigo']:h for h in leer(carpeta/'hechos.jsonl')} if version=='v5' else {}
    evidencias = []
    for texto,meta,distancia in zip(datos['documents'][0],datos['metadatas'][0],datos['distances'][0]):
        contexto = ''
        if version=='v5':
            relacionados = [hechos[c] for c in meta['codigos_hecho'].split(' | ')]
            contexto = '\n'.join(f"{h['codigo']}: {h['descripcion']}. Decisión: {h['decision']} ({h['estado_revision']})." for h in relacionados)
            decisiones = {d['unidad']:d for h in relacionados for d in h['decisiones']}
            contexto += '\n' + '\n'.join(f"Artículo {d['articulo']}, páginas {d['paginas']}: {d['disposicion']}" for d in decisiones.values())
        evidencias.append(dict(texto=texto,contexto_hechos=contexto,similitud=1-float(distancia),**meta))
    return dict(evidencias=evidencias,coleccion=coleccion.name,modelo_embeddings=llave[0],revision_embeddings=llave[1],disponibles=disponibles)

if __name__=='__main__':
    for linea in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                resultado = buscar(json.loads(linea))
            print(json.dumps({'ok':True,'resultado':resultado},ensure_ascii=False),flush=True)
        except Exception as exc:
            print(json.dumps({'ok':False,'error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False),flush=True)
