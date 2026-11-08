"""Motor aislado: se ejecuta con el Python del entorno de Jupyter."""
import contextlib
import json
import os
import sqlite3
import pickle
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELOS = {}
CLIENTES = {}

def leer(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def reparar_configuracion_vacia(carpeta):
    """Compatibilidad de índices heredados con Chroma 0.5.23; conserva metadatos."""
    from chromadb.api.configuration import (
        CollectionConfigurationInternal, ConfigurationParameter, HNSWConfigurationInternal,
    )
    db = carpeta / 'chroma.sqlite3'
    with sqlite3.connect(str(db)) as conexion:
        filas = conexion.execute('SELECT id, config_json_str FROM collections').fetchall()
        vacias = [identificador for identificador, config in filas
                  if config is not None and json.loads(config) == {}]
        if not vacias:
            return
        respaldo = db.with_suffix('.sqlite3.bak')
        if not respaldo.exists():
            with sqlite3.connect(str(respaldo)) as copia:
                conexion.backup(copia)
        for identificador in vacias:
            metadatos = {}
            for clave, texto, entero, real in conexion.execute(
                'SELECT key, str_value, int_value, float_value FROM collection_metadata WHERE collection_id = ?',
                (identificador,),
            ):
                if clave.startswith('hnsw:'):
                    metadatos[clave] = texto if texto is not None else entero if entero is not None else real
            hnsw = HNSWConfigurationInternal.from_legacy_params(metadatos)
            config = CollectionConfigurationInternal([
                ConfigurationParameter('hnsw_configuration', hnsw)
            ]).to_json_str()
            conexion.execute('UPDATE collections SET config_json_str = ? WHERE id = ?',
                             (config, identificador))
        print('Configuración heredada vacía reparada; respaldo SQLite conservado.', file=sys.stderr)


def reparar_metadatos_vectoriales(carpeta):
    """Convierte solo metadatos de diccionario heredados, conservando respaldo."""
    from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData
    class LectorRestringido(pickle.Unpickler):
        def find_class(self, modulo, nombre):
            if modulo == PersistentData.__module__ and nombre == 'PersistentData':
                return PersistentData
            raise ValueError('Clase no admitida en los metadatos del índice.')
    with sqlite3.connect(str(carpeta / 'chroma.sqlite3')) as conexion:
        segmentos = conexion.execute(
            'SELECT segments.id, collections.dimension FROM segments '
            'JOIN collections ON segments.collection = collections.id '
            "WHERE segments.scope = 'VECTOR'"
        ).fetchall()
    for segmento, dimension in segmentos:
        archivo = carpeta / segmento / 'index_metadata.pickle'
        if not archivo.exists():
            continue
        with archivo.open('rb') as entrada:
            datos = LectorRestringido(entrada).load()
        if not isinstance(datos, dict):
            continue
        campos = ['dimensionality', 'total_elements_added', 'id_to_label', 'label_to_id', 'id_to_seq_id']
        if not all(campo in datos for campo in campos) or not dimension:
            raise ValueError('Los metadatos vectoriales necesitan una revisión manual.')
        respaldo = archivo.with_suffix('.pickle.bak')
        if not respaldo.exists():
            shutil.copy2(archivo, respaldo)
        valores = {campo: datos[campo] for campo in campos}
        valores['dimensionality'] = valores['dimensionality'] or dimension
        persistencia = PersistentData(**valores)
        persistencia.max_seq_id = datos.get('max_seq_id')
        temporal = archivo.with_suffix('.pickle.tmp')
        temporal.write_bytes(pickle.dumps(persistencia))
        temporal.replace(archivo)

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
        reparar_configuracion_vacia(carpeta / 'chroma')
        reparar_metadatos_vectoriales(carpeta / 'chroma')
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
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'ok':False,'error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False),flush=True)
