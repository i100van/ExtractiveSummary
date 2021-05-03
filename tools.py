import sys

import pandas as pd
from tika import parser
import gensim
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import gensim.corpora as corpora

def read_txt(path:str)->str:
    '''
    Lector del pdf haciendo uso de Apache Tika
    Input: Ruta del fichero pdf
    Output: texto parseado
    '''
    raw = parser.from_file(path)
    res = raw['content'].replace('\n',' ')
    return res

def sent_to_words(sentences:str)-> str:
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts:str)-> list:
    return [[word for word in gensim.utils.simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

def LDA_analysis(df:pd.DataFrame,num_topics:int)->gensim.models.LdaMulticore:
    '''
    Haciendo uso de la libreria gensim realizamos un analisis LDA de cada capitulo
    Input: dataframe con los capitulos como textos y el numero de topics a extraer
    Output: Modelo entrenado
    '''
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    data = df.text.values.tolist()
    data_words = list(sent_to_words(data))
    data_words = remove_stopwords(data_words)
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word,num_topics=num_topics)
    return lda_model


def output_summary(output_path:str, graphs:list, lda:gensim.models.LdaMulticore):
    '''
    Creacion del texto output, con los centroides asociados al factor de compresion,
    las frases con mayor numero de estos centroides, y los resultados de LDA
    Input: path de salida, lista de grafos, y modelo lda entrenado
    Output: fichero resumen de los cpaitulos.
    '''
    sys.stdout = open(output_path, "w", encoding='UTF8')
    print('# JRR TOLKIEN LA COMUNIDAD DEL ANILLO')
    for i,cap in enumerate(graphs):
        print('## Capitulo',i,'')
        print("### Centroides asociados al factor de compresión ",cap.get_compresion(),":")
        print(cap.get_centroids())
        centroides=list()
        for item in cap.get_centroids():
            centroides.append(item[0])
        print("### Resumen:")
        for phrase in cap.get_summary():
            for palabra in phrase.split(" "):
                if palabra in centroides:
                    print('**',palabra,'**',end=' ',sep='')
                else:
                    print(palabra,end=' ')
            print('\n')
        print("### LDA:\n")
        for idx, topic in lda.print_topics(-1):
            print("**Topic**: {} \n**Words**: {}".format(idx, topic))
            print("\n")
    sys.stdout.close()