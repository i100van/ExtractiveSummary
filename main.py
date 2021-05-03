from semantic_graph import SemanticGraph
import pandas as pd
import nltk
import tools as tools

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


if __name__ == '__main__':
    # Settings del lanzamiento
    book_path='data/lord_ring.pdf'
    output_path='data/summary.md'
    compresion=0.01
    num_caps=23
    num_topics=5

    #Lectura del texto en pdf
    full_text=tools.read_txt(book_path)
    graphs=list()

    #Separamos el texto en capitulos y procesamos cada capitulo
    splitted=full_text.split('Chapter')
    table = list()
    for cap in range(2,num_caps):
        #Creamos un grafo para cada capitulo, junto a su factor de compresion
        graphs.append(SemanticGraph(splitted[cap],compresion))
        row = ['Char_' + str(cap), splitted[cap]]
        table.append(row)

    #Creamos un dataframe con filas por capitulos para LDA
    df=pd.DataFrame(table)
    df.columns=['chapter','text']
    lda=tools.LDA_analysis(df, num_topics)
    tools.output_summary(output_path, graphs, lda)














