from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import spacy

nlp = spacy.load('en_core_web_sm')

class SemanticGraph:

    def __init__(self, text:str,compresion:float):
        '''
        Inicializador del grafo, junto al proceso necesario de obtencion del grafo.
        Calculo de centroides, seleccion de frases resumen.
        Input: EL texto y el factor de compresion
        Output: Objeto con el grafo, resumen y centroides.
        '''
        doc = nlp(text)
        self.compresion=compresion
        self.sentences = [sent.string.strip() for sent in doc.sents]
        self.n_frases = int(self.compresion * len(self.sentences))
        self.name = self.sentences[1]
        self.graph = self.__create_graph()
        self.centroids= self.__get_k_centroids()
        self.summary= self.__create_summary()


    def to_plot(self):
        '''
        Dibujar el plot del grafo
        '''
        pos = nx.spring_layout(self.graph)
        edge_labels = nx.get_edge_attributes(self.graph, 'verb')
        nx.draw(self.graph, pos, with_labels=True, width=1.0)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

    def __get_k_centroids(self)-> list:
        '''
        Calcula los centroides correspondientes al factor de compresion seleccionado
        '''
        k_centroids = int(self.compresion*self.graph.number_of_nodes())
        if k_centroids <= 0:
            k_centroids=1
        node_degree=dict(self.graph.degree())
        sor={k: v for k, v in sorted(node_degree.items(), key=lambda item: item[1],reverse=True)}
        sorted_items=sor.items()
        return list(sorted_items)[:k_centroids]

    def __oldcreate_summary(self)-> str:
        indices=list()
        for key in self.centroids:
            for i,sentence in enumerate(self.sentences):
                if(key[0] in sentence):
                    indices.append(i)
        indices=set(indices)
        indices=sorted(list(indices))

        summary=''
        for ind in indices:
            summary=' '.join([summary,self.sentences[ind]])
        return summary

    def __create_summary(self)-> str:
        '''
        Creacion del resumen en base a los centroides
        y al numero de frases a seleccionar asociado el factor de compresion
        '''
        aux=list()
        for s in self.sentences:
            for c in self.centroids:
                if c[0] in s:
                    aux.append(s)
        counter=Counter(aux)
        sor ={k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
        sor_key= sor.keys()
        return  list(sor_key)[:self.n_frases]

    def __create_graph(self)-> nx.DiGraph():
        '''
        Creacion del grafo semantico en base al codigo de Carlos
        '''
        conjunto_relaciones = set()
        for sentence in self.sentences:
            tokens = nlp(sentence)
            noun_list = []
            verb_list = []
            for token in tokens:
                if token.pos_ in ['NOUN', 'PROPN']:
                    noun_list.append(token.lemma_)
                if token.pos_ in ['VERB','AUX']:
                    verb_list.append(token.lemma_)
            str_verb = "  "
            for verb in verb_list:
                str_verb+=verb+"/"
            for i in range(len(noun_list)):
                for j in range(i+1,len(noun_list)):
                    conjunto_relaciones.add((noun_list[i], noun_list[j], str_verb[:-1]))
            G = nx.DiGraph()
            for parent, child, verbs in conjunto_relaciones:
                G.add_edge(parent, child, verb=verbs)
        return G

    def get_graph(self)-> nx.DiGraph():
         return self.graph

    def get_name(self)->str:
         return self.name

    def get_sentences(self)->list:
        return self.sentences

    def get_graph_info(self)->list:
        return nx.info(self.graph)

    def get_centroids(self)->list:
        return self.centroids

    def get_summary(self)->str:
        return self.summary

    def get_n_frases(self)->int:
        return self.n_frases

    def get_compresion(self)->float:
        return self.compresion

