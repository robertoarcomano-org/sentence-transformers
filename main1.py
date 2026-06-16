from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Modello embedding (scaricato una volta, poi locale)
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# I tuoi documenti
documenti = [
    "Come richiedere un rimborso per un prodotto acquistato",
    "Procedura per cambiare la password dell'utente",
    "Metodi di pagamento disponibili con carta",
    "Politica di restituzione degli articoli entro trenta giorni",
    "Come aprire una richiesta di assistenza tecnica"
]


# Creo i vettori
embeddings = model.encode(
    documenti,
    normalize_embeddings=True
)


# Creo indice FAISS
dimensione = embeddings.shape[1]

indice = faiss.IndexFlatIP(dimensione)

indice.add(
    np.array(embeddings)
)


print("Documenti indicizzati:", indice.ntotal)

query = "restituire un prodotto"


# trasformo la query in embedding
query_vector = model.encode(
    [query],
    normalize_embeddings=True
)


# cerco i più vicini
k = 3

scores, risultati = indice.search(
    np.array(query_vector),
    k
)


for score, posizione in zip(scores[0], risultati[0]):
    print(
        round(float(score), 3),
        "->",
        documenti[posizione]
    )