from sentence_transformers import SentenceTransformer, util

# Modello locale per embedding
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Piccolo archivio documenti
documenti = [
    "Come richiedere un rimborso per un acquisto",
    "Procedura per cambiare la password dell'account",
    "Modalità di pagamento tramite carta di credito",
    "Regole per la restituzione dei prodotti acquistati"
]

# La domanda dell'utente
query = "Voglio restituire un articolo che ho comprato"

# Creo gli embedding dei documenti
embeddings_documenti = model.encode(
    documenti,
    convert_to_tensor=True
)

# Creo embedding della query
embedding_query = model.encode(
    query,
    convert_to_tensor=True
)

# Calcolo similarità semantica
similarita = util.cos_sim(
    embedding_query,
    embeddings_documenti
)[0]


# Ordino i risultati dal più simile
risultati = sorted(
    zip(documenti, similarita),
    key=lambda x: x[1],
    reverse=True
)


for testo, score in risultati:
    print(f"{score:.3f} -> {testo}")