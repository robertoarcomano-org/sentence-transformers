"""
Ispeziona passo-passo cosa fa model.encode() internamente:
tokenizzazione -> embedding dei token -> transformer -> mean pooling -> normalizzazione
"""

import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

frase = "Procedura per cambiare la password dell'account senza cambiare ruolo"

# I sentence-transformers sono composti da moduli in sequenza.
# model[0] è il Transformer HuggingFace (tokenizer + modello)
# model[1] è il layer di Pooling
print("Moduli del modello:")
for i, m in enumerate(model):
    print(f"  [{i}] {m}")

transformer_module = model[0]
tokenizer = transformer_module.tokenizer
auto_model = transformer_module.auto_model  # il modello HF vero e proprio

# ---------------------------------------------------------------
# STEP 1: Tokenizzazione
# ---------------------------------------------------------------
encoded = tokenizer(
    frase,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

token_ids = encoded["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("\n--- STEP 1: Tokenizzazione ---")
print("Frase originale :", frase)
print("Token           :", tokens)
print("Token IDs       :", token_ids.tolist())
print("Attention mask  :", encoded["attention_mask"][0].tolist())
print("N. token        :", len(tokens))

# ---------------------------------------------------------------
# STEP 2 + 3: Embedding iniziali + passaggio nel Transformer (12 layer)
# ---------------------------------------------------------------
with torch.no_grad():
    output = auto_model(**encoded)

# output.last_hidden_state ha shape (batch=1, n_token, hidden_size=384)
token_embeddings = output.last_hidden_state

print("\n--- STEP 2/3: Embedding contestuali dei singoli token ---")
print("Shape token_embeddings:", token_embeddings.shape)  # (1, n_token, 384)

for tok, vec in zip(tokens, token_embeddings[0]):
    print(f"  {tok:15s} -> vettore di dim {vec.shape[0]} | primi 5 valori: {vec[:5].tolist()}")

# ---------------------------------------------------------------
# STEP 4: Mean pooling (media dei token embeddings, pesata dall'attention mask)
# ---------------------------------------------------------------
attention_mask = encoded["attention_mask"][0].unsqueeze(-1)  # (n_token, 1)

# maschero i token di padding, poi sommo e divido per il numero di token reali
sommati = (token_embeddings[0] * attention_mask).sum(dim=0)
n_token_reali = attention_mask.sum(dim=0)
mean_pooled_manuale = sommati / n_token_reali

print("\n--- STEP 4: Mean pooling manuale ---")
print("Shape vettore frase:", mean_pooled_manuale.shape)  # (384,)
print("Primi 5 valori     :", mean_pooled_manuale[:5].tolist())

# ---------------------------------------------------------------
# STEP 5: Normalizzazione L2
# ---------------------------------------------------------------
embedding_normalizzato_manuale = torch.nn.functional.normalize(
    mean_pooled_manuale.unsqueeze(0), p=2, dim=1
)[0]

print("\n--- STEP 5: Normalizzazione L2 ---")
print("Norma prima  :", mean_pooled_manuale.norm().item())
print("Norma dopo   :", embedding_normalizzato_manuale.norm().item())  # deve essere ~1.0

# ---------------------------------------------------------------
# CONFRONTO: il risultato manuale coincide con model.encode()?
# ---------------------------------------------------------------
embedding_ufficiale = model.encode(frase, convert_to_tensor=True, normalize_embeddings=True)

print("\n--- CONFRONTO con model.encode() ---")
print("Embedding ufficiale (primi 5 valori):", embedding_ufficiale[:5].tolist())
print("Embedding manuale   (primi 5 valori):", embedding_normalizzato_manuale[:5].tolist())

differenza = torch.dist(embedding_ufficiale, embedding_normalizzato_manuale).item()
print(f"\nDistanza euclidea tra i due vettori: {differenza:.8f}  (deve essere ~0)")