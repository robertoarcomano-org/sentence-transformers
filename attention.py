import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

dim=2

# 3 parole, vettori di dimensione 4
np.random.seed(0)
Q = np.random.rand(3, 2)
K = np.random.rand(3, 2)
V = np.random.rand(3, 2)

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

d_k = Q.shape[-1]
print("d_k:", d_k)

scores = Q @ K.T    
print("Scores:\n", scores)              # (3, 3)
scores_scaled = scores / np.sqrt(d_k)
weights = softmax(scores_scaled)  # (3, 3) - pesi di attenzione
output = weights @ V              # (3, 4) - output finale

print("Pesi di attenzione:\n", weights)
print("Output:\n", output)