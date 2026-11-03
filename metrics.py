import math

def dot_product(u,v):
    assert len(u)==len(v), f'incompatible vector dimensions to compute the dot product between {u} and {v}'
    result= sum(i*j for i, j in zip(u,v))
    return result

def calc_norm(v):
    norm= math.sqrt(sum(i*i for i in v))
    return norm

def normalize(v):
    norm=calc_norm(v)
    if norm==0.0:
        return [0.0] * len(v)
    return[i/norm for i in v]

def cosine_similarity(u,v):
    u_norm=calc_norm(u)
    v_norm=calc_norm(v)
    if v_norm ==0.0 or u_norm==0.0:
        return 0.0 # there is zero shared semantic signal
    return dot_product(u,v)/(u_norm*v_norm)

def euclidean_distance(u,v):
    assert len(u) == len(v), f"Vector dimension mismatch: {len(u)} vs {len(v)}"
    distance=math.sqrt(sum((i-j)**2 for i,j in zip(u,v)))
    return distance




