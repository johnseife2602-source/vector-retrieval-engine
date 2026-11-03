from main import query
from IndexFlat_engine import VectorStore
from IndexMMap_engine import MMapVectorStore
import pytest
import random
from metrics import (dot_product, 
                    cosine_similarity,
                    euclidean_distance,
                    calc_norm,
                    normalize)

def make_vector(dim=128):
    return [ random.uniform(-10.0,10.0) for _ in range(dim)]

def test_dot_product():
    v1=[2,3,4,5]
    v2=[3,4,5,6]
    result=dot_product(v1,v2)
    assert result==pytest.approx(68), f"dot product is not working with vectors {v1} and {v2}"

def test_normalization_invariant():
    for _ in range(50):
        v=make_vector(64)
        norm_v=normalize(v)

        assert calc_norm(norm_v)==pytest.approx(1.0), "normalization invariants are not working"


def test_euclidean_similarity():
    a=make_vector(dim=32)
    b=make_vector(dim=32)

    assert euclidean_distance(a,b) == pytest.approx(euclidean_distance(b,a))

@pytest.fixture
def populated_store():
    store=VectorStore(dim=32)
    for i in range(100):
        store.insert(f"doc_{i}", make_vector(dim=32), {"index": i})

    return store

def test_top_k_search(populated_store):
    query=make_vector(dim=32)
    results=populated_store.search(query)

    assert len(results)==5
    scores=[ score for score,_,_ in results]
    assert scores == sorted(scores, reverse=True)

def test_mmap_vector_store():
    query=make_vector(dim=3)
    with MMapVectorStore() as mmap_store:
        results=mmap_store.search(query, top_k=3, metric="cosine")
        scores = [item["score"] for item in results]
        assert scores == sorted(scores, reverse=True)

    assert len(results)==3
    
@pytest.mark.parametrize("v1, v2 , expected",[ 
    ([1.0,0.0],[0.0,1.0], 0.0), #orthogonal vectors
    ([1.0,1.0],[1.0,1.0], 1.0), # identical vectors
    ([1.0, 0.0], [-1.0, 0.0], -1.0), # opposite vectors
    ([0.0, 0.0], [1.0, 2.0], 0.0), # zero vector guard

])
def test_cosine_edge_cases(v1,v2,expected):
    assert cosine_similarity(v1,v2)==pytest.approx(expected)

@pytest.mark.parametrize("v1, v2 , expected",[ 
    ([1.0,0.0],[0.0,1.0], 0.0), #orthogonal vectors
    ([1.0,1.0],[1.0,1.0], 2.0), # identical vectors
    ([1.0, 0.0], [-1.0, 0.0], -1.0), # opposite vectors
    ([0.0, 0.0], [1.0, 2.0], 0.0), # zero vector guard

])
def test_dot_product_edge(v1,v2,expected):
    assert dot_product(v1,v2)==pytest.approx(expected)

def test_save_load_roundtrip(tmp_path):
    test_bin=tmp_path/"test_vectors.bin"
    test_jsonl=tmp_path/"test_meta.jsonl"

    store=VectorStore(dim=32)
    for i in range(50):
        store.insert(f"doc_{i}", make_vector(dim=32), {"index": f"meta_{i}"})
    store.save(test_bin, test_jsonl)

    assert test_bin.exists()
    assert test_jsonl.exists()

    new_store=VectorStore()
    new_store.load(test_bin, test_jsonl)
    assert len(new_store) ==50
    assert new_store.dim== 32




    