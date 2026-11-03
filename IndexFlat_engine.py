"""
This is the first benchmark engine i built from scratch, but it can not handle
big number of vectors, since it loads everything in memory. and looping through 
every vector in big data-base is too slow. so i will build another engine that solves those issues.
i called that engine "IndexMMap_engine.py".
"""

import struct
import json
import heapq
from pathlib import Path
from metrics import (dot_product, 
                    cosine_similarity,
                    euclidean_distance)


class VectorStore:
    def __init__(self,dim=None):
        self.dim=dim
        self.vectors={}
        self.metadata={}
        self.base_dir=Path(__file__).resolve().parent

    def insert(self,id_,vector, metadata=None):
        if self.dim is None:
            self.dim=len(vector)
        else:
            assert len(vector)==self.dim, f"Vector dimension mismatch: expected {self.dim}, got {len(vector)}"

        self.vectors[id_]=vector
        self.metadata[id_]=metadata if metadata is not None else {}

    def get(self,id_):
        if id_ in self.vectors:
            return (self.vectors[id_], self.metadata[id_])
        return None

    def __len__(self):
        return len(self.vectors)

    def search(self, query_vector, top_k=5, metric="cosine"):
        assert len(query_vector) == self.dim, f"Query dimension {len(query_vector)} != index dimension {self.dim}"
        heap=[]
        for doc_id, doc_vector in self.vectors.items():
            if metric =="cosine":
                score=cosine_similarity(query_vector,doc_vector)
            elif metric == "dot":
                score=dot_product(query_vector,doc_vector)
            elif metric =="euclidean":
                score=-euclidean_distance(query_vector,doc_vector)

            else:
                raise ValueError(f"Unknown metric: {metric}")

            if len(heap)< top_k:
                heapq.heappush(heap,(score,doc_id))
            elif score > heap[0][0]:
                heapq.heappushpop(heap, (score, doc_id))

        results=[]
        for score, doc_id in sorted(heap, key=lambda x: x[0], reverse=True): 
            final_score=-score if metric =="euclidean" else score
            results.append(
                {
                    "id":doc_id,
                    "score": final_score,
                    "metadata": self.metadata[doc_id]
                }
            )
        return results
    
    def save(self, bin_path=None, jsonl_path=None):

        bin_path=self.base_dir/"data"/"vector_data.bin" if bin_path ==None else Path(bin_path)
        jsonl_path=self.base_dir/"data"/"meta_data.jsonl" if jsonl_path ==None else Path(jsonl_path)
        bin_path.parent.mkdir(parents=True, exist_ok=True)
       
        count=len(self.vectors)
        Header=struct.pack("4sII", b"VEC1", self.dim, count)
        with open(bin_path, "wb") as f_bin, open(jsonl_path,"w", encoding="utf-8") as f_jsonl:
            f_bin.write(Header)
         
            for id_, vector in self.vectors.items():  
                f_bin.write(struct.pack(f"{self.dim}f", *vector))
                #In RAM only (converts Python dict -> JSON str & writes nothing to disk)
                line=json.dumps({"id": id_, "metadata": self.metadata[id_]})
                #Disk I/O (writes that string + a newline character to the file)
                f_jsonl.write(line + "\n")

    def load(self, bin_path=None , jsonl_path=None):
        bin_path=self.base_dir/"data"/"vector_data.bin" if bin_path==None else Path(bin_path)
        jsonl_path=self.base_dir/"data"/"meta_data.jsonl" if jsonl_path ==None else Path(jsonl_path)

        if jsonl_path.exists() and bin_path.exists():
          with open(jsonl_path, "r", encoding="UTF-8") as f_jsonl, open(bin_path, "rb") as f_bin:
            header_bytes=f_bin.read(12)
            magic,dim,count=struct.unpack("4sII", header_bytes)

            assert magic==b"VEC1", "Invalid file format: Magic bytes do not match 'VEC1' "
            self.dim=dim
            for line in f_jsonl:
                if line.strip():
                    record=json.loads(line)
                    self.metadata[record["id"]]=record["metadata"]

                    raw_byte=f_bin.read(dim*4)
                    vector=list(struct.unpack(f"{dim}f", raw_byte))

                    self.vectors[record["id"]]=vector

        else:
          raise FileNotFoundError(f"Database files not found at {bin_path} and {jsonl_path}")

