import struct
import json
import heapq
import mmap
from pathlib import Path
from metrics import (dot_product, 
                    cosine_similarity,
                    euclidean_distance)

class MMapVectorStore:
    def __init__(self, bin_path=None, jsonl_path=None):
        base = Path(__file__).resolve().parent 
        self.bin_path = Path(bin_path) if bin_path is not None else base / "data" / "vector_data.bin"
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else base / "data" / "meta_data.jsonl"

        self.f_bin = None
        self.mm = None

        self.dim = 0
        self.count = 0

    def open(self):
        if not self.bin_path.exists() or not self.jsonl_path.exists():
            raise FileNotFoundError(f"Database files not found at {self.bin_path} and {self.jsonl_path}")
        self.f_bin = open(self.bin_path, "rb")
        self.mm = mmap.mmap(self.f_bin.fileno(), length=0, access=mmap.ACCESS_READ)
        magic, self.dim, self.count = struct.unpack("4sII", self.mm[0:12])
        assert magic == b"VEC1", "Corrupted file: Magic header must be b'VEC1'"

        self.bytes_per_vector = self.dim * 4

        self.id_map = []
        self.metadata = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f_jsonl:
            for line in f_jsonl:
                if line.strip():
                    record = json.loads(line)
                    doc_id = record["id"]
                    self.id_map.append(doc_id)
                    self.metadata[doc_id] = record["metadata"]

        assert len(self.id_map) == self.count, f"Sync error: {self.count} vectors in binary != {len(self.id_map)} in JSONL"

    def get_vector(self, index):
        assert 0 <= index < self.count, f"Index {index} out of bounds (count={self.count})"

        start = 12 + (index * self.bytes_per_vector)
        end = start + self.bytes_per_vector

        raw_bytes = self.mm[start:end]
        return list(struct.unpack(f"{self.dim}f", raw_bytes))

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None

        if self.f_bin is not None:
            self.f_bin.close()
            self.f_bin = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def search(self, query_vector, top_k=5, metric="cosine"):
        assert len(query_vector) == self.dim, f"Query dim {len(query_vector)} != index dim {self.dim}"
        heap = []
        for i in range(self.count):
            doc_vector = self.get_vector(i)

            if metric == "cosine":
                score = cosine_similarity(query_vector, doc_vector)
            elif metric == "dot":
                score = dot_product(query_vector, doc_vector)
            elif metric == "euclidean":
                score = -euclidean_distance(query_vector, doc_vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if len(heap) < top_k:
                heapq.heappush(heap, (score, i))
            elif score > heap[0][0]:
                heapq.heappushpop(heap, (score, i))
        
        results = []
        for s, i in sorted(heap, key=lambda x: x[0], reverse=True):
            doc_id = self.id_map[i]
            final_score = -s if metric == "euclidean" else s

            results.append(
                {
                    "id": doc_id,
                    "score": final_score,
                    "metadata": self.metadata[doc_id]
                }
            )
        return results
