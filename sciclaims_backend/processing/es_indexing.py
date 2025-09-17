import argparse

import srsly
from elasticsearch import Elasticsearch
from tqdm import tqdm


class ESSearcher:
    def __init__(
        self,
        endpoint,
        es_user,
        es_pwd,
        index_name,
    ):
        self.es_client = Elasticsearch(endpoint, basic_auth=(es_user, es_pwd))
        self.index_name = index_name

    def search(self, sentence, k=10):
        resp = self.es_client.search(
            index=self.index_name,
            query={"multi_match": {"query": sentence, "fields": ["title", "abstract"]}},
        )
        ids = []
        ranks = []
        scores = []
        for i, hit in enumerate(resp["hits"]["hits"][:k]):
            ids.append(int(hit["_id"]))
            ranks.append(i)
            scores.append(hit["_score"])
        return [ids, ranks, scores]

    def index(self, title, abstract, doc_id):
        self.es_client.index(
            index=self.index_name,
            id=int(doc_id),
            document={
                "title": title,
                "abstract": abstract,
            },
        )


def parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--jsonl_path",
        help="path to the jsonl file with the publications",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--es_endpoint",
        help="Endpoint to the Elasticsearch instance",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--es_user",
        help="elastic username",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--es_pwd",
        help="elastic password",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--index_name",
        help="name for the index",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    print(args)
    return args


def main(
    jsonl_path: str,
    es_endpoint: str,
    es_user: str,
    es_pwd: str,
    index_name: str,
):
    searcher = ESSearcher(
        endpoint=es_endpoint, es_user=es_user, es_pwd=es_pwd, index_name=index_name
    )
    ds = srsly.read_jsonl(jsonl_path)
    for doc in tqdm(ds):
        searcher.index(
            title=doc["title"], abstract=" ".join(doc["abstract"]), doc_id=doc["doc_id"]
        )

    print("Index completed")
    print(searcher.search("Hello world"))


if __name__ == "__main__":
    main(**vars(parse_flags()))
