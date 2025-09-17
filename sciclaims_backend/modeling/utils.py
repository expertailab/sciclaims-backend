from typing import Dict

import huggingface_hub
import srsly
from transformers import AutoTokenizer

from sciclaims_backend.processing.es_indexing import ESSearcher


def init_llm(cfg):
    from vllm import LLM, SamplingParams

    huggingface_hub.login(token=cfg["llm"]["hf_token"])
    model = LLM(model=cfg["llm"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["model_name"])
    sampling_params = SamplingParams(
        temperature=float(cfg["llm"]["temperature"]),
        top_p=float(cfg["llm"]["top_p"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
        logprobs=0,
    )
    return model, tokenizer, sampling_params


def init_searcher(cfg: Dict):
    if "colbert" in cfg:
        searcher_args = {"type": "colbert", "args": cfg["colbert"]}
    elif "elastic" in cfg:
        searcher_args = {"type": "elastic", "args": cfg["elastic"]}
    else:
        raise ValueError(
            "You have to configure at least one searcher, elastic or colbert"
        )
    if searcher_args["type"] == "colbert":
        from colbert import Searcher

        searcher = Searcher(**searcher_args["args"])
    else:
        searcher = ESSearcher(**searcher_args["args"])
    return searcher


def load_system_prompts(cfg):
    system_prompts = {}
    for k, v in cfg["system_prompts"].items():
        with open(cfg["system_prompts"][k], "r") as file:
            system_prompts[k] = file.read()
    return system_prompts


def init_verification_dataset(cfg):
    jsonl_path = cfg["verification_dataset"]["path"]
    verification_dataset = {}
    for doc in srsly.read_jsonl(jsonl_path):
        doc["abstract"] = " ".join(doc["abstract"])
        verification_dataset[doc["doc_id"]] = doc
    return verification_dataset