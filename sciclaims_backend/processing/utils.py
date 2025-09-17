import json
import math
import re
from collections import Counter
from json import JSONDecodeError
from typing import Dict, List

import json_repair
import numpy as np
import regex


def extract_claims(text, system_prompt, model, tokenizer, sampling_params, use_tqdm):
    user_prompt = f"""

        TEXT: {text}

        OUTPUT:

        """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = model.generate([prompt], sampling_params, use_tqdm=use_tqdm)
    generated_text = outputs[0].outputs[0].text
    lines = generated_text.split("\n")
    claims = []
    original_texts = []
    for line in lines:
        split_line = line.split("--")
        if len(split_line) > 1:
            claim = split_line[1].strip()
            claims.append(claim)
            original_texts.append(text)
    return claims, original_texts


def clean_generated_claims(
    claim_list,
    original_texts,
    clean_prompt,
    model,
    tokenizer,
    sampling_params,
    use_tqdm=True,
):
    prompts = []
    for claim, original_text in zip(claim_list, original_texts):
        user_prompt = f"""
                        CLAIM: {claim}
                        PASSAGE: {original_text}
                        """
        messages = [
            {"role": "system", "content": clean_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    outputs = model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    refined_claims = []
    for output, claim in zip(outputs, claim_list):
        response = output.outputs[0].text
        json_response = json_repair.loads(response)
        try:
            refined_claims.append(json_response["refined_claim"])
        except (TypeError, KeyError):
            print(f"Error with response {response}. Using original claim.")
            refined_claims.append(claim)
    return refined_claims


def generate_claims(
    text, system_prompt, clean_prompt, model, tokenizer, sampling_params, use_tqdm=True
):
    claims, original_texts = extract_claims(
        text=text,
        system_prompt=system_prompt,
        model=model,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
    )
    print(claims)
    refined_claims = clean_generated_claims(
        claim_list=claims,
        original_texts=original_texts,
        clean_prompt=clean_prompt,
        model=model,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
    )
    print(refined_claims)
    return [{"claim": x, "claim_type": "generated"} for x in refined_claims]


def get_claim_analysis_prompt(claim, search_result, system_prompt, tokenizer):
    user_prompt = f"""

                    CLAIM: {claim}

                    """
    user_prompt += "EVIDENCE: \nTitle: {}\nText: {}\n\n".format(
        search_result["title"], search_result["abstract"]
    )
    user_prompt += 'OUTPUT: {{"output": "<INSERT OUTPUT HERE>"}}'
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def format_report(text):
    pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    json_dicts = pattern.findall(text)
    try:
        report = json.loads(json_dicts[0])
    except (JSONDecodeError, IndexError):
        report = {}
    return report


def find_sub_list(sublist, full_list):
    sub_list_length = len(sublist)
    for ind in (i for i, e in enumerate(full_list) if e == sublist[0]):
        if full_list[ind : ind + sub_list_length] == sublist:
            return ind, ind + sub_list_length


def get_confidence(log_probs, output_tokens, label_tokens):
    label_span = find_sub_list(sublist=label_tokens, full_list=output_tokens)
    linear_probs = []
    for label_token_index, log_prob_index in enumerate(
        range(label_span[0], label_span[1])
    ):
        label_token_id = label_tokens[label_token_index]
        log_prob = log_probs[log_prob_index][label_token_id].logprob
        linear_prob = np.round(np.exp(log_prob) * 100, 2)
        linear_probs.append(linear_prob)
    confidence = np.round(np.mean(linear_probs), decimals=2)
    return confidence


def generate_analysis(
    claims, system_prompt, model, tokenizer, sampling_params, label_tokens_dict
):
    prompts = []
    for claim_dict in claims:
        claim = claim_dict["claim"]
        search_results = claim_dict["search_results"]
        for search_result in search_results:
            prompt = get_claim_analysis_prompt(
                claim=claim,
                search_result=search_result,
                system_prompt=system_prompt,
                tokenizer=tokenizer,
            )
            prompts.append(prompt)
    outputs = model.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        claim_index = int(i / len(claim_dict["search_results"]))
        search_index = i % len(claim_dict["search_results"])
        search_results = claims[claim_index]["search_results"][search_index]
        report = format_report(output.outputs[0].text)
        if "response" in report and report["response"] != "NEI":
            report["confidence"] = get_confidence(
                log_probs=output.outputs[0].logprobs,
                output_tokens=output.outputs[0].token_ids,
                label_tokens=label_tokens_dict[report["response"]],
            )
            search_results["report"] = report
    final_claims = []
    for claim_dict in claims:
        out_claim_dict = {}
        for k, v in claim_dict.items():
            if k == "search_results":
                out_claim_dict["claim_analysis"] = [x for x in v if "report" in x]
            else:
                out_claim_dict[k] = v
        if len(out_claim_dict["claim_analysis"]) > 0:
            final_claims.append(out_claim_dict)

    return final_claims


def text_to_vector(text):
    words = re.compile(r"\w+").findall(text)
    return Counter(words)


def get_cosine_sim(str1, str2):
    vec1 = text_to_vector(str1)
    vec2 = text_to_vector(str2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def filter_similar_strings(claims, cos_threshold):
    final_output = []
    added_claims = [x["claim"] for x in claims if x["claim_type"] == "extracted"]
    for claim_dict in claims:
        if claim_dict["claim_type"] == "extracted":
            final_output.append(claim_dict)
        else:
            valid_claim = True
            for claim in added_claims:
                cos_sim = get_cosine_sim(claim, claim_dict["claim"])
                if cos_sim > cos_threshold:
                    valid_claim = False
                    break
            if valid_claim:
                final_output.append(claim_dict)
                added_claims.append(claim_dict["claim"])
    return final_output


def print_report(text: str, claims: List[Dict]):
    report = json.dumps({"input_text": text, "claims": claims}, indent=4)
    print(report)


def run_claim_analysis(
    text: str,
    system_prompts: Dict,
    llm,
    tokenizer,
    sampling_params,
    searcher,
    verification_dataset,
    n_colbert_results: int,
):
    claim_candidates = generate_claims(
        text=text,
        system_prompt=system_prompts["claim_extraction"],
        clean_prompt=system_prompts["claim_clean"],
        model=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
    )
    print(claim_candidates)
    for i, claim_dict in enumerate(claim_candidates):
        searcher_response = searcher.search(claim_dict["claim"], k=n_colbert_results)
        verified_ids = searcher_response[0]
        verified_ranks = searcher_response[1]
        verified_scores = [x / 100 for x in searcher_response[2]]
        search_results = []
        for v_id, v_score, v_rank in zip(verified_ids, verified_scores, verified_ranks):
            search_res = {k: v for k, v in verification_dataset[v_id].items()}
            search_res["score"] = v_score
            search_res["rank"] = v_rank
            search_results.append(search_res)
        claim_dict["id"] = i
        claim_dict["search_results"] = search_results
    label_tokens_dict = {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in ["SUPPORT", "REFUTE"]
    }
    claims = generate_analysis(
        claims=claim_candidates,
        system_prompt=system_prompts["claim_analyst"],
        model=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        label_tokens_dict=label_tokens_dict,
    )
    claims = filter_similar_strings(claims=claims, cos_threshold=0.5)[:10]
    print_report(text=text, claims=claims)
    return claims