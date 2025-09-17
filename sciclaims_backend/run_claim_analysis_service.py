import configparser

import flask

from flask import abort, jsonify, request

from sciclaims_backend.modeling.utils import (
    init_llm,
    init_searcher,
    load_system_prompts,
    init_verification_dataset,
)
from sciclaims_backend.processing.utils import run_claim_analysis

seed = 42
# TODO:= add seed

cfg = configparser.ConfigParser()
cfg.read("service_config.ini")

searcher = init_searcher(cfg)
llm, tokenizer, sampling_params = init_llm(cfg)
verification_dataset = init_verification_dataset(cfg)
system_prompts = load_system_prompts(cfg)

ssl_context = None if "ssl" not in cfg else (cfg["ssl"]["crt"], cfg["ssl"]["key"])
app = flask.Flask(__name__)
app.config["DEBUG"] = False
port = cfg["config"]["port"]


@app.route("/services/claim_analysis", methods=["GET"])
def home():
    return "<h1>Claim Analysis API</h1><p> To be constructed </p>"


@app.route("/services/claim_analysis", methods=["POST"])
def claim_analysis():
    text = request.data.decode("utf-8")
    n_colbert_results = request.args.get(
        "max_verification_results", default=3, type=int
    )
    if n_colbert_results < 1 or n_colbert_results > 10:
        abort(
            400,
            "Not valid max_verification_results value. "
            "Please set a number between 1 and 10",
        )
    output = run_claim_analysis(
        text=text,
        system_prompts=system_prompts,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        searcher=searcher,
        verification_dataset=verification_dataset,
        n_colbert_results=n_colbert_results,
    )
    return jsonify(output)


if ssl_context:
    app.run(host="0.0.0.0", port=port, ssl_context=ssl_context)
else:
    app.run(host="0.0.0.0", port=port)
