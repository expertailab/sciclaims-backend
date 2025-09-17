# sciclaims-backend

How to run the service:

- Create and configure a new conda environment:
  - conda create -n sciclaims python==3.9.0
  - conda activate sciclaims
  - pip install -r requirements.txt
- Populate your elasticsearch index with sciclaims_backend/es_indexing.py. Arguments:
  - jsonl_path: a jsonl file with the verification datasets. It must have titles and abstracts. Our dataset will be available once the anonymization process is over.
  - es_endpoint: the endpoint of your elasticsearch instance. Usually http://localhost:9020.
  - es_user: elastic search user.
  - es_pwd: elastic search password.
  - index_name: select the name of your index.
- Fill the fields in service_config.ini
- Run run_claim_analysis_service.py