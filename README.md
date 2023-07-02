# ia368v_dd_final

UNICAMP-IR dataset creation, for passage retrieval evaluation in pt-BR, based on the Clueweb22-pt dataset and a set of pt-BR-written questions. The whole working pipeline is depicted below:

![UNICAMP-IR semi-automatic creation pipeline](Pipeline%20UNICAMP-IR.png)

<br/>

Check [the final report](UNICAMP_IR_final_report_20230702.pdf) for details on the work execution.


## Notebooks and source code description

### Retrieval pipelines work
* [Create BM25 index.ipynb](Create%20BM25%20index.ipynb): This notebook creates a BM25 index over the pre-processed Clueweb22-pt corpus, and also executes the [validation queries](queries_validation.tsv) created.
* [searching_BM25_MT5.ipynb](searching_BM25_MT5.ipynb): This notebook performs the [validation queries](queries_validation.tsv) retrieval using BM25 as the first stage, and mT5 model as reranker.
* [colbertx_fine_tune_48_000.ipynb](searching_BM25_MT5.ipynb): Performs the ColBERT-X base model fine-tuning, using the first 1M triplets of the [mMARCO-pt dataset](https://huggingface.co/datasets/unicamp-dl/mmarco).
* [split_clueweb22_pt_10M.ipynb](split_clueweb22_pt_10M.ipynb): Split the pre-processed Clueweb22-pt dataset in 2M-passage chunks, converting each one of them to the ColBERT-X input format. This is required to keep the ColBERT-X index creation at reasonable memory budgets.
* [create_clueweb22_pt_colbertx_index.ipynb](create_clueweb22_pt_colbertx_index.ipynb): Create the ColBERT-X indexes (including the approximate search using FAISS) over a given chunk of the Clueweb22-pt dataset.
* [test_colbertx_part_00_index.ipynb](test_colbertx_part_00_index.ipynb): [Validation queries](queries_validation.tsv) retrieval using ColBERT-X index over the first Clueweb22-pt part.
* [colbertx_train_retrieval.ipynb](colbertx_train_retrieval.ipynb): [Train queries](train_queries.tsv) retrieval using ColBERT-X over all the Clueweb22-pt parts.
* [handle_colbertx_validation_retrievals.ipynb](handle_colbertx_validation_retrievals.ipynb): merge the ColBERT-X validation queries retrieval performed over the Clueweb22-pt corpus parts.
* [handle_colbertx_train_retrievals.ipynb](handle_colbertx_train_retrievals.ipynb): merge the ColBERT-X train queries retrieval performed over the Clueweb22-pt corpus parts.

### LLM passage revelance evaluation
* [check_LLM_evaluations.ipynb](check_LLM_evaluations.ipynb): Compare different LLMs performance for passage relevance evaluation.
* [LLM_query_passage_evaluation.ipynb](LLM_query_passage_evaluation.ipynb): Execute LLM passage relevance evaluation.
* [analysis_LLM_validation_queries_evaluation.ipynb](analysis_LLM_validation_queries_evaluation.ipynb): Analize the LLM passage relevance evaluations, comparing with [human evaluations].(human_validation_LLM_evaluations_eduseiti.tsv).
* [common_tools.py](common_tools.py): Common functions to interact with OpenAI API for LLMs usage.

### LLM queries creation
* [LLM_question_creation.ipynb](LLM_question_creation.ipynb): Queries creation using LLMs.
