# ia368v_dd_final

UNICAMP-IR dataset creation, for passage retrieval evaluation in pt-BR, based on the Clueweb22-pt dataset and a set of pt-BR-written questions. The whole working pipeline is depicted below:

![UNICAMP-IR semi-automatic creation pipeline](Pipeline%20UNICAMP-IR.png)

## Notebooks and source code description

* [Create BM25 index.ipynb](Create%20BM25%20index.ipynb): This notebook creates a BM25 index over the pre-processed Clueweb22-pt corpus, and also executes the [validation queries](queries_validation.tsv) created.
* [searching_BM25_MT5.ipynb](searching_BM25_MT5.ipynb): This notebook performs the [validation queries](queries_validation.tsv) retrieval using BM25 as the first stage, and mT5 model as reranker.
* [colbertx_fine_tune_48_000.ipynb](searching_BM25_MT5.ipynb): Performs the ColBERT-X base model fine-tuning, using the first 1M triplets of the [mMARCO-pt dataset](https://huggingface.co/datasets/unicamp-dl/mmarco).
* [split_clueweb22_pt_10M.ipynb](split_clueweb22_pt_10M.ipynb): Split the pre-processed Clueweb22-pt dataset in 2M-passage chunks, converting each one of them to the ColBERT-X input format. This is required to keep the ColBERT-X index creation at reasonable memory budgets.
* [create_clueweb22_pt_colbertx_index.ipynb](create_clueweb22_pt_colbertx_index.ipynb): Create the ColBERT-X indexes (including the approximate search using FAISS) over a given chunk of the Clueweb22-pt dataset.
