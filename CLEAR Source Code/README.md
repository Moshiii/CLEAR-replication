# CLEAR
---

This repo is the source code of the paper CLEAR: Contrastive Learning for API Recommendation.

# Files
---

sentence bert train_bi_encoder.ipynb is the training script of the CLEAR-filter model.

sentence bert train_cross_encoder.ipynb is the training script of the CLEAR-re-rank model.

The training data is in the data folder. "BIKER_train.QApair.csv" is the original BIKER dataset. 

Pocessed data for training and validation of CLEAR is in full_data_min_5_max_10_ir_10 folder.

Trianed model can be found at models/drive-download-20210422T032447Z-001.zip(moved to [here]() due to file limit)

retrieve_rerank_simple_method_level.ipynb is for method-level inference.
retrieve_rerank_simple_class_level.ipynb is for class-level inference.
