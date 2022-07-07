# DistilmBERT-and-XLM-RoBERTa-for-Multilingual-Toxic-Comment-Classification
As user-generated content on the internet increases, so does the spread of toxic comments. There is no formal definition of hate and abusive speeches but, the threat of abuse and harassment online means that many people limit expressing themselves and give up on seeking different opinions. Therefore, detecting toxic comments becomes an active research area, and it is often handled as a text classification task. As recent popular methods for text classification tasks, classification tasks, pre-trained language model-based methods are at the forefront of Natural Language Processing, achieving state-of-the-art performance on various NLP tasks. However, there is a paucity of studies using such methods on toxic comment classification. In this work, we study how to make use of pre-trained language model-based methods for toxic comment classification and the performances of different pre-trained language models on these tasks. This study introduces an ensemble approach, where we have made use of pre-trained models - DistilmBert and xlm-roberta-large-xnli to perform the toxic comment classification task. We trained our model on an English dataset and tested it on Wikipedia talk page comments in several languages and have achieved an accuracy of over 93%. 
In order to test these models you can make submission on https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification.

##Models

The works in the above three notebooks can be explained as:
### 1) DistilmBERT
The DistilBERT model is the product of Transfer Learning approaches in Natural Language Processing (NLP). It is a distilled version of BERT which is a fast, cheap, and light Transformer model trained by distilling BERT base. It has 40% fewer parameters than Bert and runs 60% faster and preserves over 97% of BERT’s performances as measured on the GLUE language understanding benchmark.
We used a batch size of 64 with a maximum length of 192 and trained the model over 3 epochs with a learning rate of 1.0000e-05. The insufficient part is complemented by <PAD> markers, and the excess part is truncated. The optimal learning rate, the number of training rounds, and the size of training batches are determined experimentally, and the appropriate truncation length of the input sentence length is determined according to the length of the text in the dataset.

### 2) XLM - RoBERTa
XLM-RoBERTa is a multilingual version of RoBERTa. It is pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages.RoBERTa is a transformers model pre-trained on a large corpus in a self-supervised fashion.
We used a batch size of 32 with a maximum length of 192 and trained the model over 2 epochs with a learning rate of 1.0000e-05. We used the TPU server on the google cloud platform for training. TensorFlow2 framework was used for model construction.

### 3) Ensembling
We have made use of two models DistilmBERT and XLM-RoBERTa and used their ensemble to compile the final results. The weighted ensembling has been done to obtain results. Here each model was weighted proportionally to its capability or skill. XLM-RoBERTa was weighted more as compared to DistilmBERT because of its superior accuracy score and bigger pre-trained data. Weights were chosen according to the best possible results.

## Architecture

![Architecture drawio (1) (1)](https://user-images.githubusercontent.com/97459403/177744759-713eb92e-6d28-477c-a529-4f29cfd164f3.png)

## Results

To verify the effectiveness of pre-training on different models this paper sets up the comparison experiments using accuracy as an evaluation metric to compare different models. Using  DistilmBERT we were able to achieve an accuracy score of 0.8807. On the other hand, XLM-RoBERTa was able to attain an accuracy score of 0.9289. Ensemble modeling brought up the accuracy score to 0.9322 using weights as 0.97 for the XLM-RoERTa model and 0.03 for the DistilmBERT model.
