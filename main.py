from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime

if __name__ == "__main__":

    # Read the dataset

    model_name = 'camembert-base'
    batch_size = 16
    nli_reader = NLIDataReader('../processed')
    train_num_labels = nli_reader.get_num_labels()
    model_save_path = './training_nli_' + model_name + '-' + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")
    num_epochs= 10


    # "camembert-base" is the name of Camembert Model from Hugging Face
    word_embedding_model = models.CamemBERT("camembert-base")
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_data = SentencesDataset(nli_reader.get_examples('train'),
                                  model=model)
    train_dataloader = DataLoader(train_data, shuffle=True,
                                  batch_size=batch_size)
    train_loss = losses.SoftmaxLoss(model=model,
                                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=train_num_labels)

    dev_data = SentencesDataset(nli_reader.get_examples('valid'),
                                  model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=True,
                                  batch_size=batch_size)

    evaluator = LabelAccuracyEvaluator(train_dataloader, softmax_model=model)

    warmup_steps = 4

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )

    model.save("model.pt")

    # traubèdata = DataLoader("")

    # TODO : Train on XNLI French dataset

    # TODO : Déplacer Dossier git à la place de ce dossier Sentence_BERT

    # TODO : https://github.com/UKPLab/sentence-transformers  Finetuner CamemBERT sur ce dataset