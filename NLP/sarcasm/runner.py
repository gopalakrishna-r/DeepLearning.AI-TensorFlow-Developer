import json

import numpy as np
from wandb.keras import WandbCallback

import wandb
from data_loader import load_sarcasm
from model import build_embedding_model
from preprocessor import process_sequence

sarcasm_data = load_sarcasm()

with open(sarcasm_data, "r") as f:
    datastore = json.load(f)

embedding_size = 16
vocab_size = 10000
max_length = 32
trunc_type = "post"
padding_type = "post"
oov_token = "<OOV>"
training_size = 20000

sentences, labels = zip(
    *[(item["headline"], item["is_sarcastic"]) for item in datastore]
)

training_sentences, training_labels, testing_sentences, testing_labels = (
    sentences[:training_size],
    labels[:training_size],
    sentences[training_size:],
    labels[training_size:],
)

padded, padded_testing = process_sequence(
    training_sentences,
    testing_sentences,
    vocab_size,
    max_length,
    trunc_type,
    padding_type,
    oov_token,
)

with wandb.init(
    project="embedded-hyperparameter-sweeps", name="embeddings-rnn", entity="goofygc316"
) as run:
    config = run.config
    model = build_embedding_model(config)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())

    num_epochs = 30
    history = model.fit(
        padded,
        np.array(training_labels),
        validation_data=(padded_testing, np.array(testing_labels)),
        epochs=num_epochs,
        workers=8,
        verbose=2,
        callbacks=[WandbCallback()],
    )
