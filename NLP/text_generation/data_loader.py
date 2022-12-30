import keras as keras
import numpy as np
import tensorflow as tf


def load_corpus_data():
    data = (
        "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made "
        "him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and "
        "relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes "
        "glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, "
        "\nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere "
        "dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink "
        "for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. "
        "\nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, "
        "\nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round "
        "as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat "
        "Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical "
        "polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them "
        "the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the "
        "ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. "
        "\nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I "
        "spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, "
        "\nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps "
        "for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in "
        "couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss "
        "Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered "
        "them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the "
        "midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the "
        "lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, "
        "so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from "
        "under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were "
        "runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked "
        "up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, "
        "bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to "
        "Lanigans Ball. "
    )
    return data


def load_irish_songs():
    keras.utils.get_file(
        fname="irish-lyrics-eof.txt",
        origin="https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt",
        cache_dir="..\\data\\",
    )
    txt = open(
        r"D:\codebase\DeepLearning.AI-TensorFlow-Developer\course_3\irish-lyrics-eof.txt"
    ).read()
    return txt


def load_shakespeare_text():
    shakespeare_url = "https://homl.info/shakespeare"
    filepath = keras.utils.get_file("shakespeare", shakespeare_url)
    with open(filepath) as f:
        shakespeare_txt = f.read()
    return shakespeare_txt


def generate_shakespeare_dataset(
    encoded_text, dataset_size, max_id, n_steps=100, batch_size=32
):
    train_size = dataset_size * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded_text[:train_size])
    window_length = n_steps + 1
    dataset = (
        dataset.window(window_length, shift=1, drop_remainder=True)
        .flat_map(lambda window: window.batch(window_length))
        .shuffle(10000)
        .batch(batch_size)
        .map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        .map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        .prefetch(1)
    )
    return dataset


def generate_shakespeare_dataset_stateful(
    encoded_text, dataset_size, max_id, n_steps=100, batch_size=32
):
    train_size = dataset_size * 90 // 100
    window_length = n_steps + 1
    encoded_parts = np.array_split(encoded_text[:train_size], batch_size)
    datasets = []
    for encoded_part in encoded_parts:
        dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
        dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_length))
        datasets.append(dataset)
    dataset = tf.data.Dataset.zip(tuple(datasets))
    dataset = dataset.map(lambda *windows: tf.stack(windows))
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    dataset = dataset.map(
        lambda x_batch, y_batch: (tf.one_hot(x_batch, depth=max_id), y_batch)
    )
    return dataset.prefetch(1)
