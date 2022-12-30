import io


def reverse_word_index(word_index):
    return dict([(value, key) for (key, value) in word_index.items()])


def write_embeddings(weights, vocab_size, reverse_word_index_dict_fn):
    out_v = io.open("vecs.tsv", "w", encoding="utf-8")
    out_m = io.open("meta.tsv", "w", encoding="utf-8")
    for word_num in range(1, vocab_size):
        word = reverse_word_index_dict_fn(word_num)
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write("\t".join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()
