import tensorflow.keras as keras


def load_sarcasm():
    json = keras.utils.get_file('sarcasm.json',
                                'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json',
                                cache_dir='..\\data\\')
    return json
