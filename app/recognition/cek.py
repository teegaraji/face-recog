import tensorflow as tf

model = tf.saved_model.load("saved_models/embedding_model_savedmodel")
print(list(model.signatures.keys()))
