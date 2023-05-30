import ktrain

predictor = ktrain.load_predictor("./models/bert_")
message = "Though I do not know how to deal with it, I have to still work on it"
prediction = predictor.predict(message)

print("predicted: {} ({:.2f})".format(prediction))
