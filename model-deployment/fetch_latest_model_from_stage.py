import mlflow.pyfunc

model_name = "sk-learn-random-forest-reg-model"
stage = "Staging"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

model.predict(data)