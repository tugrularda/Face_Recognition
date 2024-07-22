(ง◉_◉)ง

Initializing MLFlow:

  pip install mlflow

  The following command is to be run every time the app is opened: 
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 

  Include the following script to set uri and set experiment:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Face Recognition")

  Log any metric by:
    mlflow.log_metric("f1_score", f1)

  In order to track metrics, run the commanc:
    mlflow ui

  Log model by:
    mlflow.pytorch.log_model(model, "model")

  Fetching models is explained in ui
