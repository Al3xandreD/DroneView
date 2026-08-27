For MLFlow:
export MLFLOW_TRACKING_URI=file:./outputs.nosync/mlf_runs
python3 -c "
from mlflow.server import app
app.run(host='127.0.0.1', port=5001)
"