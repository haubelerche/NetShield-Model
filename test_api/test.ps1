python - << 'PY'
import numpy as np, json, requests
x = np.load("data/processed/X_test.npy")[0].tolist()
r = requests.post("http://127.0.0.1:8000/predict", json={"values": x})
print(json.dumps(r.json(), indent=2))
PY




