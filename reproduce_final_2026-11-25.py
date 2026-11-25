"""Reproduce the selected final model without rerunning model selection."""
from pathlib import Path
import csv, io, zipfile, importlib.util, hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from threadpoolctl import threadpool_limits
p = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('helper', p/'improve_2026-11-21.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
with zipfile.ZipFile(p/'small-sample-classification-challenge.zip') as z:
    data = {k: list(csv.DictReader(io.StringIO(z.read(k+'.csv').decode()))) for k in ['train','test','sample_submission']}
X = np.array([[float(r[str(j)]) for j in range(150)] for r in data['train']])
y = np.array([int(r['target']) for r in data['train']])
T = np.array([[float(r[str(j)]) for j in range(150)] for r in data['test']])
model = make_pipeline(helper.PearsonTopK(30), StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=1e-6, tol=1e-10))
with threadpool_limits(1):
    raw = model.fit(X,y).decision_function(T)
assert np.isfinite(raw).all()
scores = rankdata(raw,method='average')/(len(raw)+1)
lookup = dict(zip([r['id'] for r in data['test']], scores))
rows = [{'id':r['id'],'target':float(lookup[r['id']])} for r in data['sample_submission']]
output = p/'reproduced_submission_2026-11-23.csv'
if output.exists(): raise FileExistsError('Preserve existing output: '+str(output))
helper.write_csv(output,rows)
expected_sha256 = '9ed1cffad27cdefd837685738db87affbf9694dde4d0b59dde5cef4d20568703'
identical = hashlib.sha256(output.read_bytes()).hexdigest() == expected_sha256
print('Exact match to submitted CSV:', identical)
print('SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())
if not identical: raise RuntimeError('Output differs from archived submission; inspect environment before using.')
print(output)
