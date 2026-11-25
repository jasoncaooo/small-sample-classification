"""Follow-up exploration; no public leaderboard optimization or external data."""
import csv,io,json,zipfile,importlib.util,warnings
from pathlib import Path
from scipy.stats import rankdata
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
warnings.simplefilter('error',ConvergenceWarning)
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous',p/'improve_2026-11-21.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
out=p/'qda_precision_2026-11-23'; n=1
while out.exists():
 n+=1; out=p/f'qda_precision_2026-11-23_v{n}'
out.mkdir()
def dump(name,rows): old.write_csv(out/(name+'_2026-11-23.csv'),rows)
lr=lambda c:LogisticRegression(C=c,l1_ratio=1,solver='liblinear',random_state=42,max_iter=10000)
models={'previous_RBF8':make_pipeline(old.PearsonTopK(8),StandardScaler(),SVC(C=10,gamma=.1))}
for k in [10,15,20,30,40]:
 for reg in [1e-6,1e-4,.001,.01]:
  models[f'QDA_k{k}_reg{reg}']=make_pipeline(old.PearsonTopK(k),StandardScaler(),QuadraticDiscriminantAnalysis(reg_param=reg, tol=1e-10))
(out/'plan_2026-11-23.json').write_text(json.dumps({'models':list(models),'selection':'5 folds x 5 repeats, seed 20261123','nested':'5 folds x 2 repeats, seed 314159; inner 5 folds x 5 repeats, seed 20261123','caveat':'Adaptive follow-up to previous experiments; nested CV not a new untouched test set.'},indent=2))
with zipfile.ZipFile(p/'small-sample-classification-challenge.zip') as z:
 data={k:list(csv.DictReader(io.StringIO(z.read(k+'.csv').decode()))) for k in ['train','test','sample_submission']}
X=np.array([[float(r[str(j)]) for j in range(150)] for r in data['train']]); y=np.array([int(r['target']) for r in data['train']]); T=np.array([[float(r[str(j)]) for j in range(150)] for r in data['test']])
def score(m,x):
 return m.decision_function(x)
def search(x,y,verbose=False):
 cv=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=20261123).split(x,y)); results=[]
 for name,m in models.items():
  a=[roc_auc_score(y[v],score(clone(m).fit(x[t],y[t]),x[v])) for t,v in cv]
  results.append({'model':name,'mean_auc':float(np.mean(a)),'sd':float(np.std(a,ddof=1))})
  if verbose: print(results[-1],flush=True)
 return max(results,key=lambda r:r['mean_auc'])['model'],results
with threadpool_limits(1):
 best,results=search(X,y,True); dump('candidate_comparison',results)
 outer=[]; predictions=[]
 for i,(t,v) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=314159).split(X,y)):
  name,_=search(X[t],y[t]); a=score(clone(models[name]).fit(X[t],y[t]),X[v]); b=score(clone(models['previous_RBF8']).fit(X[t],y[t]),X[v])
  outer.append({'repeat':i//5+1,'fold':i%5+1,'chosen':name,'new_auc':roc_auc_score(y[v],a),'previous_fixed_auc':roc_auc_score(y[v],b)})
  predictions.extend({'repeat':i//5+1,'fold':i%5+1,'id':data['train'][j]['id'],'target':int(y[j]),'new_score':float(aa),'previous_score':float(bb)} for j,aa,bb in zip(v,a,b))
  print('outer',outer[-1],flush=True)
 dump('nested_scores',outer); dump('nested_oof',predictions)
 final=clone(models[best]).fit(X,y)
 raw=score(final,T)
 # Monotonic sigmoid preserves SVM ranking; these are not calibrated probabilities.
 assert np.isfinite(raw).all()
 pred=rankdata(raw,method="average")/(len(raw)+1)
 lookup=dict(zip([r['id'] for r in data['test']],pred))
 rows=[{'id':r['id'],'target':float(lookup[r['id']])} for r in data['sample_submission']]
 assert len(rows)==19750 and all(np.isfinite(r['target']) and 0<=r['target']<=1 for r in rows)
 dump('submission_qda_precision',rows)
 summary={'best':best,'selection_auc':max(r['mean_auc'] for r in results),'nested_auc':float(np.mean([r['new_auc'] for r in outer])),'previous_fixed_nested_auc':float(np.mean([r['previous_fixed_auc'] for r in outer])),'public_auc':None,'submitted':False}
 (out/'metrics_2026-11-23.json').write_text(json.dumps(summary,indent=2)); print(summary,flush=True)
 dump('confirmed_public_scores',[{'file':'submission_improved_2026-11-21.csv','public_auc':.88729,'evidence':'User screenshot 2026-11-23 18:53:24','private_auc':''},{'file':'submission.csv','public_auc':.88047,'evidence':'Same screenshot; Ron Li; model unknown','private_auc':''}])
 print(out,flush=True)
