"""Extension 4: descriptive bootstrap and fold-local predictive evaluation.
Run: python feature_stability_2026-11-25.py --zip PATH --output NEW_DIRECTORY
"""
import argparse,csv,io,json,zipfile,hashlib,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

TOP_COUNTS=(10,30,50)
def correlation(X,y):
    xc=X-X.mean(0); yc=y-y.mean()
    denom=np.linalg.norm(xc,axis=0)*np.linalg.norm(yc)
    return np.abs(np.divide(xc.T@yc,denom,out=np.zeros(X.shape[1]),where=denom>0))
def rank(a): return np.argsort(-a,kind='stable')
def bootstrap(X,y,B,seed):
    rng=np.random.default_rng(seed); counts=np.zeros((3,X.shape[1]),dtype=int)
    for _ in range(B):
        rows=rng.integers(0,len(y),len(y)); order=rank(correlation(X[rows],y[rows]))
        for j,k in enumerate(TOP_COUNTS): counts[j,order[:k]]+=1
    return counts/B

def cut(freq):
    order=rank(freq); gaps=freq[order[:-1]]-freq[order[1:]]
    n=int(np.argmax(gaps))+1 if np.max(gaps)>0 else len(freq)
    return order[:n],float(np.max(gaps))
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(a.zip) as z: rows=list(csv.DictReader(io.StringIO(z.read('train.csv').decode())))
    X=np.array([[float(r[str(j)]) for j in range(150)] for r in rows]);y=np.array([int(r['target']) for r in rows])
    assert X.shape==(250,150) and np.isfinite(X).all() and set(y)=={0,1}
    datahash=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
    with threadpool_limits(limits=1):
        freq=bootstrap(X,y,1000,20261125); desc=[]; fr=[]
        for j,k in enumerate(TOP_COUNTS):
            selected,gap=cut(freq[j]);desc.append({'top_count':k,'selected_count':len(selected),'max_drop':gap,'above_frequency':float(freq[j,selected[-1]]),'below_frequency':float(freq[j,rank(freq[j])[len(selected)]]) if len(selected)<150 else None,'features':selected.tolist()})
            fr.extend({'top_count':k,'feature':i,'frequency':float(freq[j,i]),'selected':int(i in selected)} for i in range(150))
        write(a.output/'frequencies_2026-11-25.csv',fr)
        (a.output/'descriptive_2026-11-25.json').write_text(json.dumps(desc,indent=2));print('Descriptive',desc,flush=True)
        scores=[];audit=[];warn=[]
        cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=420304)
        for i,(tr,va) in enumerate(cv.split(X,y)):
            f=bootstrap(X[tr],y[tr],1000,20261125+i+1)
            sets={'all150':np.arange(150),'pearson30':rank(correlation(X[tr],y[tr]))[:30]}
            for j,k in enumerate(TOP_COUNTS):
                chosen,gap=cut(f[j]);sets[f'stable_top{k}']=chosen
                audit.append({'repeat':i//5+1,'fold':i%5+1,'top_count':k,'n_selected':len(chosen),'max_drop':gap,'features':';'.join(map(str,chosen)),'train_ids':';'.join(rows[t]['id'] for t in tr),'validation_ids':';'.join(rows[t]['id'] for t in va)})
            for family in ['LR','QDA']:
                for variant,features in sets.items():
                    name=family+'_'+variant
                    if family=='QDA' and variant=='all150': continue # documented rank failure, not a scored model
                    estimator=LogisticRegression(C=1,solver='lbfgs',max_iter=10000,random_state=42) if family=='LR' else QuadraticDiscriminantAnalysis(reg_param=1e-6,tol=1e-10)
                    model=make_pipeline(StandardScaler(),estimator)
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always');pred=model.fit(X[tr][:,features],y[tr]).decision_function(X[va][:,features])
                    assert np.isfinite(pred).all()
                    scores.append({'repeat':i//5+1,'fold':i%5+1,'model':name,'n_features':len(features),'auc':roc_auc_score(y[va],pred)})
                    warn.extend({'repeat':i//5+1,'fold':i%5+1,'model':name,'warning':str(w.message)} for w in ws)
            print('Completed fold',i+1,'/25',flush=True)
    write(a.output/'fold_scores_2026-11-25.csv',scores);write(a.output/'selection_audit_private_2026-11-25.csv',audit)
    (a.output/'warnings_2026-11-25.json').write_text(json.dumps(warn,indent=2))
    repeats=[]; summary=[]
    for name in dict.fromkeys(r['model'] for r in scores):
        vals=[]
        for repeat in range(1,6):
            v=np.mean([r['auc'] for r in scores if r['repeat']==repeat and r['model']==name]);vals.append(float(v));repeats.append({'model':name,'repeat':repeat,'mean_fold_auc':float(v)})
        features=[r['n_features'] for r in scores if r['model']==name]
        summary.append({'model':name,'mean_auc':float(np.mean(vals)),'repeat_sd':float(np.std(vals,ddof=1)),'repeat_min':min(vals),'repeat_max':max(vals),'feature_min':min(features),'feature_max':max(features)})
    write(a.output/'repeat_scores_2026-11-25.csv',repeats);write(a.output/'summary_2026-11-25.csv',summary)
    (a.output/'verification_2026-11-25.json').write_text(json.dumps({'data_sha256':datahash,'class_counts':np.bincount(y).tolist(),'n_folds':25,'rows_scores':len(scores),'warning_count':len(warn),'protocol':'fixed before scores; all selectors fitted on train rows only'},indent=2))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
