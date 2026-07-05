"""Stress battery A (optimized): incremental normal equations, h=21."""
import numpy as np, pandas as pd, sys, json

vix = pd.read_csv("vix.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
lv = np.log(vix)
f = pd.DataFrame(index=lv.index)
f["d1"]=lv.diff(); f["d5"]=lv.diff(5)/5
f["dev5"]=lv-lv.rolling(5).mean(); f["dev21"]=lv-lv.rolling(21).mean()
f["dev63"]=lv-lv.rolling(63).mean(); f["slope"]=lv.rolling(5).mean()-lv.rolling(63).mean()
f["lvl"]=lv
f=f.dropna()
START=f.index.searchsorted(pd.Timestamp("1998-01-01")); RETRAIN,WASH=252,100
mu,sd=f.iloc[:START].mean(),f.iloc[:START].std()
U=np.clip(((f-mu)/sd).values,-3,3); n=len(f); lvv=lv.reindex(f.index).values
har=np.column_stack([lvv,pd.Series(lvv).rolling(5).mean().bfill(),
                     pd.Series(lvv).rolling(21).mean().bfill()])
of=lambda Z,y: np.linalg.lstsq(Z,y,rcond=None)[0]
H=21
ylev=lv.reindex(f.index).shift(-H).values; ychg=ylev-lvv

def run(dim=400, delays=(0,1,5,21), scale=1.5, seed0=50, blendwin=252, lams=(1e3,1e4,1e5)):
    D=np.hstack([np.vstack([np.tile(U[0],(d,1)),U[:n-d]]) for d in delays])
    Zs=[]
    for s in range(8):
        r=np.random.default_rng(seed0+s)
        R=r.normal(0,1/np.sqrt(D.shape[1]),(D.shape[1],dim)); b=r.uniform(-.5,.5,dim)
        Zs.append(np.ascontiguousarray(np.hstack([np.ones((n,1)),np.tanh(D@R*scale+b),D])))
    m=Zs[0].shape[1]
    G=[np.zeros((m,m)) for _ in range(8)]; c=[np.zeros(m) for _ in range(8)]
    prev=WASH
    P=np.full(n,np.nan); Phar=np.full(n,np.nan)
    i=START
    while i<n:
        j=min(i+RETRAIN,n); fe=i-H
        Zh=np.column_stack([np.ones(fe-WASH),har[WASH:fe]])
        Phar[i:j]=np.column_stack([np.ones(j-i),har[i:j]])@of(Zh,ylev[WASH:fe])
        acc=np.zeros(j-i)
        for k,Z in enumerate(Zs):
            B=Z[prev:fe]; yb=ychg[prev:fe]
            G[k]+=B.T@B; c[k]+=B.T@yb
            for lam in lams:
                acc+=Z[i:j]@np.linalg.solve(G[k]+lam*np.eye(m),c[k])
        prev=fe
        P[i:j]=lvv[i:j]+acc/(8*len(lams)); i=j
    e_rc=pd.Series((P-ylev)**2).shift(H); e_ha=pd.Series((Phar-ylev)**2).shift(H)
    ir=1/e_rc.rolling(blendwin,min_periods=63).mean(); ih=1/e_ha.rolling(blendwin,min_periods=63).mean()
    w=(ir/(ir+ih)).fillna(.5).values
    C=w*P+(1-w)*Phar
    oos=slice(START,n-H); yt=ylev[oos]
    e_rw=np.nanmean((lvv[oos]-yt)**2)
    return (1-np.nanmean((C[oos]-yt)**2)/e_rw, 1-np.nanmean((P[oos]-yt)**2)/e_rw)

if __name__=="__main__":
    which=sys.argv[1]
    batches={
     "0":[("BASELINE",{})],
     "1":[("dim=200",dict(dim=200)),("dim=800",dict(dim=800)),
          ("delays=(0,1,5)",dict(delays=(0,1,5))),("delays=(0,1,5,21,63)",dict(delays=(0,1,5,21,63)))],
     "2":[("tanh scale=1.0",dict(scale=1.0)),("tanh scale=2.0",dict(scale=2.0)),
          ("fresh seeds 1000+",dict(seed0=1000)),("fresh seeds 7777+",dict(seed0=7777))],
     "3":[("blend win=126",dict(blendwin=126)),("blend win=504",dict(blendwin=504)),
          ("lambda grid /10",dict(lams=(1e2,1e3,1e4))),("lambda grid x10",dict(lams=(1e4,1e5,1e6)))]}
    out={}
    for name,kw in batches[which]:
        cc,rr=run(**kw); out[name]=(cc,rr); print(f"{name:26s} COMBO {cc:.4f}  RC {rr:.4f}", flush=True)
    json.dump(out, open(f"stressA_{which}.json","w"))
