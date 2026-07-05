"""Project 3a: Next-Generation Reservoir Computing on the VIX suite.
Models: HAR, ESN (Project 2 champion), NG-RC (delay embedding + quadratic
features, deterministic), RP-NGRC (random tanh projection of delay vector).
All walk-forward, change-space targets, OOS 1998-2026, horizons 1/5/10/21."""
import numpy as np, pandas as pd
from esn_lib import ESN

vix = pd.read_csv("vix.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
lv = np.log(vix)
f = pd.DataFrame(index=lv.index)
f["d1"]=lv.diff(); f["d5"]=lv.diff(5)/5
f["dev5"]=lv-lv.rolling(5).mean(); f["dev21"]=lv-lv.rolling(21).mean()
f["dev63"]=lv-lv.rolling(63).mean(); f["slope"]=lv.rolling(5).mean()-lv.rolling(63).mean()
f["lvl"]=lv
f=f.dropna(); FEATS=list(f.columns)
START=f.index.searchsorted(pd.Timestamp("1998-01-01"))
RETRAIN,WASH=252,100
mu,sd=f.iloc[:START].mean(),f.iloc[:START].std()
U=np.clip(((f-mu)/sd).values,-3,3)
n=len(f); lvv=lv.reindex(f.index).values
har=np.column_stack([lvv,pd.Series(lvv).rolling(5).mean().bfill(),
                     pd.Series(lvv).rolling(21).mean().bfill()])

# ---------- feature builders (all causal) ------------------------------------
DELAYS=(0,1,5,21)
def delay_embed(U, delays):
    cols=[]
    for d in delays:
        V=np.roll(U,d,axis=0); V[:d]=U[0]
        cols.append(V)
    return np.hstack(cols)

D = delay_embed(U, DELAYS)                       # n x (7*4=28)

def quad_features(D):
    m=D.shape[1]
    iu=np.triu_indices(m)
    Q=(D[:,iu[0]]*D[:,iu[1]])                    # 406 quadratic terms
    return np.hstack([np.ones((len(D),1)), D, Q])

Z_ngrc = quad_features(D)                        # deterministic NG-RC features
print("NG-RC feature dim:", Z_ngrc.shape[1])

rng=np.random.default_rng(7)
def rp_features(D, dim=400, seeds=8):
    outs=[]
    for s in range(seeds):
        r=np.random.default_rng(50+s)
        R=r.normal(0,1.0/np.sqrt(D.shape[1]),(D.shape[1],dim))
        b=r.uniform(-0.5,0.5,dim)
        outs.append(np.tanh(D@R*1.5+b))
    return [np.hstack([np.ones((len(D),1)),X,D]) for X in outs]

Z_rp = rp_features(D)                            # 8 random projections
print("RP-NGRC feature dim:", Z_rp[0].shape[1], "x", len(Z_rp), "projections")

# ESN states (Project 2 setup)
print("running ESN reservoirs...")
states=[]
for k in range(8):
    esn=ESN(len(FEATS),n_res=400,spectral_radius=0.9,leak=0.3,input_scale=0.4,seed=900+k)
    states.append(np.hstack([np.ones((n,1)),esn._run(U),U]))

rf=lambda Z,y,l: np.linalg.solve(Z.T@Z+l*np.eye(Z.shape[1]),Z.T@y)
of=lambda Z,y: np.linalg.lstsq(Z,y,rcond=None)[0]

def pick_lambda(Zs, ychg, h, grid=(1.,10.,100.,1000.,1e4)):
    inner=int(START*0.75); best,bl=np.inf,None
    for lam in grid:
        m=np.mean([np.mean((Z[inner:START-h]@rf(Z[WASH:inner-h],ychg[WASH:inner-h],lam)
             -ychg[inner:START-h])**2) for Z in Zs[:3]])
        if m<best: best,bl=m,lam
    return bl

summary=[]; store={}
for h in (1,5,10,21):
    ylev=lv.reindex(f.index).shift(-h).values; ychg=ylev-lvv
    P={m:np.full(n,np.nan) for m in ("RW","HAR","ESN","NGRC","RPNG")}
    lam_esn = pick_lambda(states, ychg, h)
    lam_ng  = pick_lambda([Z_ngrc], ychg, h)
    lam_rp  = pick_lambda(Z_rp, ychg, h)
    i=START
    while i<n:
        j=min(i+RETRAIN,n); fe=i-h
        P["RW"][i:j]=lvv[i:j]
        Zh=np.column_stack([np.ones(fe-WASH),har[WASH:fe]])
        P["HAR"][i:j]=np.column_stack([np.ones(j-i),har[i:j]])@of(Zh,ylev[WASH:fe])
        acc=np.zeros(j-i)
        for Z in states: acc+=Z[i:j]@rf(Z[WASH:fe],ychg[WASH:fe],lam_esn)
        P["ESN"][i:j]=lvv[i:j]+acc/8
        w=rf(Z_ngrc[WASH:fe],ychg[WASH:fe],lam_ng)
        P["NGRC"][i:j]=lvv[i:j]+Z_ngrc[i:j]@w
        acc=np.zeros(j-i)
        for Z in Z_rp: acc+=Z[i:j]@rf(Z[WASH:fe],ychg[WASH:fe],lam_rp)
        P["RPNG"][i:j]=lvv[i:j]+acc/8
        i=j
    oos=slice(START,n-h); yt=ylev[oos]
    mrw=np.mean((P["RW"][oos]-yt)**2)
    row={"h":h,"lam(ESN/NG/RP)":f"{lam_esn:g}/{lam_ng:g}/{lam_rp:g}"}
    for m in ("HAR","ESN","NGRC","RPNG"):
        row[m]=round(1-np.mean((P[m][oos]-yt)**2)/mrw,4)
    summary.append(row); store[h]=(P,ylev)
    print(f"h={h} done: {row}")

print("\n=== OOS R^2 vs random walk ===")
print(pd.DataFrame(summary).to_string(index=False))
np.savez("p3_preds.npz", **{f"h{h}_{m}":store[h][0][m] for h in store for m in store[h][0]},
         **{f"h{h}_y":store[h][1] for h in store}, lv=lvv,
         idx=f.index.values.astype("datetime64[D]").astype(str))
