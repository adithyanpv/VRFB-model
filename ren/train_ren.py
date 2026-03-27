"""
REN Training Script — VRFB SOC Estimator (hybrid, physics-constrained)
=======================================================================

Features (7):
  voltage, current, temperature_stack, temperature_tank, flow_rate,
  soc_cc, transport_ratio_approx

Architecture constraints enforced:
  1. use_current_gate=True  — z frozen at I=0, dSOC/dt=0 guaranteed
  2. use_feedthrough=False  — no D(x_t) path, no flow→SOC shortcuts
  3. Zero-current invariance loss — explicitly penalises SOC changes at I=0

OUTPUT:
  ren/scaler.pkl
  ren/ren_soc_best.pth
  ren/training_log.csv
  ren/training_curves.png
"""

import os, time, pickle, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from ren.ren_model import REN

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_COLS = [
    "voltage",                   # OCV at I=0 directly encodes SOC
    "current",                   # I=0 → gate freezes z
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
    "soc_cc",                    # CC estimate — corrected by REN
    "transport_ratio_approx",    # |I|/I_limit — 0 at I=0, safe
]
TARGET_COL = "target"
INPUT_DIM  = len(FEATURE_COLS)   # 7
CURRENT_IDX = 1                  # index of current in FEATURE_COLS

# Model
HIDDEN_DIM      = 128
ALPHA           = 0.5
DROPOUT         = 0.1
N_POWER_ITERS   = 10

# Training
SEQ_LEN          = 256
BATCH_SIZE       = 32
EPOCHS           = 80
LR               = 3e-4
LR_WARMUP_EPOCHS = 5
WEIGHT_DECAY     = 1e-5
PATIENCE         = 25
GRAD_CLIP        = 1.0
WARMUP_STEPS     = 32
P_Z0_RESET       = 0.30

# Loss weights
MSE_WEIGHT       = 0.7
MAE_WEIGHT       = 0.3
BOUNDARY_SCALE   = 3.0
SMOOTH_WEIGHT    = 0.10
SMOOTH_I_REF     = 0.3    # scaled current units
HARD_CASE_WEIGHT = 3.0
HARD_CASE_THRESH = 0.10
# Zero-current invariance loss
ZC_WEIGHT        = 2.0    # strong penalty — this is a physics law, not a hint
ZC_THRESH        = 0.05   # |I_scaled| below which SOC must not change

SOC_MIN = 0.05
SOC_MAX = 0.95

TRAIN_CSV = "datasets/vrfb_train.csv"
TEST_CSV  = "datasets/vrfb_test.csv"
SAVE_DIR  = "ren"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA
# =============================================================================

def load_episodes(df, scaler, fit_scaler=False):
    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)
    if fit_scaler:
        scaler.fit(X_all)
    X_scaled = scaler.transform(X_all).astype(np.float32)
    episodes = []
    for _, group in df.groupby("episode_id", sort=True):
        idx  = group.index
        X_ep = torch.tensor(X_scaled[idx], dtype=torch.float32)
        y_ep = torch.tensor(y_all[idx],    dtype=torch.float32).unsqueeze(-1)
        episodes.append((X_ep, y_ep))
    return episodes


# =============================================================================
# LOSS — includes zero-current invariance
# =============================================================================

def composite_loss(pred, target, X_chunk=None):
    """
    pred, target : (batch, seq, 1)
    X_chunk      : (batch, seq, input_dim) — needed for ZC invariance loss

    Loss terms:
      base      = boundary-weighted MSE + MAE + hard-case MSE
      smooth    = |ΔSOC| × exp(-|I_scaled|/I_ref)  — suppress noise at low I
      zc_inv    = |ΔSOC| when |I_scaled| < ZC_THRESH  — PHYSICS CONSTRAINT
                  This is separate from smooth: it's a hard invariance penalty,
                  not a soft smoothness preference. ZC_WEIGHT >> SMOOTH_WEIGHT.
    """
    err = pred - target
    with torch.no_grad():
        w = (1.0 + BOUNDARY_SCALE * (
                torch.exp(-50.0 * (target - SOC_MIN).clamp(min=0.0)) +
                torch.exp(-50.0 * (SOC_MAX - target).clamp(min=0.0))
             )) * torch.where(err.abs() > HARD_CASE_THRESH,
                              torch.full_like(err, HARD_CASE_WEIGHT),
                              torch.ones_like(err))
    base = MSE_WEIGHT * (w * err**2).mean() + MAE_WEIGHT * F.l1_loss(pred, target)

    if X_chunk is not None and pred.shape[1] > 1:
        I_scaled = X_chunk[:, :, CURRENT_IDX:CURRENT_IDX+1]  # (B, seq, 1)
        delta    = (pred[:, 1:, :] - pred[:, :-1, :]).abs()   # (B, seq-1, 1)
        I_mid    = I_scaled[:, 1:, :].abs()                   # (B, seq-1, 1)

        # Smooth loss — soft suppression at all low currents
        with torch.no_grad():
            smooth_w = torch.exp(-I_mid / SMOOTH_I_REF)
        smooth_loss = (smooth_w * delta).mean()

        # Zero-current invariance — hard penalty at I≈0
        # This enforces the physics law: dSOC/dt = 0 when I = 0
        with torch.no_grad():
            zc_mask = (I_mid < ZC_THRESH).float()
        zc_loss = (zc_mask * delta).mean()

        return base + SMOOTH_WEIGHT * smooth_loss + ZC_WEIGHT * zc_loss

    return base


# =============================================================================
# STATEFUL EPOCH
# =============================================================================

def run_epoch(model, episodes, optimiser=None):
    is_train = optimiser is not None
    model.train() if is_train else model.eval()
    perm = torch.randperm(len(episodes)).tolist()
    total_loss = 0.0; sq_err = []; abs_err = []; n_b = 0
    ctx = torch.enable_grad if is_train else torch.no_grad

    for bs in range(0, len(perm), BATCH_SIZE):
        batch    = [episodes[i] for i in perm[bs:bs+BATCH_SIZE]]
        B        = len(batch)
        n_chunks = min(ep[0].shape[0] for ep in batch) // SEQ_LEN
        if n_chunks == 0: continue

        z = model.z0.expand(B, -1).contiguous().to(DEVICE)
        if is_train: z = z.detach()
        bloss = 0.0

        for ci in range(n_chunks):
            s, e = ci*SEQ_LEN, (ci+1)*SEQ_LEN
            Xc = torch.stack([ep[0][s:e] for ep in batch]).to(DEVICE)
            Xc_raw = Xc.clone()
            yc = torch.stack([ep[1][s:e] for ep in batch]).to(DEVICE)

            use_z0 = is_train and ci > 0 and torch.rand(1).item() < P_Z0_RESET
            with ctx():
                yp, zn = model(Xc, z=None if use_z0 else z, x_raw=Xc_raw)
                # Warmup masking
                if WARMUP_STEPS > 0 and SEQ_LEN > WARMUP_STEPS:
                    yp_l = yp[:, WARMUP_STEPS:, :]
                    yc_l = yc[:, WARMUP_STEPS:, :]
                    Xc_l = Xc[:, WARMUP_STEPS:, :]
                else:
                    yp_l, yc_l, Xc_l = yp, yc, Xc

                loss = composite_loss(yp_l, yc_l, X_chunk=Xc_l)
                bloss += loss.item()
                e_arr = (yp_l - yc_l).detach().cpu().numpy().ravel()
                sq_err.append(e_arr**2); abs_err.append(np.abs(e_arr))

                if is_train:
                    optimiser.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimiser.step()
            z = zn.detach()

        total_loss += bloss / max(n_chunks, 1); n_b += 1

    sq_all = np.concatenate(sq_err); ab_all = np.concatenate(abs_err)
    return total_loss/max(n_b,1), math.sqrt(sq_all.mean()), ab_all.mean(), ab_all.max()


# =============================================================================
# LR SCHEDULE
# =============================================================================

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_r=0.01):
        self.w=warmup; self.t=total; self.m=min_r; super().__init__(opt)
    def get_lr(self):
        e=self.last_epoch
        s=((e+1)/self.w if e<self.w else
           self.m+0.5*(1-self.m)*(1+math.cos(math.pi*(e-self.w)/max(self.t-self.w,1))))
        return [b*s for b in self.base_lrs]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*65}")
    print(f"  REN Training — physics-constrained hybrid (7 features)")
    print(f"{'='*65}")
    print(f"  Device: {DEVICE}  Hidden: {HIDDEN_DIM}  Alpha: {ALPHA}")
    print(f"  use_current_gate=True  (z frozen at I=0 — charge conservation)")
    print(f"  use_feedthrough=False  (no D(x_t) — no flow shortcuts)")
    print(f"  ZC invariance loss weight: {ZC_WEIGHT}  threshold: {ZC_THRESH}")
    print(f"  Features ({INPUT_DIM}): {FEATURE_COLS}")
    print(f"{'-'*65}")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    df_train["target"] = df_train["SOC_true"] - df_train["soc_cc"]
    df_test["target"]  = df_test["SOC_true"]  - df_test["soc_cc"]

    missing = [c for c in FEATURE_COLS + ["SOC_true", "soc_cc", "episode_id"]
           if c not in df_train.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nRun dataset_gen.py first.")

    print(f"\n  Train: {len(df_train):,} rows ({df_train['episode_id'].nunique()} eps)")
    print(f"  Test:  {len(df_test):,} rows ({df_test['episode_id'].nunique()} eps)")
    
   

    scaler    = StandardScaler()
    train_eps = load_episodes(df_train, scaler, fit_scaler=True)
    test_eps  = load_episodes(df_test,  scaler, fit_scaler=False)

    # Print what I=0 looks like after scaling (for gate calibration info)
    I_mean = scaler.mean_[CURRENT_IDX]
    I_std  = scaler.scale_[CURRENT_IDX]
    print(f"\n  Current scaler: mean={I_mean:.2f}A  std={I_std:.2f}A")
    print(f"  I=0A scaled: {-I_mean/I_std:.4f}  (gate will be {math.tanh(4*abs(-I_mean/I_std)):.4f})")
    print(f"  I=50A scaled: {(50-I_mean)/I_std:.4f}  (gate will be ≈{math.tanh(4*(50-I_mean)/I_std):.4f})")

    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    print(f"  Scaler → {scaler_path}")

    cc_err  = np.abs(df_test[TARGET_COL].values - df_test["soc_cc"].values)
    cc_rmse = math.sqrt(np.mean(cc_err**2))
    cc_mae  = cc_err.mean(); cc_max = cc_err.max()
    print(f"\n  CC baseline — RMSE:{cc_rmse:.5f}  MAE:{cc_mae:.5f}  Max:{cc_max:.5f}")

    model = REN(
        input_dim        = INPUT_DIM,
        hidden_dim       = HIDDEN_DIM,
        output_dim       = 1,
        alpha            = ALPHA,
        dropout          = DROPOUT,
        n_power_iters    = N_POWER_ITERS,
        use_feedthrough  = False,
        use_current_gate = True,
        current_feat_idx = CURRENT_IDX,
    ).to(DEVICE)

    print(f"\n  Parameters     : {model.count_parameters():,}")
    print(f"  Contraction    : {model.contraction_rate():.4f} (< {1-ALPHA:.2f})")
    print(f"  Gate at I=0A   : {model.gate_value_at(abs(-I_mean/I_std)):.4f} (target: ~0)")
    print(f"  Gate at I=50A  : {model.gate_value_at(abs((50-I_mean)/I_std)):.4f} (target: ~1)")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = WarmupCosine(opt, LR_WARMUP_EPOCHS, EPOCHS)

    print(f"\n  {'Ep':>4}  {'Trn':>10}  {'Val':>10}  {'EMA':>10}  "
          f"{'RMSE':>9}  {'MAE':>8}  {'gate@0':>7}  {'z0':>6}  {'LR':>9}")
    print(f"  {'-'*90}")

    best_loss=math.inf; ema_loss=math.inf; pat=0
    log=[]; t_l=[]; v_l=[]; v_r=[]; lrs=[]
    t0=time.time()

    for ep in range(1, EPOCHS+1):
        t_ep = time.time()
        tl,tr,tm,_ = run_epoch(model, train_eps, opt)
        vl,vr,vm,vx = run_epoch(model, test_eps)
        sched.step()

        ema_loss = vl if ep==1 else 0.7*ema_loss+0.3*vl
        cr = model.contraction_rate()
        g0 = model.gate_value_at(abs(-I_mean/I_std))
        z0n = model.z0_norm()
        lr  = sched.get_last_lr()[0]
        t_l.append(tl); v_l.append(vl); v_r.append(vr); lrs.append(lr)
        log.append(dict(epoch=ep,trn_loss=tl,val_loss=vl,ema_loss=ema_loss,
                        val_rmse=vr,val_mae=vm,gate_at_zero=g0,z0_norm=z0n,lr=lr))

        marker=""
        if ema_loss < best_loss:
            best_loss=ema_loss; pat=0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,"ren_soc_best.pth"))
            marker=" ◀"
        else:
            pat+=1

        print(f"  {ep:>4d}  {tl:>10.6f}  {vl:>10.6f}  {ema_loss:>10.6f}  "
              f"{vr:>9.5f}  {vm:>8.5f}  {g0:>7.4f}  {z0n:>6.3f}  {lr:>9.2e}"
              f"  [{time.time()-t_ep:.0f}s]{marker}")

        if pat >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}"); break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR,"ren_soc_last.pth"))
    pd.DataFrame(log).to_csv(os.path.join(SAVE_DIR,"training_log.csv"),index=False)

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR,"ren_soc_best.pth"),
                                     map_location=DEVICE))
    _,ren_rmse,ren_mae,ren_max = run_epoch(model, test_eps)

    print(f"\n{'='*65}")
    print(f"  FINAL — Physics-constrained REN vs Coulomb Counter")
    print(f"{'='*65}")
    for rv,cv,lbl in [(ren_rmse,cc_rmse,"RMSE"),(ren_mae,cc_mae,"MAE"),(ren_max,cc_max,"Max Error")]:
        print(f"  {lbl:<16} REN={rv:.5f}  CC={cv:.5f}  {(1-rv/cv)*100:+.1f}%")
    print(f"\n  Training time  : {(time.time()-t0)/60:.1f} min")
    print(f"  Best EMA loss  : {best_loss:.6f}")
    print(f"  Final z0 norm  : {model.z0_norm():.4f}")
    print(f"  Gate @ I=0     : {model.gate_value_at(abs(-I_mean/I_std)):.4f}  (should be <<0.1)")

    # Training curves
    ep_x=list(range(1,len(t_l)+1))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(ep_x,t_l,label="Train",color="steelblue",lw=1.5)
    axes[0].plot(ep_x,v_l,label="Val",  color="tomato",   lw=1.5)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.4)
    axes[1].plot(ep_x,v_r,color="purple",lw=2,label="REN RMSE")
    axes[1].axhline(cc_rmse,color="orange",ls="--",lw=1.5,label=f"CC={cc_rmse:.4f}")
    axes[1].set_title("Val RMSE vs CC"); axes[1].legend(); axes[1].grid(alpha=0.4)
    axes[2].plot(ep_x,[row["gate_at_zero"] for row in log],
                 color="red",lw=1.5,label="gate @ I=0")
    axes[2].axhline(0,color="black",ls="--",lw=0.8)
    axes[2].set_title("Gate value at I=0  (target: 0)"); axes[2].legend(); axes[2].grid(alpha=0.4)
    plt.suptitle("REN hybrid — physics-constrained",fontsize=12); plt.tight_layout()
    fp=os.path.join(SAVE_DIR,"training_curves.png")
    plt.savefig(fp,dpi=150,bbox_inches="tight"); plt.close()
    print(f"\n  Saved → {fp}\n  Saved → {SAVE_DIR}/ren_soc_best.pth")

if __name__=="__main__":
    main()