import glob, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "Finished_Almost_important_plots"
files = glob.glob("_isolated_eval_chunks*/*/chunk_result.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
g = df.groupby(["variant","receiver","num_users","ebno_db"], as_index=False).agg(
    block_errors=("block_errors","sum"), num_blocks=("num_blocks","sum"))
g["bler"] = g.block_errors / g.num_blocks
g = g[g.block_errors > 0].sort_values("ebno_db")
g.to_csv(OUT + "/points_used.csv", index=False)

def pts(variant, rx, u):
    s = g[(g.variant==variant)&(g.receiver==rx)&(g.num_users==u)]
    return s.ebno_db.values, s.bler.values

def newfig():
    fig, ax = plt.subplots(figsize=(5.0,3.9))
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=.3)
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("BLER")
    return fig, ax

def save(fig, ax, name):
    ax.legend(fontsize=9, framealpha=.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=300); fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig); print("wrote", name)

# Receiver comparison at each user count (one file per U)
RX = [("upair5g_lmmse",            "UPAIR (proposed)",  "#0072B2","o","-" ,2.0),
      ("baseline_ls_2dlmmse_lmmse","2D-LMMSE + LMMSE",  "#E69F00","s","-" ,1.7),
      ("baseline_ls_lmmse",        "LS + LMMSE",        "#D55E00","v","-" ,1.7),
      ("perfect_csi_lmmse",        "Perfect CSI",       "black",  "d","--",1.7)]
for u in [2,3,4]:
    fig, ax = newfig()
    for rx,lab,c,m,ls,lw in RX:
        x,y = pts("main_d256_b4_r2", rx, u)
        if len(x): ax.plot(x,y,ls,marker=m,color=c,label=lab,lw=lw,ms=5.5)
    save(fig, ax, f"receivers_U{u}")

# Ablation comparison at U = 3
ARMS = [("main_d256_b4_r2",                    "M0: full model",                    "black",  "o","-" ,2.4),
        ("width_light_d192_b4_r2",             r"A1: width $d{=}192$",              "#0072B2","s","--",1.6),
        ("shallow_light_d256_b2_r2",           r"A2: depth $B{=}2$",                "#56B4E9","v","--",1.6),
        ("local_only_no_axial_attn_d256_b4_r2","A4: no axial attention",            "#D55E00","D","-" ,1.9),
        ("freq_attn_only_d256_b4_r2",          "A5: frequency attention only",      "#CC79A7","P","--",1.6),
        ("errvar_eval_swap_d256_b4_r2",        r"A8$^\dagger$: LS err.\ var.\ at eval","#009E73","p",":" ,1.8)]
fig, ax = newfig()
for v,lab,c,m,ls,lw in ARMS:
    x,y = pts(v, "upair5g_lmmse", 3)
    if len(x): ax.plot(x,y,ls,marker=m,color=c,label=lab,lw=lw,ms=5.5)
save(fig, ax, "ablations_U3")
