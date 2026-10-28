"""Standalone scientific figures from completed, audited scalar measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path("experiments/stateloop-20261028-local48")
COLORS={"vanilla":"#64748b","reset":"#d97706","persistent":"#0891b2"}


def save(fig,name):
    out=ROOT/"figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out/(name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(out/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def main():
    audit=json.load(open(ROOT/"completion-audit.json"))
    assert audit["status"]=="PASSED"
    assessment=json.load(open(ROOT/"assessment.json"))
    frozen=json.load(open("configs/frozen.json"))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for run in audit["runs"]:
        path=Path("experiments")/run["run_id"]
        rows=[json.loads(line) for line in (path/"training.jsonl").read_text().splitlines()]
        style="-" if run["seed"]==17 else "--"
        label=f"{run['variant']} / seed {run['seed']}"
        axes[0].plot([r["tokens"]/1e6 for r in rows],[r["train_nll"] for r in rows],style,color=COLORS[run["variant"]],linewidth=1.2,label=label)
        validation=[json.loads(line) for line in (path/"validation.jsonl").read_text().splitlines()]
        # In the original logs, tokens denotes EVALUATION tokens (65536).
        # Reconstruct the training horizontal axis from optimizer steps.
        axes[1].plot([r["step"]*frozen["batch_sequences"]*frozen["model"]["seq_len"]/1e6 for r in validation],[r["nll"] for r in validation],style,color=COLORS[run["variant"]],linewidth=1.2,label=label)
    for ax,title in zip(axes,["Training minibatch NLL","Fixed development subset NLL"]):
        ax.set(xlabel="Training prediction tokens (million)",ylabel="NLL (nat/token)",title=title)
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=8)
    fig.suptitle("StateLoop scaled reproduction: fixed 100M-token runs",fontsize=12)
    fig.tight_layout()
    save(fig,"learning-curves")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,variant in enumerate(["vanilla","reset","persistent"]):
        rows=sorted([r for r in audit["runs"] if r["variant"]==variant],key=lambda r:r["seed"])
        axes[0].scatter([i-.06,i+.06],[r["test_nll"] for r in rows],s=50,color=COLORS[variant])
        for offset,r in zip([-.06,.06],rows):
            axes[0].annotate(str(r["seed"]),(i+offset,r["test_nll"]),xytext=(4,3),textcoords="offset points",fontsize=8)
    axes[0].set(xticks=[0,1,2],xticklabels=["Vanilla","Reset","StateLoop"],ylabel="Test NLL (nat/token)",title="Final checkpoints; labels indicate seeds")
    for i,pair in enumerate(audit["recomputed_pairs"]):
        p=pair["persistence"]
        lo,hi=p["document_bootstrap_95pct"]
        axes[1].hlines(i,lo,hi,color=COLORS["persistent"],linewidth=2)
        axes[1].scatter([p["delta_nll"]],[i],color=COLORS["persistent"],s=50)
    axes[1].axvline(0,color="black",linewidth=.8)
    axes[1].axvline(.005,color="gray",linestyle=":",linewidth=1,label="Registered effect threshold")
    axes[1].set(yticks=[0,1],yticklabels=["Seed 17","Seed 42"],xlabel="NLL(reset) - NLL(StateLoop)",title="Paired persistence effect")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.text(.5,-.02,"Intervals: 95% paired document bootstrap, conditional on each trained pair; not seed uncertainty.",ha="center",fontsize=8)
    fig.tight_layout()
    save(fig,"primary-comparison")
    names=["feedback_off","terminal_off","all_off","single_S1","frozen_S1","single_S2","frozen_S2","single_S3","frozen_S3"]
    fig,ax=plt.subplots(figsize=(10,4.5))
    for shift,seed,color in [(-.18,17,"#0891b2"),(.18,42,"#7c3aed")]:
        values={r["intervention"]:r["delta_nll"] for r in assessment["interventions"] if r["seed"]==seed}
        ax.bar(np.arange(len(names))+shift,[values[n] for n in names],width=.36,label=f"Seed {seed}",color=color)
    ax.axhline(0,color="black",linewidth=.8)
    ax.set(xticks=np.arange(len(names)),xticklabels=[n.replace("_"," ") for n in names],ylabel="NLL increase over unmodified StateLoop",title="Read masking and calibrated state reuse")
    plt.setp(ax.get_xticklabels(),rotation=35,ha="right")
    ax.legend()
    ax.grid(axis="y",alpha=.18)
    fig.tight_layout()
    save(fig,"state-interventions")


if __name__=="__main__":
    main()
