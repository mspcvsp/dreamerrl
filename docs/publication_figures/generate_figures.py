"""Render publication graphics from the tracked aggregate evidence snapshot."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

HERE = Path(__file__).parent
P = json.loads((HERE / "palette.json").read_text())
D = json.loads((HERE / "evidence.json").read_text())
N, T, M, O, I = (P[k] for k in ("navy", "teal", "mist", "orange", "ivory"))
FONT = next((name for name in (P["preferred_font"], P["fallback_font"])
             if name in {face.name for face in font_manager.fontManager.ttflist}), "sans-serif")
plt.rcParams.update({"font.family": FONT, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "text.color": N, "axes.labelcolor": N, "figure.facecolor": I, "axes.facecolor": I,
                     "savefig.facecolor": I, "axes.unicode_minus": True})


def canvas(size, title=None, footer=None, dark=False):
    fig = plt.figure(figsize=size, facecolor=N if dark else I)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    if title:
        label(ax, .065, .90, title, 22, bold=True, color=I if dark else N)
    if footer:
        label(ax, .065, .058, footer, 11, color=I if dark else N)
    return fig, ax


def label(ax, x, y, s, size=16, *, bold=False, align="left", color=N):
    return ax.text(x, y, s, fontsize=size, va="center", ha=align, color=color,
                   weight="bold" if bold else "normal", linespacing=1.14)


def box(ax, x, y, w, h, s, color=M, size=16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.022",
                                facecolor=color, edgecolor="none"))
    label(ax, x+w/2, y+h/2, s, size, align="center", bold=True)


def arrow(ax, a, b, color=N, curve=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=2.2, color=color,
                                 connectionstyle=f"arc3,rad={curve}"))


def boxes(ax, items, ys=.40, h=.22, size=16, colors=(M,T,O)):
    gap = .055
    width = (.88 - gap*(len(items)-1))/len(items)
    for j, item in enumerate(items):
        x = .06 + j*(width+gap)
        box(ax, x, ys, width, h, item, colors[j % len(colors)], size)
        if j:
            arrow(ax, (x-gap+.003, ys+h/2), (x-.007, ys+h/2))


def cover():
    fig, ax = canvas((10, 5.35))
    label(ax, .07, .79, "Beyond Reconstruction", 38, bold=True)
    label(ax, .07, .65, "Memory and information bottlenecks in DreamerV3", 20)
    boxes(ax, ("Observation", "Soft posterior", "Hard state"), .23, .24, 24)
    return fig


def architecture(scout=False):
    fig, ax = canvas((9.6, 3.7) if scout else (5.1, 5.1),
                     None if scout else "How Dreamer learns",
                     None if scout else "Conceptual architecture · DreamerV3")
    if scout:
        boxes(ax, ("Observe\nthe world", "Build\na model", "Practice\nchoices"), .38, .27, 26, (M,T,M))
        label(ax,.5,.19,"Real experience updates the model.",22,align="center")
    else:
        for y, txt, color in ((.68,"Observation + past action",M),(.49,"Encoder + recurrent state",T),
                              (.30,"Imagine likely outcomes",M),(.11,"Train actor and critic",I)):
            box(ax,.11,y,.78,.13,txt,color,17)
        for y in (.68,.49,.30):
            arrow(ax,(.5,y-.008),(.5,y-.054))
    return fig


def behavior(scout=False):
    fig, ax = canvas((9.6, 3.7) if scout else (5.1, 5.3), None if scout else "Similar scores in this test",
                     None if scout else "Dots · training seeds    |    Tick · mean")
    cohorts = D["behavior"]
    if scout:
        for x,heading in ((.13,"Opponent"),(.52,"Flat"),(.79,"Two levels")):
            label(ax,x,.88,heading,23,align="center",bold=True)
        for i,(opponent,data) in enumerate(cohorts.items()):
            row=.68-i*.22
            box(ax,.05,row-.075,.23,.15,opponent,M,23)
            for x,key in ((.52,"flat"),(.79,"hierarchical")):
                label(ax,x,row,f"{data[key]['mean']:.1f}",31,align="center",bold=True)
        return fig
    x0,x1=.31,.9

    def scale(n):
        return x0+(x1-x0)*(n+42)/44

    for j,(opponent,data) in enumerate(cohorts.items()):
        y=.74-j*.24
        label(ax,.06,y+.055,opponent,17,bold=True)
        for var,off,col,mark,name in (("flat",0,T,"o","Flat"),("hierarchical",-.105,O,"s","Levels")):
            score=data[var]["mean"]
            label(ax,.063,y+off,name,13)
            ax.plot([x0,x1],[y+off,y+off],color=M,lw=1)
            seeds=D["behavior_seed_means"][opponent][var]
            for offset,value in zip((-.013,0,.013),seeds):
                ax.plot(scale(value),y+off+offset,marker=mark,color=col,ms=6,mec=N,mew=.6)
            ax.vlines(scale(score),y+off-.021,y+off+.021,color=N,lw=2)
            label(ax,.976,y+off,f"{score:.1f}",10,align="right")
    return fig


def maze(scout=False):
    fig, ax = canvas((9.6,3.7) if scout else (5.1,4.8), None if scout else "The clue disappears",
                     None if scout else "Conceptual maze · matched eight-step stem")
    ax.plot([.20,.80],[.69,.69],color=M,lw=24,solid_capstyle="round")
    ax.plot([.50,.50],[.25,.69],color=M,lw=24,solid_capstyle="round")
    for j in range(8):
        ax.plot(.50,.28+j*.052,"o",ms=4,color=N,alpha=.75)
    ax.plot(.50,.22,marker="D",ms=17,color=O,mec=N,mew=1)
    label(ax,.20,.80,"LEFT",19,align="center",bold=True)
    label(ax,.80,.80,"RIGHT",19,align="center",bold=True)
    label(ax,.55,.22,"Clue at start",21 if scout else 16,bold=True)
    label(ax,.54,.42,"Same eight moves",20 if scout else 14)
    return fig


def bars(ax, entries, *, start=.72, step=.17, scale=.40, x=.47, size=17):
    for i,(name,val,color) in enumerate(entries):
        y=start-i*step
        label(ax,.06,y,name,size,bold=True)
        ax.add_patch(Rectangle((x,y-.035),scale,.07,facecolor=M,alpha=.34))
        ax.add_patch(Rectangle((x,y-.035),scale*val,.07,facecolor=color))
        label(ax,x+scale+.028,y,f"{100*val:.1f}%" if val not in (0,.5,1) else f"{100*val:g}%",size,bold=True)


def bottleneck(scout=False):
    fig, ax = canvas((9.6,3.7) if scout else (5.1,5.05), None if scout else "The hard state lost the cue")
    reset=D["reset"]
    if scout:
        entries=[("Clue in the picture",mean(reset["raw"]),M),("Clue in the model's probabilities",mean(reset["probabilities"]),T),
                 ("Clue after one hard choice",mean(reset["sample"]),O)]
        bars(ax,entries,start=.78,step=.23,scale=.29,x=.58,size=22)
        label(ax,.06,.09,"50% is chance for two equally likely clues.",18)
    else:
        entries=[("Encoder",mean(reset["encoder"]),T),("Soft posterior",mean(reset["probabilities"]),T),
                 ("Argmax",mean(reset["z"]),O),("One sample",mean(reset["sample"]),O)]
        bars(ax,entries,scale=.38,x=.43)
        low, high=sorted(D["bounds"].values())[::2]
        label(ax,.06,.105,f"Ideal single draw ≤ {100*low:.2f}–{100*high:.2f}%",15,bold=True)
        label(ax,.06,.052,"Three seeds · 16 held-out pairs each · chance 50%",10)
    return fig


def intervention(scout=False):
    fig,ax=canvas((9.6,3.7) if scout else (5.1,4.65),None if scout else "Retained clue, same choices")
    factual=[row["factual"] for row in D["intervention"].values()]
    a,b=mean(x["junction_h_accuracy"] for x in factual),mean(x["decision_accuracy"] for x in factual)
    bars(ax,[("Read the clue",a,T),("Choose correctly",b,O)],start=.69 if scout else .65,
         step=.28 if scout else .25,scale=.37 if scout else .34,x=.48 if scout else .44,size=22 if scout else 16)
    label(ax,.06,.13,"Selected model states · chance 50%",18 if scout else 12)
    return fig


def training_signal():
    fig,ax=canvas((5.1,5.1),"Averages hid the critical cell")
    r=D["reconstruction"]
    bars(ax,[("All cells",mean(x["all_accuracy"] for x in r.values()),T),
             ("Cue cell",mean(x["cue_accuracy"] for x in r.values()),O)],start=.70,step=.18,scale=.39,x=.39)
    label(ax,.06,.33,"Terminal reward’s loss window",15,bold=True)
    for j in range(10):
        ax.add_patch(Rectangle((.07+j*.087,.22),.075,.063,facecolor=O if j in (0,9) else M if j>=2 else I,
                               edgecolor=N,lw=.6,hatch="//" if j<2 else None))
    label(ax,.07,.15,"Cue",11,bold=True)
    label(ax,.85,.15,"Reward",11,bold=True,align="center")
    label(ax,.06,.064,"Hatched rows · detached context",11)
    return fig


def cage_evidence():
    fig,ax=canvas((5.1,5.4),"A small raw-history gain")
    rows=D["return_forecast"]["representations"]
    u=lambda x:.41+(x+.052)/.081*.47
    ax.plot([u(0),u(0)],[.35,.79],color=N,lw=1,ls="--")
    for i,(key,text) in enumerate((("observation","Raw input"),("posterior","Posterior"),("hz","Combined"))):
        delta=rows[key]["paired_history_increment"]["fixed_mlp_minus_current_mlp"]
        y=.74-i*.15
        label(ax,.06,y,text,15,bold=True)
        low,hi=delta["delta_r2_ci95"]
        ax.plot([u(low),u(hi)],[y,y],color=P["dark_teal"],lw=3)
        ax.plot(u(delta["delta_r2"]),y,marker="o",ms=7,color=N)
    for v in (-.05,0,.02):
        label(ax,u(v),.31,f"{v:+.02f}" if v else "0",12,align="center")
    label(ax,.06,.235,"Change in return forecast R² · 95% interval",12)
    c=D["common_support"]["coverage"]
    label(ax,.06,.123,f"Separate study · {c['fit']['pairs']} fit pair, {c['validation']['pairs']} validation pairs",13,bold=True)
    label(ax,.06,.067,"The hidden-history test stayed closed.",11)
    return fig


def loop():
    fig,ax=canvas((9.6,3.7))
    box(ax,.08,.39,.30,.27,"Agent",T,30)
    box(ax,.62,.39,.30,.27,"Environment",M,30)
    arrow(ax,(.38,.57),(.61,.57)); arrow(ax,(.62,.45),(.39,.45))
    label(ax,.5,.73,"Action",23,align="center")
    label(ax,.5,.25,"Observation + reward",22,align="center")
    return fig


def imagine():
    fig,ax=canvas((9.6,3.7))
    box(ax,.07,.38,.27,.24,"What I see",M,25)
    box(ax,.65,.63,.28,.20,"Choice A",T,25)
    box(ax,.65,.19,.28,.20,"Choice B",M,25)
    arrow(ax,(.35,.52),(.64,.72)); arrow(ax,(.35,.47),(.64,.30))
    label(ax,.48,.50,"Imagine",20,bold=True,align="center")
    return fig


def network():
    fig,ax=canvas((9.6,3.7))
    box(ax,.08,.36,.26,.27,"Blue defender",T,25)
    box(ax,.69,.64,.25,.20,"Workstations",M,23)
    box(ax,.69,.19,.25,.20,"Service",M,23)
    for y in (.73,.29):
        arrow(ax,(.35,.49),(.68,y))
    label(ax,.49,.83,"Simulator",22,align="center",bold=True)
    return fig


def workflow():
    fig,ax=canvas((9.6,3.7))
    boxes(ax,("Read on\nGitHub","View a\nresult","Check the\nevidence"),.39,.26,21,(M,T,M))
    label(ax,.5,.19,"Use the saved results to ask a new question.",20,align="center")
    return fig


def careers():
    fig,ax=canvas((9.6,3.7))
    boxes(ax,("Build\nsoftware","Defend\nsystems","Test\nideas"),.38,.27,24,(M,T,M))
    label(ax,.5,.18,"Try one skill with a team this month.",21,align="center")
    return fig


def conductor(scout=False):
    fig, ax = canvas((9.6,3.7) if scout else (5.3,6.6),
                     None if scout else "Human-directed AI engineering")
    if scout:
        boxes(ax, ("Ask a clear\nquestion", "Get AI help\nwith one piece", "Test what\nhappened"),
              .47,.25,24,(M,T,O))
        label(ax,.5,.855,"Microsoft Copilot + OpenAI Codex",19,align="center")
        ax.plot([.81,.81,.18],[.44,.31,.31],color=O,lw=2.2)
        arrow(ax,(.18,.31),(.18,.44),O)
        label(ax,.5,.16,"Use the evidence to revise your instructions.",20,align="center",bold=True)
    else:
        rows = (
            (.68, "Human technical director", "Objectives · requirements · domain knowledge\nScientific questions · acceptance criteria · decisions", N),
            (.48, "AI-assisted engineering", "Microsoft Copilot · OpenAI Codex\nImplement · debug · test · run · analyze · document", T),
            (.28, "Project outputs", "DreamerV3 implementation · experimental results\nDiagnostic tools · reproducible publications", M),
            (.08, "Verify and review", "Evaluate tests and evidence. Refine instructions.", O),
        )
        for y, heading, detail, color in rows:
            ax.add_patch(FancyBboxPatch((.065,y),.84,.16,boxstyle="round,pad=.008,rounding_size=.02",
                                        facecolor=color,edgecolor="none"))
            ink=I if color==N else N
            label(ax,.095,y+.113,heading,17,bold=True,color=ink)
            label(ax,.095,y+.051,detail,11.7,color=ink)
        for y in (.68,.48,.28):
            arrow(ax,(.48,y-.006),(.48,y-.034))
        ax.plot([.92,.948,.948],[.16,.16,.76],color=O,lw=2.3)
        arrow(ax,(.948,.76),(.914,.76),O)
        label(ax,.065,.033,"Define → Direct → Implement → Test → Review → Refine",11,bold=True)
    return fig


FIGURES = {"linkedin": dict(cover=cover,conductor=conductor,architecture=architecture,behavior=behavior,maze=maze,
                            bottleneck=bottleneck,intervention=intervention,training_signal=training_signal,cage_evidence=cage_evidence),
           "presentation": dict(loop=loop,imagine=imagine,architecture=lambda:architecture(True),network=network,conductor=lambda:conductor(True),
                                behavior=lambda:behavior(True),maze=lambda:maze(True),bottleneck=lambda:bottleneck(True),
                                intervention=lambda:intervention(True),workflow=workflow,careers=careers)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    provenance = json.loads((HERE / "figure_manifest.json").read_text())
    if {a: set(g) for a, g in FIGURES.items()} != {a: set(g) for a, g in provenance.items()}:
        raise ValueError("Figure builders and provenance inventory disagree")
    for audience, graphics in FIGURES.items():
        dest=args.output/audience/"figures"
        dest.mkdir(parents=True,exist_ok=True)
        for name,builder in graphics.items():
            fig=builder()
            for suffix in ("svg","png"):
                fig.savefig(dest/f"{name}.{suffix}",dpi=320)
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(dest/f"{name}.pdf"),
                            str(dest/f"{name}.svg")], check=True)
            plt.close(fig)
    (args.output/"figure_provenance.json").write_text(json.dumps(dict(
        matplotlib=matplotlib.__version__, typography=FONT, dpi=320, figure_sources=provenance,
        source_sha256=D["source_sha256"]), indent=2)+"\n")
    print(f"Figure typography: {FONT}")


if __name__=="__main__":
    main()
