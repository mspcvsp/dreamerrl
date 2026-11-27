"""Editorial layouts for the LinkedIn figures, composed in export pixels."""
import json
from pathlib import Path
from statistics import mean

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from contrast import require_contrast

HERE = Path(__file__).parent
STYLE = json.loads((HERE / "linkedin_style.json").read_text())
P = json.loads((HERE / "palette.json").read_text())
D = json.loads((HERE / "evidence.json").read_text())
DPI = STYLE["dpi"]


class Layout:
    def __init__(self, title, preset="standard"):
        self.w, self.h = STYLE["presets"][preset]
        self.fig = plt.figure(figsize=(self.w / DPI, self.h / DPI), dpi=DPI, facecolor=P["ivory"])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, self.w), ylim=(self.h, 0))
        self.ax.axis("off")
        self.title = title
        self.labels = []
        self.pairs = {}
        self.fig.linkedin_layout = self
        if title:
            self.text(75, 113, title, "title")

    def color(self, fg, bg, role):
        key = f"{fg}/{bg}/{role}"
        recorded = STYLE["approved_pairs"].get(key)
        ratio = require_contrast(P[fg], P[bg], STYLE["thresholds"][role])
        if recorded != round(ratio, 3):
            raise ValueError(f"Record calculated contrast for {key}: {ratio:.3f}")
        self.pairs[key] = recorded
        return P[fg]

    def text(self, x, y, content, role="secondary", *, bg="ivory", fg="navy", align="left"):
        artist = self.ax.text(x, y, content, ha=align, va="center", linespacing=1.15,
                              fontsize=STYLE["type_px"][role]*72/DPI,
                              color=self.color(fg,bg,"essential" if role!="secondary" and bg!="mist" else "text"),
                              weight="bold" if role!="secondary" else "medium")
        self.labels.append(artist)

    def box(self, x, y, w, h, bg):
        self.ax.add_patch(Rectangle((x,y),w,h,facecolor=P[bg],edgecolor="none"))

    def line(self, xs, ys, fg="navy", *, dashed=False, bg="ivory"):
        self.ax.plot(xs,ys,color=self.color(fg,bg,"indicator"),lw=2.35,
                     linestyle="--" if dashed else "-",solid_capstyle="round")

    def arrow(self, a, b, fg="navy"):
        self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,
                                        lw=2.35,color=self.color(fg,"ivory","indicator")))

    def validate(self):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        placed=[]
        for label in self.labels:
            box=label.get_window_extent(renderer)
            if not (45 <= box.x0 < box.x1 <= self.w-45 and 30 <= box.y0 < box.y1 <= self.h-30):
                raise ValueError(f"Clipped or crowded label: {label.get_text()}")
            for previous, bounds in placed:
                if box.overlaps(bounds):
                    raise ValueError(f"Overlapping labels: {previous!r} and {label.get_text()!r}")
            placed.append((label.get_text(), box))
        return dict(width_px=self.w,height_px=self.h,contrast_ratios=self.pairs)


def cards(g, entries, y=270, h=300, margin=75, gap=52):
    width=(g.w-2*margin-gap*(len(entries)-1))/len(entries)
    centers=[]
    for i,(heading, detail,bg) in enumerate(entries):
        x=margin+i*(width+gap)
        g.box(x,y,width,h,bg)
        centers.append(x+width/2)
        g.text(centers[-1],y+h*(.29 if detail else .5),heading,"primary",
               align="center",bg=bg,fg="ivory" if bg=="navy" else "navy")
        if detail:
            g.text(centers[-1],y+h*.71,detail,bg=bg,align="center",fg="ivory" if bg=="navy" else "navy")
        if i:
            g.arrow((x-gap+6,y+h/2),(x-7,y+h/2))
    return centers


def cover():
    g=Layout(None,"cover")
    g.text(90,143,"Beyond Reconstruction","title")
    g.text(90,248,"Memory and information bottlenecks in DreamerV3")
    cards(g,[("Observation",None,"mist"),("Soft posterior",None,"pale_teal"),
             ("Hard state",None,"pale_orange")],330,186,90,62)
    return g.fig


def conductor():
    g=Layout("Human-directed AI engineering")
    xs=cards(g,[("Human\ndirection","Define goals","navy"),("AI\nassistants","Microsoft\nCopilot\nOpenAI Codex","pale_teal"),
                ("Build &\ntest","Debug · run\nAnalyze","mist"),("Evidence","Results\nTools","pale_orange")],213,365,65,42)
    g.line([xs[-1],xs[-1],xs[0]],[592,697,697],"dark_orange")
    g.arrow((xs[0],697),(xs[0],591),"dark_orange")
    g.text(800,796,"Review → Verify → Refine or approve","primary",align="center")
    return g.fig


def architecture():
    g=Layout("How Dreamer learns")
    cards(g,[("Observe","Inputs and\npast actions","mist"),("Encode","Recurrent\nmodel state","pale_teal"),
             ("Imagine","Predict likely\noutcomes","mist"),("Practice","Train actor\nand critic","pale_teal")],249,361)
    g.text(75,753,"Conceptual architecture · DreamerV3")
    return g.fig


def behavior():
    g=Layout("CAGE2 results varied by opponent","detailed")
    for name,x in (("Opponent",75),("Flat",510),("Two levels",1055)):
        g.text(x,225,name,"primary")
    for j,(opponent,data) in enumerate(D["behavior"].items()):
        y=377+j*245
        g.text(75,y,opponent,"primary")
        for kind,left,fg,marker in (("flat",510,"dark_teal","o"),
                                    ("hierarchical",1055,"dark_orange","s")):
            g.text(left,y,f"{data[kind]['mean']:.1f}","number")
            x=lambda v:left+395*(v+42)/44
            g.line([left,left+395],[y+124,y+124])
            for row, value in enumerate(D["behavior_seed_means"][opponent][kind]):
                g.ax.plot(x(value),y+78+(row-1)*25,marker=marker,
                          color=g.color(fg,"ivory","indicator"),markersize=5.4)
    g.text(75,1082,"Dots: 3 seed means per design · 100 episodes per seed")
    g.text(75,1142,"Large values: design means · higher is better")
    return g.fig


def maze():
    g=Layout("The clue disappears")
    g.line([225,790],[341,341],"dark_teal")
    g.line([509,509],[341,685],"dark_teal")
    for i in range(8):
        g.ax.plot(537,383+i*36,"o",color=g.color("navy","ivory","indicator"),markersize=4)
    g.ax.plot(509,691,"D",color=g.color("dark_orange","ivory","indicator"),markersize=11)
    g.text(205,253,"Left","primary")
    g.text(719,253,"Right","primary")
    g.text(577,698,"Clue at reset","primary")
    g.text(908,445,"Same eight moves\nfor both clues","primary")
    g.text(75,815,"Conceptual maze · the cue determines the rewarded branch")
    return g.fig


def bars(g, rows, ys, *, approximate=(), x=707):
    w=540
    for (name,v,fg),y in zip(rows,ys,strict=True):
        g.text(75,y,name,"primary")
        g.box(x,y-30,w*v,60,fg)
        g.color(fg,"ivory","indicator")
        value=f"{100*v:.1f}%" if v not in (.5,1.) else f"{100*v:.0f}%"
        g.text(1520,y,("≈" if name in approximate else "")+value,"number",align="right")
    if all(v in (.5,1.) for _,v,_ in rows):
        for y in ys:
            g.line([x+w/2,x+w/2],[y+36,y+55],dashed=True)


def bottleneck():
    g=Layout(None)
    g.text(75,112,"Categorical sampling retained\nnegligible cue information.","title")
    reset=D["reset"]
    bars(g,[(label,mean(reset[key]),color) for key,label,color in (
        ("encoder","Encoder","dark_teal"),("probabilities","Posterior probabilities","dark_teal"),
        ("z","Argmax states","dark_orange"),("sample","Sampled states","dark_orange"))],
        [264,371,478,614],approximate=("Sampled states",),x=735)
    g.text(75,537,"Identical for opposing cues")
    low,high=min(D["bounds"].values()),max(D["bounds"].values())
    g.text(75,738,f"Single-sample Bayes accuracy upper bounds: {100*low:.2f}–{100*high:.2f}%")
    g.text(75,822,"Linear probes · chance 50% · 3 seeds · 16 held-out pairs each")
    return g.fig


def intervention():
    g=Layout("Retained clue, same choices")
    factual=[row["factual"] for row in D["intervention"].values()]
    bars(g,[("Read the clue",mean(x["junction_h_accuracy"] for x in factual),"dark_teal"),
            ("Choose correctly",mean(x["decision_accuracy"] for x in factual),"dark_orange")],[325,535])
    g.text(75,744,"Factual reset-state injection · chance 50%")
    return g.fig


def training_signal():
    g=Layout("Averages hid the critical cell")
    data=D["reconstruction"].values()
    bars(g,[("All cells",mean(x["all_accuracy"] for x in data),"dark_teal"),
            ("Cue cell",mean(x["cue_accuracy"] for x in data),"dark_orange")],[240,376])
    g.text(75,521,"Terminal reward’s loss window","primary")
    for i in range(10):
        bg="pale_orange" if i in (0,9) else "mist" if i>=2 else "ivory"
        color=g.color("navy",bg,"indicator")
        g.ax.add_patch(Rectangle((75+149*i,576),132,72,facecolor=P[bg],edgecolor=color,
                                 lw=1.2,hatch="//" if i<2 else None))
    g.text(75,700,"Cue")
    g.text(1515,700,"Reward",align="right")
    g.text(75,808,"Hatched rows · detached context")
    return g.fig


def cage_evidence():
    g=Layout("A small raw-history gain","detailed")
    data=D["return_forecast"]["representations"]
    x=lambda v:690+(v+.052)/.081*790
    g.line([x(0),x(0)],[228,698],dashed=True)
    for i,(key,name) in enumerate((("observation","Raw observations"),("posterior","Posterior"),("hz","Combined"))):
        row=data[key]["paired_history_increment"]["fixed_mlp_minus_current_mlp"]
        y=318+i*160
        g.text(75,y,name,"primary")
        g.line([x(n) for n in row["delta_r2_ci95"]],[y,y],"dark_teal")
        g.ax.plot(x(row["delta_r2"]),y,"o",color=g.color("navy","ivory","indicator"),markersize=8)
    for v in (-.05,0,.02):
        g.text(x(v),762,f"{v:+.02f}" if v else "0",align="center")
    g.text(75,853,"Change in test R² · 95% interval · 72 test episodes")
    g.box(75,930,1450,185,"pale_orange")
    c=D["common_support"]["coverage"]
    g.text(108,985,"Separate hidden-history study","primary",bg="pale_orange")
    g.text(108,1060,f"{c['fit']['pairs']} fit pair, {c['validation']['pairs']} validation pairs · test stayed closed",bg="pale_orange")
    return g.fig


FIGURES=dict(cover=cover, conductor=conductor, architecture=architecture, behavior=behavior,
             maze=maze, bottleneck=bottleneck, intervention=intervention, training_signal=training_signal,
             cage_evidence=cage_evidence)
