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

    def text(self, x, y, content, role="secondary", *, bg="ivory", fg="navy", align="left", size=None):
        artist = self.ax.text(x, y, content, ha=align, va="center", linespacing=1.15,
                              fontsize=(size if size is not None else STYLE["type_px"][role])*72/DPI,
                              color=self.color(fg,bg,"essential" if role!="secondary" and bg!="mist" else "text"),
                              weight="bold" if role!="secondary" else "medium")
        self.labels.append(artist)

    def box(self, x, y, w, h, bg):
        self.ax.add_patch(Rectangle((x,y),w,h,facecolor=P[bg],edgecolor="none"))

    def line(self, xs, ys, fg="navy", *, dashed=False, bg="ivory", lw=2.35):
        self.ax.plot(xs,ys,color=self.color(fg,bg,"indicator"),lw=lw,
                     linestyle="--" if dashed else "-",solid_capstyle="round")

    def arrow(self, a, b, fg="navy", *, lw=2.35, head=10, shrink=2):
        self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=head,
                                        lw=lw,shrinkA=shrink,shrinkB=shrink,color=self.color(fg,"ivory","indicator")))

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
    g = Layout(None, "cover")
    g.text(72, 93, "Beyond Reconstruction", "title", size=65)
    g.text(74, 153, "Tracing Information Flow in DreamerV3", size=37)
    g.line([72, 1528], [188, 188], lw=1.0)
    g.line([815, 815], [219, 790], "dark_teal", lw=.55)
    cover_architecture(g)
    cover_cue_results(g)
    g.line([72, 1528], [812, 812], lw=.75)
    g.text(72, 845, "Hafner et al. (2025). “Mastering diverse control tasks through world models.” Nature, 640, 647–653.",
           size=22)
    g.text(1528, 845, "DOI: 10.1038/s41586-025-08744-2", align="right", size=22)
    return g.fig


def cover_architecture(g):
    g.text(72, 220, "DreamerV3 implementation", "primary", size=35)

    # Both conceptual environments feed the same observation path.
    g.box(72, 250, 318, 102, "mist")
    g.box(458, 250, 330, 102, "pale_teal")
    g.text(181, 278, "Simulated cyber defense", bg="mist", size=17)
    g.text(181, 316, "TTCP CAGE Challenge #2", bg="mist", size=16)
    for x, y in ((107, 276), (107, 327), (145, 302)):
        g.line([x+8, 153], [y+5, 307], bg="mist", lw=.7)
    for x, y in ((92, 267), (92, 316), (146, 291)):
        g.box(x, y, 26, 18, "navy")
        g.box(x+4, y+4, 18, 10, "ivory")

    g.text(577, 272, "One-clue T-maze", "primary", bg="pale_teal", size=22)
    g.text(577, 304, "Left or right at junction", bg="pale_teal", size=17)
    g.text(577, 333, "Cue visible only at start", bg="pale_teal", size=17)
    g.line([481, 553], [274, 274], "dark_teal", bg="pale_teal", lw=1.0)
    g.line([517, 517], [283, 324], "dark_teal", dashed=True, bg="pale_teal", lw=1.0)
    g.ax.plot(517, 334, "D", ms=3.7, color=g.color("dark_orange", "pale_teal", "indicator"))

    def node(x, y, w, heading, detail=None):
        g.box(x, y, w, 62, "pale_teal")
        g.text(x+w/2, y+(19 if detail else 31), heading, "primary",
               align="center", bg="pale_teal", size=27)
        if detail:
            g.text(x+w/2, y+46, detail, align="center", bg="pale_teal", size=19)

    def route(points):
        if len(points) > 2:
            g.line(*zip(*points[:-1]), fg="dark_teal", lw=.8)
        g.arrow(points[-2], points[-1], "dark_teal", lw=.8, head=5, shrink=.5)

    g.line([231, 231, 623, 623], [352, 368, 368, 352], "dark_teal", lw=.8)
    route([(231, 368), (231, 414)])
    g.text(248, 389, "Observation xₜ", size=21)
    node(72, 414, 318, "Encoder")
    node(458, 414, 330, "Previous step", "hₜ₋₁, zₜ₋₁, aₜ₋₁")
    route([(231, 476), (231, 513)])
    g.text(249, 496, "Features eₜ", size=20)
    route([(623, 476), (623, 513)])
    node(72, 513, 318, "Posterior")
    node(458, 513, 318, "GRU", "Deterministic state hₜ")
    route([(458, 544), (390, 544)])
    g.text(424, 526, "hₜ", align="center", size=20)
    route([(231, 575), (231, 623)])
    g.text(249, 600, "Probabilities", size=20)
    node(72, 623, 318, "Categorical Sampling", "Discrete latent representation")
    node(458, 623, 318, "Actor + Critic", "Policy and value")
    route([(390, 654), (458, 654)])
    g.text(424, 636, "zₜ", align="center", size=20)
    route([(776, 544), (800, 544), (800, 654), (776, 654)])

    # Branch each current state to the next update. Only the selected action
    # comes from the policy. The critic receives the same states to predict value.
    route([(424, 654), (424, 712), (231, 712), (231, 742)])
    route([(800, 654), (800, 712), (747, 712), (747, 742)])
    route([(623, 685), (623, 742)])
    g.text(610, 706, "Actor’s action aₜ", align="right", size=20)
    g.box(72, 742, 716, 51, "pale_teal")
    g.text(430, 767, "Next state  hₜ₊₁ = RSSMCore(hₜ, zₜ, aₜ)", "primary",
           align="center", bg="pale_teal", size=25)


def cover_cue_results(g):
    g.text(840, 220, "T-maze cue recovery", "primary", size=35)
    g.box(840, 248, 688, 86, "pale_orange")
    g.box(840, 248, 8, 86, "orange")
    g.text(864, 291, "Categorical Compression Obscured\nthe Original T-Maze Cue",
           "primary", bg="pale_orange", size=29)
    g.text(840, 371, "Held-out linear probe accuracy · 3 final checkpoints", size=24)

    register = json.loads((HERE / "figure_manifest.json").read_text())
    junction = register["linkedin"]["cover"]["junction_linear_probe"]["seed_means"]
    reset = D["reset"]
    # Group only when the displayed stages have exactly the same seed results.
    if not reset["raw"] == reset["encoder"] == reset["probabilities"]:
        raise ValueError("The reset stages need separate bars")
    if junction["sample/h/8/linear"] != junction["sample/hz/8/linear"]:
        raise ValueError("The junction stages need separate bars")
    bars = [
        (992, "Observation, encoder,\nposterior probabilities", mean(reset["raw"]), "dark_teal", ""),
        (1155, "Argmax\nstate", mean(reset["z"]), "dark_orange", ""),
        (1273, "Sampled\nstate", mean(reset["sample"]), "dark_orange", "≈"),
        (1440, "Recurrent h\nand combined h+z", mean(junction["sample/h/8/linear"]), "dark_teal", "≈"),
    ]
    base, height = 640, 200
    for x, name, value, color, prefix in bars:
        length = height * value
        g.color(color, "ivory", "indicator")
        g.box(x-29, base-length, 58, length, color)
        g.text(x, base-length-21, f"{prefix}{value:.0%}", "number", align="center", size=29)
        g.text(x, 675, name, align="center", size=19)

    g.line([906, 1519], [base, base], lw=.7)
    g.line([906, 1519], [base-height/2]*2, dashed=True, lw=.65)
    g.line([906, 906], [base-height, base], lw=.7)
    for pct in (0, 50, 100):
        y = base-height*pct/100
        g.line([899, 906], [y, y], lw=.7)
        g.text(892, y, f"{pct}%", align="right", size=20)
    g.text(892, 563, "Chance", align="right", size=16)
    g.line([1344, 1344], [413, 724], "dark_teal", dashed=True, lw=.55)
    g.text(1102, 724, "Reset", "primary", align="center", size=26)
    g.text(1440, 724, "Junction", "primary", align="center", size=26)

    lo, hi = min(D["bounds"].values()), max(D["bounds"].values())
    g.box(840, 748, 688, 45, "pale_orange")
    g.text(852, 771, "Single-sample Bayes accuracy upper bounds", bg="pale_orange", size=21)
    g.text(1516, 771, f"{lo:.2%}–{hi:.2%}", "primary", bg="pale_orange",
           align="right", size=25)


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


def bars(g, rows, ys, *, approximate=(), x=707, w=540):
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
    g = Layout(None)
    g.text(75, 127, "Information Retained,\nDecisions Unchanged", "title")
    factual = [row["factual"] for row in D["intervention"].values()]
    bars(g, [
        ("Cue recovery\nby probe", mean(x["junction_h_accuracy"] for x in factual), "dark_teal"),
        ("Agent decision\naccuracy", mean(x["decision_accuracy"] for x in factual), "dark_orange"),
    ], [325, 535])
    g.text(75, 744, "Categorical state intervention at reset · Chance baseline: 50%")
    return g.fig


def training_signal():
    g = Layout(None)
    g.text(75, 86, "High Reconstruction Accuracy Masked Cue Loss", "title", size=57)
    data = D["reconstruction"].values()
    bars(g, [
        ("Overall reconstruction\naccuracy", mean(x["all_accuracy"] for x in data), "dark_teal"),
        ("Cue-cell reconstruction\naccuracy", mean(x["cue_accuracy"] for x in data), "dark_orange"),
    ], [225, 355], x=805, w=430)

    g.line([75, 1515], [440, 440], lw=.8)
    g.text(75, 500, "Supervised Training Window", "primary")
    g.text(75, 585, "Initial cue")
    g.text(1515, 585, "Terminal reward", align="right")
    for i in range(10):
        bg = "pale_orange" if i in (0, 9) else "mist" if i >= 2 else "ivory"
        color = g.color("navy", bg, "indicator")
        g.ax.add_patch(Rectangle((75 + 145*i, 625), 135, 62, facecolor=P[bg],
                                 edgecolor=color, lw=1.2, hatch="//" if i < 2 else None))
    g.line([365, 365, 1515, 1515], [702, 728, 728, 702], lw=1.4)
    g.text(75, 810, "Hatched: context only")
    g.text(1515, 810, "Solid: supervised training", align="right")
    return g.fig


def memory_horizon():
    g = Layout("Memory Beyond the Training Window")
    left, right = 75, 1525
    step, width = 146, 136
    burn_in_records = 2
    supervised_start = left + burn_in_records * step
    boundary = supervised_start - (step - width) / 2
    cue = left + width / 2
    reward = right - width / 2

    g.text(800, 205, "Task dependency horizon", "primary", align="center", size=48)
    g.line([cue, cue, reward, reward], [275, 251, 251, 275], lw=1.4)
    g.text(left, 319, "Initial cue")
    g.text(right, 319, "Terminal reward", align="right")
    for record in range(10):
        detached = record < burn_in_records
        bg = "pale_orange" if record in (0, 9) else "ivory" if detached else "mist"
        edge = g.color("navy", bg, "indicator")
        g.ax.add_patch(Rectangle((left + record * step, 361), width, 61,
                                 facecolor=P[bg], edgecolor=edge, lw=1.2,
                                 hatch="//" if detached else None))

    g.line([boundary, boundary], [353, 643], dashed=True, lw=.95)
    g.text((left + boundary) / 2, 484, "Detached\nburn-in context", align="center", size=35)
    g.line([supervised_start, supervised_start, right, right], [435, 461, 461, 435], lw=1.4)
    g.text((supervised_start + right) / 2, 507, "Supervised training horizon",
           "primary", align="center", size=48)

    g.text((boundary + reward) / 2, 596, "Reward-loss gradient", "primary", align="center", size=48)
    g.arrow((reward, 643), (boundary, 643), "dark_orange", lw=1.6, head=12, shrink=0)

    g.text(800, 742, "Desired memory horizon (design objective)", "primary", align="center", size=48)
    g.arrow((cue, 788), (reward, 788), "dark_teal", lw=1.6, head=12, shrink=0)
    g.text(800, 849, "Recurrent memory can outlast the supervised window.", align="center", size=40)
    return g.fig


def cage_evidence():
    g = Layout(None, "detailed")
    g.text(75, 138, "Observation History Modestly\nImproved Return Forecasting", "title")
    forecast = D["return_forecast"]
    axis_min, axis_max = -.052, .029
    plot_left, plot_right = 690, 1480

    def x(value):
        return plot_left + (value - axis_min) / (axis_max - axis_min) * (plot_right - plot_left)

    g.text(x(0), 263, "No improvement", align="center", size=43)
    g.line([x(0), x(0)], [302, 654], dashed=True, lw=1.2)
    for (key, name), y in zip((
        ("observation", "Raw observation\nhistory"),
        ("posterior", "Posterior probability\nhistory"),
        ("hz", "Combined RSSM\nstate history"),
    ), (350, 475, 600), strict=True):
        row = forecast["representations"][key]["paired_history_increment"]["fixed_mlp_minus_current_mlp"]
        g.text(75, y, name, "primary", size=51)
        low, high = row["delta_r2_ci95"]
        g.line([x(low), x(high)], [y, y], "dark_teal", lw=1.8)
        for endpoint in (low, high):
            g.line([x(endpoint), x(endpoint)], [y - 9, y + 9], "dark_teal", lw=1.4)
        g.ax.plot(x(row["delta_r2"]), y, "o", color=g.color("navy", "ivory", "indicator"), markersize=8)

    g.line([plot_left, plot_right], [654, 654], lw=.85)
    for value in (-.05, -.025, 0, .025):
        g.line([x(value), x(value)], [654, 667], lw=.85)
        label = f"{value:+.3f}".rstrip("0").replace("-", "−") if value else "0"
        g.text(x(value), 700, label, align="center", size=40)

    g.text(75, 760, "Change in test R²: five-frame history vs. current state", "primary", size=44)
    g.text(75, 814, f"95% confidence intervals · {forecast['episode_counts']['test']} held-out test episodes", size=43)
    g.text(75, 865, "Posterior and RSSM effects were inconclusive: both intervals include zero.", size=39)

    g.box(75, 927, 1450, 231, "pale_orange")
    coverage = D["common_support"]["coverage"]
    g.text(106, 976, "Separate Hidden-Session History Study", "primary", bg="pale_orange", size=48)
    g.text(106, 1050, f"{coverage['fit']['pairs']} matched training pair, {coverage['validation']['pairs']} validation pairs · "
           "Insufficient matched examples to proceed", bg="pale_orange", size=34)
    g.text(106, 1111, "No probes were trained. Reserved test episodes were untouched.", bg="pale_orange", size=39)
    return g.fig


FIGURES=dict(cover=cover, conductor=conductor, architecture=architecture, behavior=behavior,
             maze=maze, bottleneck=bottleneck, intervention=intervention, training_signal=training_signal,
             memory_horizon=memory_horizon, cage_evidence=cage_evidence)
