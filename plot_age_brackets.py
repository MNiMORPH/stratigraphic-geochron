"""
plot_age_brackets.py — bracketing the age of a bed, with time down the page.

Four panels on one shared vertical age axis, young at the top and old at the
bottom, so the figure reads like a core: material above the bed plots above it,
material below plots below.

  A  Every date on its own, as the full probability density the method actually
     gives (calibrated 14C is lumpy; OSL is Gaussian). Arrows show which way each
     limiting date pushes the event age.
  B  Summed probability density per category — the union of the evidence, making
     no assumption that the dates share an age.
  C  The limiting-age constraint functions: the probability, as a function of
     candidate event age, that the event is younger than everything beneath it
     and older than everything above it.
  D  The joint posterior — panel C's constraint times the OSL likelihood — against
     the two things it is built from.

Run:  python plot_age_brackets.py
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import agedist as ad

DATA_CSV = "synthetic_ages.csv"
OUT_STEM = "age_brackets"

# ── Palette ──────────────────────────────────────────────────────────────────
# Three categorical hues, fixed to categories (never cycled), checked for
# colour-vision separation. The synthesis in panel D wears text ink rather than a
# fourth hue, so it reads as the result rather than as another category.
COLOR = {
    "older_limiting":   "#eb6834",   # orange
    "event":            "#4a3aa7",   # violet
    "younger_limiting": "#2a78d6",   # blue
}
INK, INK_2, GRID_GREY = "#0b0b0b", "#52514e", "#d5d4cf"

LABEL = {
    "older_limiting":   "below the bed ($^{14}$C)",
    "event":            "the bed (OSL)",
    "younger_limiting": "above the bed ($^{14}$C)",
}

AGE_TOP, AGE_BOTTOM = 9.0, 14.9      # ka; top of the page is young

# Densities are plotted per thousand years so the axes carry readable numbers
# instead of a 1e-4 offset tucked under the axis label.
PER_KA = 1000.0

mpl.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def order_dates(df):
    """Sort dates young-to-old within category, categories young-group first."""
    rank = {"younger_limiting": 0, "event": 1, "older_limiting": 2}
    df = df.copy()
    df["_rank"] = df["category"].map(rank)
    df["_med"] = [ad.quantiles(p, (0.5,))[0] for p in df["pdf"]]
    return df.sort_values(["_rank", "_med"]).reset_index(drop=True)


# ── Panel A: every date, one slot each ───────────────────────────────────────
def _support(pdf, frac=0.01):
    """Mask of where a density is worth drawing an outline for.

    Without this the outline of a sideways violin collapses onto the slot centre
    wherever the density is negligible, and every date grows a full-height
    hairline. The default cuts at 1% of peak — for a Gaussian, about +/-3 sigma.
    """
    return pdf >= frac * pdf.max()


def panel_dates(ax, df, ka):
    """Each date's own density, drawn sideways in its own x slot.

    Each density is scaled to the same peak width so that every date is legible.
    That costs nothing in honesty here: the axis that carries precision is the
    VERTICAL one, and it is shared — a tight 14C date is a thin band, a 7% OSL
    age is a fat one, at a glance.
    """
    half = 0.40
    for x, (_, row) in enumerate(df.iterrows()):
        col = COLOR[row["category"]]
        pdf = row["pdf"] / row["pdf"].max()
        m = _support(row["pdf"])

        ax.fill_betweenx(ka, x - half * pdf, x + half * pdf, where=m,
                         facecolor=col, alpha=0.28, lw=0, interpolate=True)
        ax.plot(x + half * pdf[m], ka[m], color=col, lw=1.1)
        ax.plot(x - half * pdf[m], ka[m], color=col, lw=1.1)

        # 95.4% and 68.3% HPD as a spine down the middle of the slot.
        for level, lw in ((0.954, 1.3), (0.683, 3.6)):
            for lo, hi in ad.hpd_intervals(row["pdf"], level):
                ax.plot([x, x], [lo / 1000, hi / 1000], color=col, lw=lw,
                        solid_capstyle="butt", zorder=3)

        # Which way does this date push the event age? Limiting dates only.
        # The arrow starts at the near edge of the 95% region and points into
        # the ages the date permits.
        if row["category"] != "event":
            up = row["category"] == "older_limiting"   # older bound -> push young -> up
            edges = ad.hpd_intervals(row["pdf"], 0.954)
            start = (edges[0][0] if up else edges[-1][1]) / 1000
            ax.annotate("", xy=(x, start - 0.62 if up else start + 0.62), xytext=(x, start),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                        mutation_scale=13), zorder=4)

    ax.set_xlim(-0.8, len(df) - 0.2)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([lab.replace("DEMO-", "") for lab in df["lab_id"]],
                       fontsize=8.5)
    for tick, cat in zip(ax.get_xticklabels(), df["category"]):
        tick.set_color(COLOR[cat])
    ax.set_ylabel("Age (ka cal BP)", fontsize=12)
    ax.set_title("A   Individual dates", loc="left", fontsize=11,
                 fontweight="bold", pad=10)

    # Direct labels: a colour bar grouping each category's slots, named beneath
    # the lab IDs it covers — the grouping belongs with the names, and it keeps
    # the space above the axes free for the panel titles.
    tf = ax.get_xaxis_transform()   # x in data units, y in axes fraction
    for cat, grp in df.groupby("category", sort=False):
        xs = grp.index.values
        ax.plot([xs.min() - 0.42, xs.max() + 0.42], [-0.055, -0.055], transform=tf,
                color=COLOR[cat], lw=3, solid_capstyle="butt", clip_on=False)
        ax.text(xs.mean(), -0.070, LABEL[cat], transform=tf, color=COLOR[cat],
                ha="center", va="top", fontsize=9, clip_on=False)

    ax.legend(handles=[
        Line2D([], [], color=INK_2, lw=3.6, label="68.3% HPD"),
        Line2D([], [], color=INK_2, lw=1.3, label="95.4% HPD"),
        Line2D([], [], color=INK_2, lw=1.6, marker="^", markersize=6,
               label="ages this date permits"),
    ], loc="lower left", fontsize=8, frameon=False)


# ── Panel B: summed probability density per category ─────────────────────────
def panel_spd(ax, df, ka):
    """SPD of each category: pooled evidence, no assumption of a shared age."""
    handles = []
    for cat in ("younger_limiting", "event", "older_limiting"):
        grp = df[df["category"] == cat]
        if grp.empty:
            continue
        spd = ad.summed_density(list(grp["pdf"])) * PER_KA
        m = _support(spd)
        ax.fill_betweenx(ka, 0, spd, facecolor=COLOR[cat], alpha=0.28, lw=0)
        ax.plot(spd[m], ka[m], color=COLOR[cat], lw=1.6)
        handles.append(Patch(facecolor=COLOR[cat], alpha=0.45, edgecolor=COLOR[cat],
                             label=f"{LABEL[cat]}  n={len(grp)}"))
    ax.set_xlim(left=0)
    ax.set_xlabel("Probability density (ka$^{-1}$)", fontsize=9)
    ax.set_title("B   Summed (SPD)", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False)


# ── Panel C: the limiting-age constraint functions ───────────────────────────
def panel_constraints(ax, J, ka):
    """P(event is younger than everything below) and P(older than everything above).

    Each curve is a product of survival/cumulative functions, so it is a
    probability on [0, 1], not a density — which is why it gets its own panel
    instead of being forced onto a density axis.
    """
    both = J["s_older"] * J["f_younger"]
    ax.fill_betweenx(ka, 0, both, facecolor="0.45", alpha=0.20, lw=0)
    ax.plot(J["s_older"], ka, color=COLOR["older_limiting"], lw=2.4, alpha=0.9)
    ax.plot(J["f_younger"], ka, color=COLOR["younger_limiting"], lw=2.4, alpha=0.9,
            ls=(0, (5, 3)))
    ax.plot(both, ka, color=INK, lw=1.1)

    ax.set_xlim(-0.03, 1.08)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Probability", fontsize=9)
    ax.set_title("C   Constraints", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.legend(handles=[
        Line2D([], [], color=COLOR["older_limiting"], lw=2.4,
               label="event younger than\ndates below it"),
        Line2D([], [], color=COLOR["younger_limiting"], lw=2.4, ls=(0, (5, 3)),
               label="event older than\ndates above it"),
        Patch(facecolor="0.45", alpha=0.30, edgecolor=INK, label="both at once"),
    ], loc="lower left", fontsize=7.5, frameon=False, labelspacing=0.8)


# ── Panel D: the joint posterior ─────────────────────────────────────────────
def panel_joint(ax, J, ka):
    """The product: limiting constraint x OSL likelihood, against its ingredients."""
    bracket = J["bracket"] * PER_KA
    event_only = J["event_only"] * PER_KA
    posterior = J["posterior"] * PER_KA

    ax.plot(bracket, ka, color=INK_2, lw=1.3, ls=(0, (5, 2.5)))
    ax.plot(event_only[_support(event_only)], ka[_support(event_only)],
            color=COLOR["event"], lw=1.7, ls=(0, (1.2, 1.6)))
    ax.fill_betweenx(ka, 0, posterior, facecolor=INK, alpha=0.12, lw=0)
    ax.plot(posterior[_support(posterior)], ka[_support(posterior)], color=INK, lw=2.2)

    # The answer, as HPD bars against the axis.
    for level, alpha in ((0.954, 0.18), (0.683, 0.38)):
        for lo, hi in ad.hpd_intervals(J["posterior"], level):
            ax.axhspan(lo / 1000, hi / 1000, xmin=0.0, xmax=0.085,
                       facecolor=INK, alpha=alpha, lw=0, zorder=5)
    mode = J["grid"][np.argmax(J["posterior"])] / 1000
    ax.plot([0, posterior.max()], [mode, mode], color=INK, lw=0.9, alpha=0.5)
    ax.annotate(f"{mode:.2f} ka", xy=(posterior.max(), mode), xytext=(7, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=INK, zorder=6)

    ax.set_xlim(0, posterior.max() * 1.32)
    ax.set_xlabel("Probability density (ka$^{-1}$)", fontsize=9)
    ax.set_title("D   Joint posterior", loc="left", fontsize=11,
                 fontweight="bold", pad=10)
    ax.legend(handles=[
        Line2D([], [], color=INK_2, lw=1.3, ls=(0, (5, 2.5)), label="$^{14}$C bracket alone"),
        Line2D([], [], color=COLOR["event"], lw=1.7, ls=(0, (1.2, 1.6)), label="OSL combined"),
        Line2D([], [], color=INK, lw=2.2, label="joint posterior"),
        Patch(facecolor=INK, alpha=0.38, label="68.3% / 95.4% HPD"),
    ], loc="lower right", fontsize=7.5, frameon=False)


# ── Assembly ─────────────────────────────────────────────────────────────────
def build_figure(csv_path=DATA_CSV, out_stem=OUT_STEM):
    df = order_dates(ad.load_dates(csv_path))
    J = ad.joint_posterior(df)
    ka = J["grid"] / 1000.0

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 8.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[2.6, 1.05, 1.0, 1.2],
                                              wspace=0.13))
    panel_dates(axes[0], df, ka)
    panel_spd(axes[1], df, ka)
    panel_constraints(axes[2], J, ka)
    panel_joint(axes[3], J, ka)

    # Carry the answer across every panel, so the raw dates are read against it.
    post = ad.summarise(J["posterior"])
    for ax in axes:
        for lo, hi in post["hpd95"]:
            ax.axhspan(lo / 1000, hi / 1000, facecolor=INK, alpha=0.045,
                       lw=0, zorder=-10)
        ax.set_ylim(AGE_BOTTOM, AGE_TOP)          # old at the bottom, like a core
        ax.yaxis.grid(True, color=GRID_GREY, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Bracketing the age of a bed — synthetic demonstration data",
        x=0.007, ha="left", fontsize=14.5, fontweight="bold", y=0.99)
    fig.text(
        0.007, 0.960,
        f"Joint posterior for the event: {post['mode'] / 1000:.2f} ka  "
        f"(68.3% {ad.format_hpd(post['hpd68'], envelope=True)}; "
        f"95.4% {ad.format_hpd(post['hpd95'], envelope=True)}; outer bounds).\n"
        f"Time runs down the page.   $^{{14}}$C calibrated with IntCal20.   "
        f"Grey band, repeated on every panel, is the 95.4% region.   "
        f"Densities are drawn down to 1% of their peak.",
        ha="left", va="top", fontsize=9.5, color=INK_2, linespacing=1.5)

    fig.tight_layout(rect=(0.006, 0.055, 0.998, 0.905))
    for ext in ("png", "pdf"):
        # Suppress the PDF creation timestamp: the rendered figures are checked
        # in, and a stamped date makes every rerun show up as a modified file
        # whose content is byte-identical apart from the date.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(f"{out_stem}.{ext}", dpi=200, metadata=meta)
        print(f"wrote {out_stem}.{ext}")
    plt.close(fig)
    return df, J


def report(df, J):
    """Print the numbers behind the figure, so the plot is never the only record."""
    print("\nIndividual dates")
    print(f"  {'lab id':13s}{'category':18s}{'method':7s}{'median':>9s}   95.4% HPD")
    for _, row in df.iterrows():
        s = ad.summarise(row["pdf"])
        print(f"  {row['lab_id']:13s}{row['category']:18s}{row['method']:7s}"
              f"{s['median'] / 1000:8.2f} ka   {ad.format_hpd(s['hpd95'])}")

    print("\nEvent age")
    for key, name in (("bracket", "14C bracket alone"),
                      ("event_only", "OSL combined"),
                      ("posterior", "joint posterior")):
        s = ad.summarise(J[key])
        print(f"  {name:20s} mode {s['mode'] / 1000:6.2f} ka   "
              f"68.3% {ad.format_hpd(s['hpd68'])}   95.4% {ad.format_hpd(s['hpd95'])}")


if __name__ == "__main__":
    report(*build_figure())
