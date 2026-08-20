"""
agedist.py — age distributions and the joint posterior for a bracketed event.

Every date in a bracketing problem does one of three jobs:

  older_limiting    the date sits BELOW the bed, so the event is YOUNGER than it.
                    (The literature calls this a *maximum-limiting* age: the event
                    is at most this old. The name is the thing everyone inverts,
                    which is why the categories here are named by direction.)
  event             the date measures the bed of interest ITSELF.
  younger_limiting  the date sits ABOVE the bed, so the event is OLDER than it.
                    (A *minimum-limiting* age.)

Ages live on one common 1-year calendar grid (cal yr BP, increasing = older).
Each date becomes a probability density on that grid:

  14C   calibrated against IntCal20 (via iosacal). Genuinely non-Gaussian —
        skewed, and multi-modal wherever the curve has a plateau.
  OSL   Gaussian, because that is what a luminescence age estimate is.

THE JOINT POSTERIOR
-------------------
Let theta be the true calendar age of the event and p_i(t) the density of date i.

  A date BELOW the bed constrains theta < t_i. Its contribution to the likelihood
  of theta is the probability that the date is older than theta:

      S_i(theta) = P(t_i > theta) = 1 - CDF_i(theta)

  A date ABOVE the bed constrains theta > t_j:

      F_j(theta) = P(t_j < theta) = CDF_j(theta)

  A date ON the bed measures theta directly, so it contributes p_k(theta).

With a uniform prior on theta,

      posterior(theta)  ∝  prod_i S_i(theta) * prod_j F_j(theta) * prod_k p_k(theta)

Note what this is NOT: it is a PRODUCT, not a sum. A summed probability density
(SPD) pools dates that need not share an age -- it answers "when was there dating
activity?" The product answers "when did this ONE event happen, given every date
speaks to it?" and is therefore always sharper than any of its inputs. Panels B
and D of the figure are exactly that contrast.

ASSUMPTIONS worth stating out loud, because the product hides them:
  1. The event has ONE true age. Multiplying the two OSL likelihoods assumes both
     aliquots date the same depositional moment, with no unmodelled scatter
     (no partial bleaching, no post-depositional mixing, no overdispersion).
     If the OSL ages disagree by more than their errors allow, the product is a
     lie and you want a scatter/overdispersion term instead.
  2. The stratigraphy is right: everything in older_limiting really is below and
     everything in younger_limiting really is above.
  3. The dates are independent.
  4. The prior on theta is uniform over the grid.

Written for a synthetic demo dataset; the machinery is general.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# numpy >= 2 renamed trapz -> trapezoid; support both so the folder is portable.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ── The common calendar-age grid ─────────────────────────────────────────────
# 1-yr resolution, wide enough that every PDF decays to zero inside it and the
# limiting-age constraint functions saturate at 0 and 1 well before the edges.
GRID = np.arange(5000, 20001, 1, dtype=float)   # cal yr BP

CATEGORIES = ("older_limiting", "event", "younger_limiting")

# Direction each category constrains the event age, for display and for asserts.
CATEGORY_MEANING = {
    "older_limiting":   "below the bed — the event is YOUNGER than this (maximum-limiting)",
    "event":            "the bed itself — this dates the event",
    "younger_limiting": "above the bed — the event is OLDER than this (minimum-limiting)",
}


# ── 1. Individual date -> density on GRID ────────────────────────────────────
def gaussian_pdf(mu, sigma, grid=GRID):
    """Density of a Gaussian age estimate (OSL, 10Be, ...) on the grid."""
    return _normalise(norm.pdf(grid, loc=mu, scale=sigma), grid)


def calibrate_14c(age_14c, error_14c, curve="intcal20", grid=GRID, label=""):
    """Calibrate a conventional 14C age and resample onto the grid.

    iosacal returns a truncated, descending (cal BP, density) array; we sort it
    ascending and place it on the grid, zero outside its support.
    """
    from iosacal import R  # imported lazily so gaussian-only use needs no iosacal

    cal = np.asarray(R(int(age_14c), int(error_14c), label or "date").calibrate(curve))
    years, dens = cal[:, 0], cal[:, 1]
    order = np.argsort(years)
    years, dens = years[order], dens[order]
    if years.min() < grid.min() or years.max() > grid.max():
        raise ValueError(
            f"calibrated range {years.min():.0f}-{years.max():.0f} cal BP for "
            f"{label or age_14c} falls outside GRID; widen GRID."
        )
    return _normalise(np.interp(grid, years, dens, left=0.0, right=0.0), grid)


def _normalise(density, grid=GRID):
    """Scale a density to unit integral over the grid."""
    density = np.asarray(density, dtype=float)
    integral = _trapz(density, grid)
    if integral <= 0:
        raise ValueError("density integrates to zero; nothing to normalise")
    return density / integral


def load_dates(csv_path, grid=GRID):
    """Read the age table and attach a normalised density to every row.

    Returns the DataFrame with added columns:
      pdf         density on `grid`, unit integral
      cdf         cumulative probability, increasing with age
    """
    df = pd.read_csv(csv_path, comment="#")
    bad = set(df["category"]) - set(CATEGORIES)
    if bad:
        raise ValueError(f"unknown category/categories {sorted(bad)}; expected {CATEGORIES}")

    pdfs, cdfs = [], []
    for _, row in df.iterrows():
        if row["method"] == "14C":
            pdf = calibrate_14c(row["age_yr"], row["error_1s_yr"], label=row["lab_id"], grid=grid)
        elif row["method"] == "OSL":
            pdf = gaussian_pdf(row["age_yr"], row["error_1s_yr"], grid=grid)
        else:
            raise ValueError(f"unknown method {row['method']!r} for {row['lab_id']}")
        pdfs.append(pdf)
        cdfs.append(cumulative(pdf, grid))

    df["pdf"] = pdfs
    df["cdf"] = cdfs
    return df


def cumulative(pdf, grid=GRID):
    """CDF of a density on the grid, increasing with age. cdf[0]=0, cdf[-1]=1."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return cdf / cdf[-1]


# ── 2. Constraint functions and the joint posterior ──────────────────────────
def constraint_older(cdfs, grid=GRID):
    """P(event age < each limiting date), for dates BELOW the bed.

    Product of S_i(theta) = 1 - CDF_i(theta). Goes 1 -> 0 with increasing age:
    the event cannot be older than the material beneath it.
    """
    out = np.ones_like(grid)
    for cdf in cdfs:
        out = out * (1.0 - cdf)
    return out


def constraint_younger(cdfs, grid=GRID):
    """P(event age > each limiting date), for dates ABOVE the bed.

    Product of F_j(theta) = CDF_j(theta). Goes 0 -> 1 with increasing age:
    the event cannot be younger than the material above it.
    """
    out = np.ones_like(grid)
    for cdf in cdfs:
        out = out * cdf
    return out


def combined_likelihood(pdfs, grid=GRID):
    """Product of densities that each measure the SAME event age.

    This is the luminescence equivalent of OxCal's R_Combine: it assumes a single
    true age and no unmodelled scatter (see module docstring, assumption 1).
    """
    out = np.ones_like(grid)
    for pdf in pdfs:
        out = out * pdf
    return _normalise(out, grid)


def summed_density(pdfs, grid=GRID):
    """SPD: the mean of a set of densities. Pools dates WITHOUT assuming one age."""
    return _normalise(np.sum(np.asarray(pdfs), axis=0), grid)


def joint_posterior(df, grid=GRID):
    """Full Bayesian posterior for the event age, plus the pieces that build it.

    Returns a dict of densities/curves on `grid`:
      s_older      product of S_i  — the older-limiting constraint (0-1)
      f_younger    product of F_j  — the younger-limiting constraint (0-1)
      bracket      s_older * f_younger, normalised: the answer from the limiting
                   14C ALONE (a broad plateau — the classic bracket)
      event_only   combined OSL likelihood alone
      posterior    bracket-constraint * event likelihood, normalised: everything
    """
    by = {c: df[df["category"] == c] for c in CATEGORIES}

    s_older = constraint_older(list(by["older_limiting"]["cdf"]), grid)
    f_younger = constraint_younger(list(by["younger_limiting"]["cdf"]), grid)
    constraint = s_older * f_younger

    # An inverted bracket — something below the bed dating younger than something
    # above it — leaves no age the limiting dates permit. Say so; the alternative
    # is a division-by-zero several frames deeper, which reads as a coding error
    # rather than as the geological contradiction it actually is.
    if not np.any(constraint > 0):
        youngest_below = grid[np.argmax(s_older < 0.5)]
        oldest_above = grid[np.argmax(f_younger > 0.5)]
        raise ValueError(
            "the limiting ages leave no permitted age for the event: the material "
            f"below the bed dates to ~{youngest_below:.0f} cal BP or younger, while "
            f"the material above dates to ~{oldest_above:.0f} cal BP or older. "
            "Either the stratigraphic assignments are swapped, or the dates and "
            "the stratigraphy genuinely disagree."
        )

    event_pdfs = list(by["event"]["pdf"])
    event_only = combined_likelihood(event_pdfs, grid) if len(event_pdfs) else np.ones_like(grid)

    return {
        "grid": grid,
        "s_older": s_older,
        "f_younger": f_younger,
        "bracket": _normalise(constraint, grid),
        "event_only": _normalise(event_only, grid),
        "event_spd": summed_density(event_pdfs, grid) if len(event_pdfs) else None,
        "posterior": _normalise(constraint * event_only, grid),
    }


# ── 3. Summarising a density ─────────────────────────────────────────────────
def hpd_intervals(pdf, level=0.954, grid=GRID):
    """Highest-posterior-density region at `level`, as a list of (lo, hi) ages.

    Returns more than one interval when the density is multi-modal — which
    calibrated 14C often is. That is the honest answer; do not collapse it.

    The region is defined by a density THRESHOLD, {t : p(t) >= c}, rather than by
    walking a sorted cumulative sum. The two agree wherever p is strictly ordered,
    but on a plateau — exactly what a 14C bracket alone produces — the walk breaks
    ties arbitrarily and shatters the interval into slivers, while the threshold
    admits all tied cells at once and keeps the region contiguous.
    """
    mass = pdf * np.gradient(grid)
    order = np.argsort(mass)[::-1]
    cum = np.cumsum(mass[order])
    reached = np.flatnonzero(cum >= level * mass.sum())
    cutoff = mass[order][reached[0] if reached.size else -1]
    keep = mass >= cutoff

    edges = np.flatnonzero(np.diff(np.r_[False, keep, False]))
    return [(grid[a], grid[b - 1]) for a, b in edges.reshape(-1, 2)]


def quantiles(pdf, probs=(0.025, 0.5, 0.975), grid=GRID):
    """Ages at the given cumulative probabilities."""
    return np.interp(probs, cumulative(pdf, grid), grid)


def summarise(pdf, grid=GRID):
    """(mode, median, 68.3% HPD, 95.4% HPD) for a density, all in cal yr BP."""
    return {
        "mode": grid[np.argmax(pdf)],
        "median": quantiles(pdf, (0.5,), grid)[0],
        "hpd68": hpd_intervals(pdf, 0.683, grid),
        "hpd95": hpd_intervals(pdf, 0.954, grid),
    }


def format_hpd(intervals, ka=True, envelope=False):
    """Render an HPD region as text.

    By default every disjoint sub-interval is listed: a calibrated 14C HPD really
    is ragged wherever the curve wiggles, and collapsing that silently overstates
    what the date says. Pass envelope=True for the outer bounds only, where space
    forces it (figure annotations) — and say so in the caption.
    """
    scale, unit = (1000.0, " ka") if ka else (1.0, " cal BP")
    if envelope:
        intervals = [(intervals[0][0], intervals[-1][1])]
    return ", ".join(f"{lo / scale:.2f}-{hi / scale:.2f}" for lo, hi in intervals) + unit
