"""
test_agedist.py — checks on the age-distribution machinery.

Run either way:
    python test_agedist.py        # plain asserts, no test runner needed
    pytest test_agedist.py

The tests that matter most are the last two. Everything else here is arithmetic
hygiene; those two check the DIRECTION of the limiting-age constraints, which is
the one thing in this code that is easy to get backwards and impossible to spot
by eye once it is wrong — a sign flip still produces a smooth, plausible-looking
posterior, just centred on the wrong side of the bracket.
"""

import numpy as np
from scipy.stats import norm

import agedist as ad


def test_pdfs_normalise():
    """Every loaded date is a density with unit integral."""
    df = ad.load_dates("synthetic_ages.csv")
    for _, row in df.iterrows():
        area = np.trapz(row["pdf"], ad.GRID)
        assert abs(area - 1.0) < 1e-6, f"{row['lab_id']} integrates to {area}"


def test_cumulative_is_a_cdf():
    pdf = ad.gaussian_pdf(12000, 900)
    cdf = ad.cumulative(pdf)
    assert cdf[0] == 0.0 and abs(cdf[-1] - 1.0) < 1e-12
    assert np.all(np.diff(cdf) >= -1e-15), "CDF must be non-decreasing"
    # Median of a Gaussian sits at its mean.
    assert abs(np.interp(0.5, cdf, ad.GRID) - 12000) < 2


def test_gaussian_hpd_is_one_sigma():
    """68.3% HPD of a Gaussian is mu +/- sigma, to within the grid step."""
    intervals = ad.hpd_intervals(ad.gaussian_pdf(12000, 900), 0.683)
    assert len(intervals) == 1
    lo, hi = intervals[0]
    assert abs(lo - 11100) < 5 and abs(hi - 12900) < 5, (lo, hi)


def test_combined_likelihood_matches_the_analytic_product():
    """Two Gaussians multiplied give the precision-weighted Gaussian.

    mu = (mu1/s1^2 + mu2/s2^2) / (1/s1^2 + 1/s2^2),  1/s^2 = 1/s1^2 + 1/s2^2
    """
    mu1, s1, mu2, s2 = 11850.0, 850.0, 12250.0, 950.0
    got = ad.combined_likelihood([ad.gaussian_pdf(mu1, s1), ad.gaussian_pdf(mu2, s2)])

    prec = 1 / s1**2 + 1 / s2**2
    mu = (mu1 / s1**2 + mu2 / s2**2) / prec
    want = norm.pdf(ad.GRID, loc=mu, scale=np.sqrt(1 / prec))

    assert np.max(np.abs(got - want)) < 1e-9
    assert abs(ad.GRID[np.argmax(got)] - mu) < 2


def test_summed_is_not_the_same_as_combined():
    """SPD and joint answer different questions; they must not coincide.

    The joint of two Gaussians has 1/s^2 = 1/s1^2 + 1/s2^2, so its 95.4% width is
    a number we can derive rather than guess: 4/sqrt(1/s1^2 + 1/s2^2). Pooling the
    same two dates instead of multiplying them must give a wider answer — that
    gap is the whole difference between panels B and D of the figure.
    """
    s1, s2 = 850.0, 950.0
    pdfs = [ad.gaussian_pdf(11850, s1), ad.gaussian_pdf(12250, s2)]
    width = lambda d: sum(hi - lo for lo, hi in ad.hpd_intervals(d, 0.954))

    expected = 4 / np.sqrt(1 / s1**2 + 1 / s2**2)
    assert abs(width(ad.combined_likelihood(pdfs)) - expected) < 20, expected
    assert width(ad.combined_likelihood(pdfs)) < width(ad.summed_density(pdfs))


def test_older_limiting_date_cuts_the_OLD_tail():
    """A date BELOW the bed forbids ages older than it — and nothing else.

    Bites if the constraint is inverted: a flipped sign would cut the YOUNG tail,
    pushing the posterior older instead of younger.
    """
    cutoff = 13000.0
    tight = ad.cumulative(ad.gaussian_pdf(cutoff, 20))   # near-step constraint
    s = ad.constraint_older([tight])

    assert s[ad.GRID < cutoff - 200].min() > 0.999, "young ages must stay permitted"
    assert s[ad.GRID > cutoff + 200].max() < 0.001, "old ages must be forbidden"

    # Applied to a broad likelihood, the posterior must move YOUNGER, not older.
    like = ad.gaussian_pdf(12500, 1200)
    post = ad._normalise(like * s)
    assert ad.quantiles(post, (0.5,))[0] < ad.quantiles(like, (0.5,))[0]


def test_younger_limiting_date_cuts_the_YOUNG_tail():
    """A date ABOVE the bed forbids ages younger than it — the mirror image."""
    cutoff = 10000.0
    tight = ad.cumulative(ad.gaussian_pdf(cutoff, 20))
    f = ad.constraint_younger([tight])

    assert f[ad.GRID < cutoff - 200].max() < 0.001, "young ages must be forbidden"
    assert f[ad.GRID > cutoff + 200].min() > 0.999, "old ages must stay permitted"

    like = ad.gaussian_pdf(10500, 1200)
    post = ad._normalise(like * f)
    assert ad.quantiles(post, (0.5,))[0] > ad.quantiles(like, (0.5,))[0]


def test_posterior_is_inside_the_bracket_and_no_wider_than_the_event_dates():
    """End to end on the demo data.

    The joint posterior must (a) sit between the limiting ages and (b) be at
    least as tight as the OSL alone — adding true constraints can only ever
    remove probability, never add it.
    """
    df = ad.load_dates("synthetic_ages.csv")
    J = ad.joint_posterior(df)

    post95 = ad.hpd_intervals(J["posterior"], 0.954)
    lo, hi = post95[0][0], post95[-1][1]

    youngest_below = min(ad.quantiles(p, (0.5,))[0]
                         for p in df[df["category"] == "older_limiting"]["pdf"])
    oldest_above = max(ad.quantiles(p, (0.5,))[0]
                       for p in df[df["category"] == "younger_limiting"]["pdf"])
    assert oldest_above < lo, f"posterior reaches younger than the material above it ({lo})"
    assert hi < youngest_below, f"posterior reaches older than the material below it ({hi})"

    ev95 = ad.hpd_intervals(J["event_only"], 0.954)
    assert (hi - lo) <= (ev95[-1][1] - ev95[0][0]) + 1e-9, "constraints widened the answer"


def test_inverted_bracket_is_refused_by_name():
    """Swapping above and below leaves no permitted age; say so, don't divide by zero."""
    df = ad.load_dates("synthetic_ages.csv")
    swap = {"older_limiting": "younger_limiting",
            "younger_limiting": "older_limiting", "event": "event"}
    df["category"] = df["category"].map(swap)

    try:
        ad.joint_posterior(df)
    except ValueError as err:
        assert "no permitted age" in str(err), f"unhelpful message: {err}"
    else:
        raise AssertionError("an inverted bracket must not produce a posterior")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall checks passed")
