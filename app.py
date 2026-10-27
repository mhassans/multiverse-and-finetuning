import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="S(M)", layout="wide")

st.title("S(M): sum over all possible numbers of actual worlds")
st.latex(r"S(M)=\sum_{i=1}^{M}P(E\mid C,i)\,P(C,i)")
st.caption(
    "i is summed out. The number of life-sustaining potential worlds m "
    "is held fixed while M varies."
)

with st.sidebar:
    st.header("Worlds")

    m = int(st.number_input(
        "Life-sustaining potential worlds: m",
        min_value=0,
        max_value=100_000,
        value=5,
        step=1,
    ))

    requested_min = int(st.number_input(
        "Minimum M",
        min_value=1,
        max_value=100_000,
        value=5,
        step=1,
    ))

    M_max = int(st.number_input(
        "Maximum M",
        min_value=1,
        max_value=100_000,
        value=1000,
        step=1,
    ))

    st.header("Joint prior parameters")

    p_c = st.slider(
        "P(C)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    rate = st.slider(
        "Exponential decay rate: λ",
        min_value=0.001,
        max_value=2.0,
        value=0.1,
        step=0.001,
    )

    alpha = st.slider(
        "Power-law exponent: α",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )

    st.header("Display")

    log_x = st.checkbox("Logarithmic M axis", value=False)
    log_y = st.checkbox("Logarithmic S(M) axis", value=False)

    max_points = st.slider(
        "Maximum plotted M values",
        min_value=50,
        max_value=1000,
        value=400,
        step=50,
    )

M_min = max(1, requested_min, m)

if M_max < M_min:
    st.error(f"Maximum M must be at least {M_min}.")
    st.stop()

if M_min != requested_min:
    st.info(f"The plotted range starts at M = {M_min}, ensuring M ≥ m.")


def life_probabilities(M, m):
    """Return P(E | C, i) for every i = 1, ..., M."""
    if m == 0:
        return np.zeros(M, dtype=float)

    probabilities = np.ones(M, dtype=float)
    n = M - m

    if n > 0:
        j = np.arange(n, dtype=float)
        log_no_life = np.cumsum(np.log1p(-m / (M - j)))
        probabilities[:n] = -np.expm1(log_no_life)

    return np.clip(probabilities, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def calculate_curves(m, M_min, M_max, p_c, rate, alpha, max_points):
    k = np.arange(1, M_max + 1, dtype=float)

    weights = {
        "Constant across i": np.ones_like(k),
        "Reciprocal: 1/i": 1.0 / k,
        f"Exponential: λ={rate:g}": np.exp(-rate * (k - 1)),
        f"Power law: α={alpha:g}": k ** (-alpha),
    }

    normalisers = {
        name: np.cumsum(w)
        for name, w in weights.items()
    }

    n_points = min(max_points, M_max - M_min + 1)
    M_values = np.unique(
        np.rint(np.linspace(M_min, M_max, n_points)).astype(int)
    )

    curves = {
        name: np.empty(len(M_values), dtype=float)
        for name in weights
    }

    for position, M in enumerate(M_values):
        p_e = life_probabilities(M, m)

        for name, w in weights.items():
            # Joint prior: sum over i of P(C, i) equals P(C).
            p_ci = p_c * w[:M] / normalisers[name][M - 1]

            # S(M) = sum over i of P(E | C, i) * P(C, i).
            curves[name][position] = np.sum(p_e * p_ci)

    return pd.DataFrame({"M": M_values, **curves})


with st.spinner("Calculating S(M)..."):
    results = calculate_curves(
        m, M_min, M_max, p_c, rate, alpha, max_points
    )

fig, ax = plt.subplots(figsize=(11, 6))

has_positive_values = False

for name in results.columns[1:]:
    values = results[name].to_numpy()
    has_positive_values |= bool(np.any(values > 0))

    displayed = (
        np.where(values > 0, values, np.nan)
        if log_y
        else values
    )

    ax.plot(
        results["M"],
        displayed,
        label=name,
        linewidth=2,
        marker="o" if len(results) == 1 else None,
    )

ax.set_xlabel("M: number of potential worlds")
ax.set_ylabel(r"$S(M)=\sum_{i=1}^{M}P(E\mid C,i)\,P(C,i)$")
ax.set_title(f"S(M), m = {m}, P(C) = {p_c:g}")

if log_x:
    ax.set_xscale("log")

if log_y and has_positive_values:
    ax.set_yscale("log")
else:
    ax.set_ylim(0, max(0.01, p_c * 1.02))
    if log_y:
        st.warning(
            "All values are zero. Showing a linear S(M) axis instead."
        )

ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Each curve sums over i = 1, ..., M. "
    "The joint priors are normalised separately at each M, "
    "so their sum equals P(C). No division by P(C) is applied to S(M)."
)

with st.expander("Model definitions"):
    st.latex(
        r"P(E\mid C,i)="
        r"1-\prod_{j=0}^{i-1}\left(1-\frac{m}{M-j}\right)"
    )
    st.latex(
        r"P(C,i)=P(C)\frac{w(i)}{\sum_{k=1}^{M}w(k)}"
    )
    st.latex(
        r"w(i)\in\left\{1,\;\frac{1}{i},\;"
        r"e^{-\lambda(i-1)},\;\frac{1}{i^\alpha}\right\}"
    )
    st.write(
        "The model assumes at least one actual world. "
        "'Constant across i' means P(C, i) = P(C)/M at each M."
    )

with st.expander("Show plotted values"):
    st.dataframe(results, use_container_width=True)

st.download_button(
    "Download plotted data as CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="S_M_curves.csv",
    mime="text/csv",
)