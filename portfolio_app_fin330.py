import streamlit as st
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Optimal Portfolio Calculator",
    layout="wide",
    page_icon="📈",
)

st.title("📈 Optimal Portfolio Calculator")
st.markdown("**FIN 330 | Dr. Efe Cotelioglu**")
st.markdown("---")

st.sidebar.header("Input Parameters")

st.sidebar.subheader("Asset 1: Bonds")
E_rB = st.sidebar.number_input(
    "Expected Return E(r_B) (%)",
    value=5.0,
    min_value=0.0,
    max_value=50.0,
    step=0.1,
) / 100
sigma_B = st.sidebar.number_input(
    "Standard Deviation σ_B (%)",
    value=8.0,
    min_value=0.1,
    max_value=100.0,
    step=0.1,
) / 100

st.sidebar.subheader("Asset 2: Stocks")
E_rS = st.sidebar.number_input(
    "Expected Return E(r_S) (%)",
    value=10.0,
    min_value=0.0,
    max_value=50.0,
    step=0.1,
) / 100
sigma_S = st.sidebar.number_input(
    "Standard Deviation σ_S (%)",
    value=19.0,
    min_value=0.1,
    max_value=100.0,
    step=0.1,
) / 100

st.sidebar.subheader("Correlation & Risk-Free Rate")
rho_BS = st.sidebar.slider(
    "Correlation ρ_BS",
    min_value=-1.0,
    max_value=1.0,
    value=0.20,
    step=0.01,
)
r_f = st.sidebar.number_input(
    "Risk-Free Rate r_f (%)",
    value=3.0,
    min_value=0.0,
    max_value=20.0,
    step=0.1,
) / 100

st.sidebar.subheader("Investor Risk Aversion")
A = st.sidebar.slider(
    "Risk Aversion Coefficient (A)",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Graph Display Options")
show_opportunity_set = st.sidebar.checkbox("Show Opportunity Set", value=True)
show_efficient_frontier = st.sidebar.checkbox("Show Efficient Frontier", value=False)
show_MVP = st.sidebar.checkbox("Show Minimum Variance Portfolio", value=True)
show_CAL = st.sidebar.checkbox("Show CAL", value=True)
allow_short = st.sidebar.checkbox(
    "Extend Curve Beyond B and S (Short Sales)",
    value=True,
    help="Plots portfolios with negative weights so the full opportunity set, including the lower inefficient branch, is visible.",
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust the parameters to see how they affect the optimal portfolio.")

cov_BS = rho_BS * sigma_B * sigma_S

try:
    # ---------- Minimum Variance Portfolio ----------
    mvp_denominator = sigma_B**2 + sigma_S**2 - 2 * cov_BS
    if abs(mvp_denominator) < 1e-10:
        st.error("❌ Invalid inputs: Minimum variance portfolio is not defined (ρ = 1 and σ_B = σ_S).")
        st.stop()

    w_B_min = (sigma_S**2 - cov_BS) / mvp_denominator
    w_S_min = 1 - w_B_min
    E_rMin = w_B_min * E_rB + w_S_min * E_rS
    sigma_Min = np.sqrt(
        max(
            w_B_min**2 * sigma_B**2
            + w_S_min**2 * sigma_S**2
            + 2 * w_B_min * w_S_min * cov_BS,
            0,
        )
    )
    mvp_short_sale = w_B_min < 0 or w_B_min > 1

    # ---------- Optimal Risky Portfolio ----------
    numerator = (E_rB - r_f) * sigma_S**2 - (E_rS - r_f) * cov_BS
    denominator = (
        (E_rB - r_f) * sigma_S**2
        + (E_rS - r_f) * sigma_B**2
        - (E_rB - r_f + E_rS - r_f) * cov_BS
    )

    if abs(denominator) < 1e-10:
        st.error("❌ Invalid inputs: Cannot calculate optimal portfolio with these parameters.")
        st.stop()

    w_B = numerator / denominator
    w_S = 1 - w_B

    E_rO = w_B * E_rB + w_S * E_rS
    sigma_O_squared = (
        w_B**2 * sigma_B**2
        + w_S**2 * sigma_S**2
        + 2 * w_B * w_S * cov_BS
    )

    if sigma_O_squared < 0:
        st.error("❌ Invalid inputs: Negative variance calculated.")
        st.stop()

    sigma_O = np.sqrt(sigma_O_squared)
    sharpe_ratio = (E_rO - r_f) / sigma_O if sigma_O > 0 else 0
    y_star = (E_rO - r_f) / (A * sigma_O_squared) if sigma_O_squared > 0 else 0

    E_rC = r_f + y_star * (E_rO - r_f)
    sigma_C = abs(y_star) * sigma_O

    alloc_B = y_star * w_B
    alloc_S = y_star * w_S
    alloc_rf = 1 - y_star

    # ---------- Opportunity set ----------
    if allow_short:
        # Extend past both assets so the lower (inefficient) branch is clearly visible
        w_low = min(-0.5, w_B_min - 0.5)
        w_high = max(1.5, w_B_min + 0.5)
    else:
        # No short sales, but always include the MVP
        w_low = min(0.0, w_B_min)
        w_high = max(1.0, w_B_min)
    weights_B = np.linspace(w_low, w_high, 400)
    weights_S = 1 - weights_B
    E_portfolio = weights_B * E_rB + weights_S * E_rS
    sigma_portfolio = np.sqrt(
        np.maximum(
            weights_B**2 * sigma_B**2
            + weights_S**2 * sigma_S**2
            + 2 * weights_B * weights_S * cov_BS,
            0,
        )
    )

    # Efficient frontier: portfolios on the opportunity set with E(r) >= E(r_MVP)
    efficient_mask = E_portfolio >= E_rMin - 1e-12
    inefficient_mask = E_portfolio <= E_rMin + 1e-12

    sigma_CAL = np.linspace(0, max(sigma_S * 1.5, sigma_C * 1.5), 100)
    E_CAL = r_f + sharpe_ratio * sigma_CAL

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Optimal Portfolio Visualization")
        fig, ax = plt.subplots(figsize=(10, 7))

        if show_opportunity_set and show_efficient_frontier:
            # Efficient part solid, inefficient part dashed
            ax.plot(
                sigma_portfolio[inefficient_mask] * 100,
                E_portfolio[inefficient_mask] * 100,
                "b--",
                linewidth=1.8,
                alpha=0.6,
                label="Inefficient Portion",
            )
            ax.plot(
                sigma_portfolio[efficient_mask] * 100,
                E_portfolio[efficient_mask] * 100,
                color="navy",
                linewidth=3.5,
                label="Efficient Frontier",
            )
        elif show_opportunity_set:
            ax.plot(
                sigma_portfolio * 100,
                E_portfolio * 100,
                "b-",
                linewidth=2.5,
                label="Opportunity Set",
            )
        elif show_efficient_frontier:
            ax.plot(
                sigma_portfolio[efficient_mask] * 100,
                E_portfolio[efficient_mask] * 100,
                color="navy",
                linewidth=3.5,
                label="Efficient Frontier",
            )

        if show_CAL:
            ax.plot(
                sigma_CAL * 100,
                E_CAL * 100,
                "r-",
                linewidth=2.5,
                label="CAL",
                zorder=5,
            )

        ax.plot(sigma_B * 100, E_rB * 100, "gs", markersize=14, label="Bond (B)", zorder=10)
        ax.text(
            sigma_B * 100 - 0.8,
            E_rB * 100,
            "B",
            fontsize=11,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        ax.plot(sigma_S * 100, E_rS * 100, "bs", markersize=14, label="Stock (S)", zorder=10)
        ax.text(
            sigma_S * 100 + 0.8,
            E_rS * 100,
            "S",
            fontsize=11,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        if show_MVP:
            ax.plot(
                sigma_Min * 100,
                E_rMin * 100,
                "D",
                color="darkorange",
                markersize=12,
                label="Minimum Variance Portfolio",
                zorder=11,
            )
            ax.text(
                sigma_Min * 100 - 0.8,
                E_rMin * 100,
                "MVP",
                fontsize=11,
                ha="right",
                va="center",
                bbox=dict(boxstyle="round", facecolor="moccasin", alpha=0.8),
            )

        ax.plot(0, r_f * 100, "ro", markersize=10, zorder=10)
        ax.text(-0.5, r_f * 100, r"$r_f$", fontsize=11, ha="right", va="center")

        if show_CAL:
            ax.plot(sigma_O * 100, E_rO * 100, "ko", markersize=12, zorder=10)
            ax.text(
                sigma_O * 100 + 0.5,
                E_rO * 100 - 0.3,
                "P",
                fontsize=11,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
            )

            ax.plot(sigma_C * 100, E_rC * 100, "mo", markersize=12, zorder=10)
            ax.text(
                sigma_C * 100 - 0.5,
                E_rC * 100,
                "C",
                fontsize=11,
                ha="right",
                va="center",
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
            )

        ax.set_xlabel("Standard Deviation (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Expected Return (%)", fontsize=12, fontweight="bold")
        ax.set_title("Opportunity Set, Efficient Frontier and Optimal Portfolio", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="lower right")

        max_sigma = max(sigma_S * 100, sigma_B * 100, sigma_C * 100, sigma_O * 100) * 1.2
        max_return = max(E_rS * 100, E_rB * 100, E_rC * 100, E_rO * 100, E_rMin * 100) * 1.15
        lowest = min(E_rB, E_rS, E_rMin)
        min_return = lowest * 100 * 1.1 if lowest < 0 else 0

        ax.set_xlim(0, max_sigma)
        ax.set_ylim(min_return, max_return)

        st.pyplot(fig)

    with col2:
        st.subheader("Results")

        st.markdown("### Minimum Variance Portfolio (MVP)")
        results_MVP = pd.DataFrame(
            {
                "Metric": [
                    "Weight in Bonds",
                    "Weight in Stocks",
                    "Expected Return",
                    "Standard Deviation",
                ],
                "Value": [
                    f"{w_B_min*100:.2f}%",
                    f"{w_S_min*100:.2f}%",
                    f"{E_rMin*100:.2f}%",
                    f"{sigma_Min*100:.2f}%",
                ],
            }
        )
        st.dataframe(results_MVP, hide_index=True, use_container_width=True)
        if mvp_short_sale:
            st.warning("⚠️ The MVP requires short selling one of the assets with these inputs.")

        st.markdown("---")
        st.markdown("### Optimal Risky Portfolio (P)")
        results_P = pd.DataFrame(
            {
                "Metric": [
                    "Weight in Bonds",
                    "Weight in Stocks",
                    "Expected Return",
                    "Standard Deviation",
                    "Sharpe Ratio",
                ],
                "Value": [
                    f"{w_B*100:.2f}%",
                    f"{w_S*100:.2f}%",
                    f"{E_rO*100:.2f}%",
                    f"{sigma_O*100:.2f}%",
                    f"{sharpe_ratio:.4f}",
                ],
            }
        )
        st.dataframe(results_P, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### Optimal Complete Portfolio (A = {A:.1f})")
        position_type = (
            "Leveraged (Borrowing)"
            if y_star > 1
            else "Conservative (Lending)"
            if y_star < 1
            else "100% in Risky Portfolio"
        )

        results_C = pd.DataFrame(
            {
                "Metric": [
                    "Allocation to P (y*)",
                    "Position Type",
                    "Expected Return",
                    "Standard Deviation",
                ],
                "Value": [
                    f"{y_star*100:.2f}%",
                    position_type,
                    f"{E_rC*100:.2f}%",
                    f"{sigma_C*100:.2f}%",
                ],
            }
        )
        st.dataframe(results_C, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### Final Asset Allocation")
        final_alloc = pd.DataFrame(
            {
                "Asset": ["Bonds", "Stocks", "Risk-Free"],
                "Allocation": [
                    f"{alloc_B*100:.2f}%",
                    f"{alloc_S*100:.2f}%",
                    f"{alloc_rf*100:.2f}%",
                ],
            }
        )
        st.dataframe(final_alloc, hide_index=True, use_container_width=True)

        if y_star > 1:
            st.warning(
                f"⚠️ This investor is leveraged, borrowing {abs(alloc_rf)*100:.2f}% to invest more in the risky portfolio."
            )
        elif y_star < 1:
            st.success(f"✅ This investor is conservative, holding {alloc_rf*100:.2f}% in risk-free assets.")
        else:
            st.info("ℹ️ This investor is fully invested in the risky portfolio with no leverage or risk-free holdings.")

except Exception as e:
    st.error(f"❌ Error in calculations: {str(e)}")
    st.stop()

st.markdown("---")
st.markdown(
    """
Built for FIN 330
American University of Sharjah | Dr. Efe Cotelioglu
"""
)
