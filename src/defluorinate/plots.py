"""Plots generated only from a completed local comparison."""

from .io import read


def plot_comparison(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read(folder / "comparison.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, layout="constrained")
    for ax, domain in zip(
        axes, ["ideal_simulator", "synthetic_noise_simulator", "readout_only_simulator"]
    ):
        for arm in ["filter", "weighted_recovery", "known_readout_inverse"]:
            selected = [
                r
                for r in rows
                if r["domain"] == domain
                and r["arm"] == arm
                and r["status"] == "completed"
            ]
            if selected:
                ax.plot(
                    [r["dimension"] for r in selected],
                    [r["error_mEh"] for r in selected],
                    "-o",
                    label=arm.replace("_", " "),
                )
        baseline = {
            r["dimension"]: r["classical_error_mEh"]
            for r in rows
            if r["domain"] == domain and r["status"] == "completed"
        }
        ax.plot(
            list(baseline),
            list(baseline.values()),
            "--",
            color="black",
            label="classical",
        )
        ax.set(
            title=domain.replace("_simulator", "").replace("_", " "),
            xlabel="Selected dimension",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Error relative to same-model FCI (mEh)")
    fig.savefig(folder / "comparison.png", dpi=180)
    fig.savefig(folder / "comparison.svg")
    plt.close(fig)
