import matplotlib.pyplot as plt
import numpy as np


def permutation_test(group_a: np.ndarray, group_b: np.ndarray,
                     num_permutations: int = 500, seed: int = 42) -> float:
    # Seeded to ensure result is reproducible
    rng = np.random.default_rng(seed)
    # Observed value of the test statistic
    observed_statistic = np.mean(group_a) - np.mean(group_b)
    # Pool all observations together-  H0 claims the group labels are meaningless
    combined_data = np.concatenate((group_a, group_b))
    # Holds the statistic from each random relabeling
    permuted_statistics = []

    for _ in range(num_permutations):
        # Shuffle the Groups
        rng.shuffle(combined_data)

        # Split into groups of original sizes
        perm_group_a = combined_data[:len(group_a)]
        perm_group_b = combined_data[len(group_a):]

        # Recompute the statistic for this relabeling and store it
        permuted_statistics.append(np.mean(perm_group_a) - np.mean(perm_group_b))

    # p-value: Fraction of shuffles strictly more extreme than the observed one.
    return np.sum(permuted_statistics > observed_statistic) / num_permutations


#   Builds the null-distribution statistics (same logic as permutation_test).
def _permuted_statistics(group_a, group_b, num_permutations, seed):
    rng = np.random.default_rng(seed)
    combined_data = np.concatenate((group_a, group_b))
    stats = []
    for _ in range(num_permutations):
        rng.shuffle(combined_data)
        stats.append(np.mean(combined_data[:len(group_a)]) - np.mean(combined_data[len(group_a):]))
    return np.array(stats)


# Plots the distribution
def dotplot_blue(group_a: np.ndarray, group_b: np.ndarray,
                 num_permutations: int = 500, seed: int = 42, n_bins: int = 45,
                 finalize: bool = True):
    diffs = _permuted_statistics(group_a, group_b, num_permutations, seed)

    edges = np.linspace(diffs.min(), diffs.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(diffs, edges) - 1, 0, n_bins - 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    stack = np.zeros(n_bins, int)
    positions = []
    for d, b in sorted(zip(diffs, idx), key=lambda z: z[0]):
        y = stack[b] + 1
        ax.plot(centers[b], y, "o", ms=5.5, color="#7e92b8",
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        positions.append((centers[b], y, d))
        stack[b] += 1

    ax.set_xlabel("difference in means under H0  (thousand L)")
    ax.set_title(f"Permutation null distribution — each dot = one of {num_permutations} shuffles")
    ax.set_yticks([])
    ax.set_ylim(0, stack.max() + 2)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    if finalize:
        plt.tight_layout()
        plt.savefig("dotplot_blue.png", dpi=150, facecolor="white")
        plt.show()
        return None
    return fig, ax, positions, stack, diffs


def dotplot_onesided(group_a: np.ndarray, group_b: np.ndarray,
                     num_permutations: int = 500, seed: int = 42, n_bins: int = 45):
    """One-sided plot built ON TOP of dotplot_blue: recolors the right-tail dots red,
    adds the observed line and the p-value annotation."""
    # draw the blue base, but don't save/show it yet
    fig, ax, positions, stack, diffs = dotplot_blue(
        group_a, group_b, num_permutations, seed, n_bins, finalize=False)

    t_obs = np.mean(group_a) - np.mean(group_b)
    p = np.mean(diffs > t_obs)
    n_tail = int((diffs > t_obs).sum())

    # overlay red dots on top of the blue ones in the tail (same positions -> covers them)
    for x, y, d in positions:
        if d > t_obs:
            ax.plot(x, y, "o", ms=5.5, color="#e0524d",
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)

    ax.axvline(t_obs, color="#1a1a2e", lw=2, zorder=2)
    ax.text(t_obs + 0.03, stack.max() * 0.97,
            f"observed = {t_obs:.2f}", fontweight="bold", va="top", fontsize=11)
    ax.text(t_obs + 0.03, stack.max() * 0.80,
            f"{n_tail} of {num_permutations} dots > observed\n→ one-sided p ≈ {p:.3f}",
            va="top", fontsize=10.5, color="#b1352f")

    plt.tight_layout()
    plt.savefig("dotplot_onesided.png", dpi=150, facecolor="white")
    plt.show()


# Our Example Case:
group_a = np.array([12.8, 13.6, 14.0, 14.2, 14.6, 15.0, 15.2, 15.8, 16.4, 17.2])
group_b = np.array([11.8, 12.6, 12.8, 13.2, 13.6, 13.8, 14.2, 14.4, 15.0, 15.6])

p_value = permutation_test(group_a, group_b)
print("p-value:", p_value)

dotplot_blue(group_a, group_b)
dotplot_onesided(group_a, group_b)