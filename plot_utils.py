from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import MaxNLocator

colors = ['#fb8072', '#bebada', '#80b1d3', '#8dd3c7', '#fddaec']   # method palette (was plot_settings.py)

_TEST_LOG = []


def _add_log1p_cbar(fig, im, ax_or_axes, label="contacts (raw count)",
                    ticks_raw=(0, 5, 25, 100), fraction=0.04, pad=0.02):
    """Attach a color bar to one or more axes, with tick labels in raw-count
    space (post-`np.expm1`) instead of `log1p` space. Used by all Hi-C
    heatmaps that plot `np.log1p(matrix)` with `vmin=log1p(0)` /
    `vmax=log1p(100)` (Fig 3, Fig 6). Reviewer-asked
    color-bar addition (T_Mp2).
    """
    cb = fig.colorbar(im, ax=ax_or_axes, fraction=fraction, pad=pad)
    cb.set_ticks([np.log1p(t) for t in ticks_raw])
    cb.set_ticklabels([str(t) for t in ticks_raw])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2, labelsize=6)
    cb.set_label(label, fontsize=7)
    return cb

def clear_test_log():
    _TEST_LOG.clear()

def dump_test_log(path=None):
    df = pd.DataFrame(_TEST_LOG)
    if path is not None:
        df.to_csv(path, sep='\t', index=False)
    return df

def _p_to_stars(p):
    return "****" if p < 1e-4 else \
           "***"  if p < 1e-3 else \
           "**"   if p < 1e-2 else \
           "*"    if p < 5e-2 else "n.s."

def _annotate_pair_asymmetric(ax, x_left, x_right, y_cap_left, y_cap_right,
                              pad_frac=0.07,  # extra padding above the cap (fraction of the y range)
                              arm_frac=0.07,   # extra height of each vertical leg (fraction)
                              text="*", lw=0.5, fontsize=7):
    """
    Draw vertical legs of different lengths above (x_left, y_cap_left) and (x_right, y_cap_right),
    join them with a horizontal line at the top and put `text` above its midpoint.
    pad_frac / arm_frac are fractions of the y range.
    """
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    # each leg starts slightly above its own cap (pad) and rises by arm
    yL_foot = y_cap_left  + pad_frac * yr
    yR_foot = y_cap_right + pad_frac * yr
    yL_top  = yL_foot + arm_frac * yr
    yR_top  = yR_foot + arm_frac * yr
    y_top = max(yL_top, yR_top)

    # horizontal line at the higher of the two leg tops
    y_top = max(yL_top, yR_top)

    # two vertical legs (different lengths)
    ax.plot([x_left,  x_left],  [yL_foot, y_top], color="black", lw=lw, clip_on=False)
    ax.plot([x_right, x_right], [yR_foot, y_top], color="black", lw=lw, clip_on=False)

    # horizontal line
    ax.plot([x_left, x_right], [y_top,  y_top],   color="black", lw=lw, clip_on=False)

    # text slightly above the horizontal line
    ax.text((x_left + x_right)/2, y_top + 0.006*yr, text,
            ha='center', va='bottom', fontsize=fontsize)

def _paired_test(x, y, test="wilcoxon", alternative="two-sided"):
    """
    x, y: 1D arrays of paired values (same length), NaN already removed in pairs by the caller
    test: 'wilcoxon' | 'ttest_rel' | 'sign'
    alternative: 'two-sided'|'greater'|'less' (wilcoxon and sign; ttest_rel is always two-sided)
    """
    x = np.asarray(x); y = np.asarray(y)
    # keep rows where both values are finite (paired alignment)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {"stat": np.nan, "p": np.nan, "n": 0, "test": test}

    if test == "wilcoxon":
        # wilcoxon needs non-zero differences; all-zero differences return n.s.
        diff = x - y
        nonzero = diff != 0
        if nonzero.sum() == 0:
            return {"stat": 0.0, "p": 1.0, "n": len(x), "test": "wilcoxon(all zero diffs)"}
        stat, p = stats.wilcoxon(x[nonzero], y[nonzero], alternative=alternative, zero_method="wilcox")
        return {"stat": float(stat), "p": float(p), "n": int(len(x)), "test": "wilcoxon"}

    elif test == "ttest_rel":
        t, p = stats.ttest_rel(x, y, nan_policy="omit")
        return {"stat": float(t), "p": float(p), "n": int(len(x)), "test": "ttest_rel"}

    elif test == "sign":
        # sign test: ignore ties, count x > y
        diff = x - y
        pos = np.sum(diff > 0)
        neg = np.sum(diff < 0)
        n_eff = pos + neg
        if n_eff == 0:
            return {"stat": 0.0, "p": 1.0, "n": len(x), "test": "sign(all zero diffs)"}
        # two-sided / one-sided p-value
        # two-sided: 2 * min(P[X>=pos], P[X<=pos]) with X~Binom(n_eff, 0.5)
        # one-sided: greater: P[X>=pos], less: P[X<=pos]
        if alternative == "greater":
            p = stats.binomtest(pos, n_eff, 0.5, alternative="greater").pvalue
        elif alternative == "less":
            p = stats.binomtest(pos, n_eff, 0.5, alternative="less").pvalue
        else:
            p = stats.binomtest(pos, n_eff, 0.5, alternative="two-sided").pvalue
        # report pos - neg as a directional statistic
        return {"stat": float(pos - neg), "p": float(p), "n": int(len(x)), "test": "sign"}

    else:
        raise ValueError("test must be 'wilcoxon', 'ttest_rel', or 'sign'")

def plot_grouped_box_with_points(
    df_long: pd.DataFrame,
    methods: list | None = None,
    metrics: list | None = None,
    ax: plt.Axes | None = None,
    box_width: float = 0.18,
    point_size: float = 0.5,
    group_width: float = 0.55,   # total width of one group (methods share it)
    group_gap: float = 0.90,     # distance between group centres (adjacent metrics)
    colors = None,
    xlabel = '',
    ylabel = '',
    show_legend = False,
    show_xticks = False,
    rotate=0,
    ymin=None,
    ymax=None,
    title=None,
    # === significance tests ===
    pairs_per_group: list[tuple[str, str]] | None = None,  # one (methodA, methodB) per metric, same order as metrics
    test: str = "wilcoxon",
    alternative: str = "two-sided",
    pad_frac: float = 0.07,   # same as _annotate_pair_asymmetric
    arm_frac: float = 0.07,
    anno_lw: float = 0.5,
    anno_fontsize: int = 7,
    # how p is turned into text; stars by default
    p_to_text = None,
    # box 'cap' height (Tukey whisker)
    whisker_iqr: float = 1.5,
    log_panel: str | None = None,
):
    """
    df_long: long-format data with at least ['Method','Metric','Value'].
    pairs_per_group: one pair (left_method, right_method) per metric; each group gets one test and annotation.
                     If None or not the same length as metrics, no significance is annotated.
    test / alternative: passed to _paired_test.
    pad_frac / arm_frac / anno_*: control the asymmetric brackets.
    p_to_text: Callable[[float], str] turning p into text; default _p_to_stars.
    whisker_iqr: whisker length in IQR units (Tukey 1.5*IQR rule).
    Returns: list[dict], one record per group test (metric, left, right, stat, p, n, test, text).
    """
    assert {'Method','Metric','Value'}.issubset(df_long.columns), \
        "df_long must contain at least the columns: Method, Metric, Value"

    if ax is None:
        ax = plt.gca()

    if p_to_text is None:
        p_to_text = _p_to_stars  # helper

    ax.clear()
    ax.set_aspect('auto')

    if methods is None:
        methods = list(df_long['Method'].dropna().unique())
    if metrics is None:
        metrics = list(df_long['Metric'].dropna().unique())

    if colors is None:
        colors = sns.color_palette("Set2", n_colors=len(methods))

    # colours
    palette = {m: colors[i] for i, m in enumerate(methods)}

    # centre of each metric group
    centers = {metric: i * (group_width + group_gap) for i, metric in enumerate(metrics)}

    # offset of each method within a group (evenly spaced over group_width)
    k = len(methods)
    if k == 1:
        offsets = {methods[0]: 0.0}
    else:
        inner_positions = np.linspace(-group_width/2, group_width/2, k)
        offsets = {m: inner_positions[i] for i, m in enumerate(methods)}

    # helper: upper whisker cap by the Tukey rule (cap ~ whisker top when no points are drawn)
    def _whisker_top(values: np.ndarray) -> float:
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        high = q3 + whisker_iqr * iqr
        cand = values[values <= high]
        if cand.size == 0:
            return float(q3)
        return float(np.max(cand))

    # —— boxplot —— #
    # cap_top of every (method, metric), used for the annotations
    cap_tops = {}  # (metric, method) -> cap_top
    for i, method in enumerate(methods):
        for metric in metrics:
            values = df_long.loc[
                (df_long['Method'] == method) & (df_long['Metric'] == metric),
                'Value'
            ].dropna().values
            if values.size == 0:
                continue
            x = centers[metric] + offsets[method]
            ax.boxplot(
                [values],  # must be wrapped in a list
                positions=[x],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='none', edgecolor=palette[method], linewidth=0.5),
                medianprops=dict(color=palette[method], linewidth=0.5),
                whiskerprops=dict(color=palette[method], linewidth=0.3),
                capprops=dict(color=palette[method], linewidth=0.3),
            )
            cap_tops[(metric, method)] = _whisker_top(values)

    # -- points (x position of every point computed explicitly instead of seaborn's dodge) -- #
    df_plot = df_long.dropna(subset=['Method', 'Metric', 'Value']).copy()
    df_plot = df_plot[df_plot['Method'].isin(methods) & df_plot['Metric'].isin(metrics)]
    df_plot['__x__'] = [
        centers[mtr] + offsets[mtd]
        for mtd, mtr in zip(df_plot['Method'].values, df_plot['Metric'].values)
    ]

    for m in methods:
        sub = df_plot[df_plot['Method'] == m]
        jitter_strength = 0.05
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(sub))
        ax.scatter(
            sub['__x__'].values + jitter,
            sub['Value'].values,
            s=point_size**2,  # s is an area
            alpha=0.6,
            linewidths=0,
            color=palette[m],
            label=m
        )

    # axes and style
    xticks = [centers[m] for m in metrics]
    if show_xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(metrics, rotation=rotate)
    else:
        ax.set_xticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1
    ax.tick_params(axis='y', length=2)

    left = min(xticks) - (group_width / 2) - 0.5
    right = max(xticks) + (group_width / 2) + 0.5
    ax.set_xlim(left, right)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    sns.despine(ax=ax)

    if show_legend:
        handles = [Line2D([0],[0], marker='o', linestyle='',
                        markerfacecolor=palette[m], markeredgewidth=0, alpha=0.8,
                        label=m) for m in methods]
        ax.legend(handles=handles, title='', frameon=False, loc='best')

    if title is not None:
        ax.set_title(title)

    # === significance (one pair per group) ===
    test_records = []
    if pairs_per_group is not None and len(pairs_per_group) == len(metrics):
        # index columns for paired alignment: ['Name','Chr'] if present, else ['Name'], else row order
        if {'Name','Chr'}.issubset(df_long.columns):
            idx_cols = ['Name', 'Chr']
        elif 'Name' in df_long.columns:
            idx_cols = ['Name']
        else:
            idx_cols = None  # fallback below

        for metric, (left_m, right_m) in zip(metrics, pairs_per_group):
            # data of the two methods for this metric, aligned in pairs
            sub = df_long[df_long['Metric'] == metric].copy()
            need = {left_m, right_m}
            if not need.issubset(set(sub['Method'].unique())):
                # one of the two methods is missing in this group: skip
                continue

            if idx_cols is not None:
                pvt = sub.pivot_table(index=idx_cols, columns='Method', values='Value', aggfunc='first')
                if (left_m not in pvt.columns) or (right_m not in pvt.columns):
                    continue
                x = pvt[left_m].values
                y = pvt[right_m].values
            else:
                # fallback: align by order of appearance within the metric (best effort)
                sub_l = sub[sub['Method'] == left_m].reset_index(drop=True)
                sub_r = sub[sub['Method'] == right_m].reset_index(drop=True)
                n = min(len(sub_l), len(sub_r))
                x = sub_l['Value'].values[:n]
                y = sub_r['Value'].values[:n]

            res = _paired_test(x, y, test=test, alternative=alternative)
            ptext = p_to_text(res["p"]) if np.isfinite(res["p"]) else "n.s."
            record = {
                "metric": metric,
                "left": left_m,
                "right": right_m,
                **res,
                "text": ptext,
            }
            test_records.append(record)
            if log_panel is not None:
                _TEST_LOG.append({"panel": log_panel, **record})

            # place the asymmetric bracket on the box caps
            capL = cap_tops.get((metric, left_m), np.nan)
            capR = cap_tops.get((metric, right_m), np.nan)
            if not (np.isfinite(capL) and np.isfinite(capR)):
                # skip if one side has no cap (no data)
                continue

            xL = centers[metric] + offsets[left_m]
            xR = centers[metric] + offsets[right_m]

            # make sure ylim is high enough (avoid clipping)
            ymin_now, ymax_now = ax.get_ylim()
            yr = ymax_now - ymin_now
            # y_top will rise a little more; raise the upper limit in advance
            y_top_expect = max(capL, capR) + (pad_frac + arm_frac + 0.03) * yr
            if y_top_expect > ymax_now:
                ax.set_ylim(top=y_top_expect)

            _annotate_pair_asymmetric(
                ax,
                x_left=xL,
                x_right=xR,
                y_cap_left=capL,
                y_cap_right=capR,
                pad_frac=pad_frac,
                arm_frac=arm_frac,
                text=ptext,
                lw=anno_lw,
                fontsize=anno_fontsize
            )
        

    return test_records
def plot_box_with_points(
    ax, df, methods, 
    colors=None, 
    title=None, 
    jitter=True, 
    alpha=0.8, 
    point_size=0.1,
    box_width=0.5,
    whis=1.5,
    line_width_box=0.5,
    line_width_whisker=0.3,
    line_width_median=0.5,
    show_methods=False,
    xlabel='',
    ylabel='',
    ymin=None,
    ymax=None,
    yticks_num = None,
    # === paired significance (one pair only) ===
    sig_pair=None,               # e.g. ("Evo2HiC","HiC-only")
    test="wilcoxon",             # 'wilcoxon' (default)|'ttest_rel'|'sign'
    alternative="greater",     # 'two-sided'|'greater'|'less' (ttest_rel is always two-sided)
    show_p_text=False,           # show the p-value instead of stars
    bracket_lw=0.5,
    log_panel=None,
    log_metric=None,
):
    ax.clear()
    ax.set_aspect('auto')
    if colors is None:
        colors = sns.color_palette("Set2", n_colors=len(methods))

    # box plot
    data_list = [df[m].dropna().values for m in methods]
    x_positions = np.arange(len(methods), dtype=float)

    bp = ax.boxplot(
        data_list,
        positions=x_positions,
        widths=box_width,
        vert=True,
        patch_artist=False,
        showfliers=False,
        whis=whis,
        manage_ticks=False
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_color(colors[i]); box.set_linewidth(line_width_box)
    for i in range(len(methods)):
        for w in bp["whiskers"][2*i:2*i+2]:
            w.set_color(colors[i]); w.set_linewidth(line_width_whisker)
        for c in bp["caps"][2*i:2*i+2]:
            c.set_color(colors[i]); c.set_linewidth(line_width_whisker)
    for i, med in enumerate(bp["medians"]):
        med.set_color(colors[i]); med.set_linewidth(line_width_median)

    # overlay points
    sns.stripplot(
        data=df[methods],
        ax=ax,
        jitter=jitter,
        alpha=alpha,
        size=point_size,
        palette=colors,
        edgecolor=None
    )

    # axes and title
    if show_methods:
        ax.set_xticks(x_positions + 0.3)
        ax.set_xticklabels(methods, rotation=30, ha='right')
    else:
        ax.set_xticks([])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if ymin is not None: ax.set_ylim(bottom=ymin)
    if ymax is not None: ax.set_ylim(top=ymax)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', length=2)

    # === paired significance (one pair only) ===
    if sig_pair is not None:
        a, b = sig_pair
        if a in methods and b in methods:
            i, j = methods.index(a), methods.index(b)

            # paired alignment
            xv = df[a].values
            yv = df[b].values
            mask = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[mask], yv[mask]

            res = _paired_test(xv, yv, test=test, alternative=alternative)
            p = res["p"]
            if log_panel is not None:
                _TEST_LOG.append({
                    "panel": log_panel,
                    "metric": log_metric,
                    "left": a,
                    "right": b,
                    **res,
                    "text": (f"p={p:.2e}" if show_p_text else _p_to_stars(p)),
                })

            def _upper_cap_y(bp, idx):
                cap = bp["caps"][2*idx + 1]  # upper cap
                return float(np.max(cap.get_ydata()))

            # ... after the test ...
            # y of the top cap of each of the two boxes
            y_cap_i = _upper_cap_y(bp, i)
            y_cap_j = _upper_cap_y(bp, j)

            # make sure ylim is high enough (after the stripplot)
            ymin0, ymax0 = ax.get_ylim()

            # asymmetric bracket: each leg sits on its own cap
            _annotate_pair_asymmetric(
                ax,
                x_left=x_positions[min(i, j)],
                x_right=x_positions[max(i, j)],
                y_cap_left=y_cap_i if i < j else y_cap_j,
                y_cap_right=y_cap_j if i < j else y_cap_i,
                text=(f"p={p:.2e}" if show_p_text else _p_to_stars(p)),
                lw=bracket_lw,
                fontsize=7
            )

            # raise the upper y limit if needed so the text is not clipped
            ymin_after, ymax_after = ax.get_ylim()
            yr = ymax_after - ymin_after
            target_top = max(y_cap_i, y_cap_j) + (0.015 + 0.02 + 0.01) * yr
            if ymax_after < target_top:
                ax.set_ylim(ymin_after, target_top)
    
    if yticks_num is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=yticks_num))

    return ax
