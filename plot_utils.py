from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import MaxNLocator

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
                              pad_frac=0.07,  # cap上方的额外padding（占y轴范围比例）
                              arm_frac=0.07,   # 每条竖直脚再往上抬的高度（比例）
                              text="*", lw=0.5, fontsize=7):
    """
    在 (x_left, y_cap_left) 和 (x_right, y_cap_right) 之上分别延伸不同长度的竖线，
    顶部用一条水平线连接，并在中点标注 text。
    pad_frac / arm_frac 按 y 轴范围的比例控制。
    """
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    # 左右脚底部放在各自cap稍上方（pad），再各自抬高一段（arm）
    yL_foot = y_cap_left  + pad_frac * yr
    yR_foot = y_cap_right + pad_frac * yr
    yL_top  = yL_foot + arm_frac * yr
    yR_top  = yR_foot + arm_frac * yr
    y_top = max(yL_top, yR_top)

    # 顶部水平线放在较高的那个顶点处
    y_top = max(yL_top, yR_top)

    # 画两条竖线（不等长）
    ax.plot([x_left,  x_left],  [yL_foot, y_top], color="black", lw=lw, clip_on=False)
    ax.plot([x_right, x_right], [yR_foot, y_top], color="black", lw=lw, clip_on=False)

    # 顶部水平线
    ax.plot([x_left, x_right], [y_top,  y_top],   color="black", lw=lw, clip_on=False)

    # 文本放在水平线正上方一点
    ax.text((x_left + x_right)/2, y_top + 0.006*yr, text,
            ha='center', va='bottom', fontsize=fontsize)

def _paired_test(x, y, test="wilcoxon", alternative="two-sided"):
    """
    x, y: 1D arrays of paired values (same length), NaN 已在外部对齐清理
    test: 'wilcoxon' | 'ttest_rel' | 'sign'
    alternative: 'two-sided'|'greater'|'less'（wilcoxon 与 sign 支持；ttest_rel 始终双侧）
    """
    x = np.asarray(x); y = np.asarray(y)
    # 仅保留两列都为非 NaN 的行 —— 配对对齐
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {"stat": np.nan, "p": np.nan, "n": 0, "test": test}

    if test == "wilcoxon":
        # 注意：wilcoxon 要求非零差；全零差时返回 n.s.
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
        # 符号检验：忽略零差，只看 x>y 的个数
        diff = x - y
        pos = np.sum(diff > 0)
        neg = np.sum(diff < 0)
        n_eff = pos + neg
        if n_eff == 0:
            return {"stat": 0.0, "p": 1.0, "n": len(x), "test": "sign(all zero diffs)"}
        # 双侧/单侧 p 值
        # 双侧：2 * min(P[X>=pos], P[X<=pos]) with X~Binom(n_eff, 0.5)
        # 单侧：greater: P[X>=pos], less: P[X<=pos]
        if alternative == "greater":
            p = stats.binomtest(pos, n_eff, 0.5, alternative="greater").pvalue
        elif alternative == "less":
            p = stats.binomtest(pos, n_eff, 0.5, alternative="less").pvalue
        else:
            p = stats.binomtest(pos, n_eff, 0.5, alternative="two-sided").pvalue
        # 用 pos-neg 作为一个“方向性统计量”展示
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
    group_width: float = 0.55,   # 组内总宽度（方法们挤在这块里）
    group_gap: float = 0.90,     # 组间距离（相邻 metric 的中心间隔）
    colors = None,
    xlabel = '',
    ylabel = '',
    show_legend = False,
    show_xticks = False,
    rotate=0,
    ymin=None,
    ymax=None,
    title=None,
    # === 新增：显著性检验相关 ===
    pairs_per_group: list[tuple[str, str]] | None = None,  # 与 metrics 同顺序的一串 (methodA, methodB)
    test: str = "wilcoxon",
    alternative: str = "two-sided",
    pad_frac: float = 0.07,   # 与 _annotate_pair_asymmetric 一致
    arm_frac: float = 0.07,
    anno_lw: float = 0.5,
    anno_fontsize: int = 7,
    # 自定义如何把 p 转成文本；默认用星号
    p_to_text = None,
    # 计算箱线图“cap高度”的策略（Tukey whisker）
    whisker_iqr: float = 1.5,
    log_panel: str | None = None,
):
    """
    df_long: 长表数据，至少包含 ['Method','Metric','Value']。
    pairs_per_group: 与 metrics 一一对应的配对 (left_method, right_method)，每组做一次检验与标注。
                     如果为 None 或长度与 metrics 不一致，则不做显著性标注。
    test / alternative: 传给 _paired_test 的参数。
    pad_frac / arm_frac / anno_*: 控制不等长支架绘制。
    p_to_text: Callable[[float], str]，将 p 值转成文本；默认使用 _p_to_stars。
    whisker_iqr: 箱线图“胡须”长度的 IQR 倍数（Tukey 1.5*IQR 规则）。
    返回：一个 list[dict]，每个元素记录该组检验的结果（metric, left, right, stat, p, n, test, text）。
    """
    assert {'Method','Metric','Value'}.issubset(df_long.columns), \
        "df_long 必须至少包含列: Method, Metric, Value"

    if ax is None:
        ax = plt.gca()

    if p_to_text is None:
        p_to_text = _p_to_stars  # 用你的 helper

    ax.clear()
    ax.set_aspect('auto')

    if methods is None:
        methods = list(df_long['Method'].dropna().unique())
    if metrics is None:
        metrics = list(df_long['Metric'].dropna().unique())

    if colors is None:
        colors = sns.color_palette("Set2", n_colors=len(methods))

    # 颜色
    palette = {m: colors[i] for i, m in enumerate(methods)}

    # 每个 metric 组的中心位置
    centers = {metric: i * (group_width + group_gap) for i, metric in enumerate(metrics)}

    # 组内每个 method 的偏移（等距分布在 group_width 里）
    k = len(methods)
    if k == 1:
        offsets = {methods[0]: 0.0}
    else:
        inner_positions = np.linspace(-group_width/2, group_width/2, k)
        offsets = {m: inner_positions[i] for i, m in enumerate(methods)}

    # 便捷函数：按 Tukey 规则计算“上胡须 cap”高度（不画点时 cap ≈ whisker 顶）
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
    # 顺手把每个 (method, metric) 的 cap_top 算好，后面标注要用
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
                [values],  # 必须包 list
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

    # —— points（显式计算每个点的 x 坐标来代替 seaborn 的 dodge） —— #
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
            s=point_size**2,  # s 是面积
            alpha=0.6,
            linewidths=0,
            color=palette[m],
            label=m
        )

    # 轴与样式
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

    # === 显著性（逐 group 一对） ===
    test_records = []
    if pairs_per_group is not None and len(pairs_per_group) == len(metrics):
        # 选配对对齐的索引列（用于“配对”对齐）。优先 ['Name','Chr']，再退化到 ['Name']，再退化到“行号对齐”。
        if {'Name','Chr'}.issubset(df_long.columns):
            idx_cols = ['Name', 'Chr']
        elif 'Name' in df_long.columns:
            idx_cols = ['Name']
        else:
            idx_cols = None  # 退化策略见下

        for metric, (left_m, right_m) in zip(metrics, pairs_per_group):
            # 只在该 metric 下取两种方法的数据并做“配对对齐”
            sub = df_long[df_long['Metric'] == metric].copy()
            need = {left_m, right_m}
            if not need.issubset(set(sub['Method'].unique())):
                # 该组缺少其中一个方法，跳过
                continue

            if idx_cols is not None:
                pvt = sub.pivot_table(index=idx_cols, columns='Method', values='Value', aggfunc='first')
                if (left_m not in pvt.columns) or (right_m not in pvt.columns):
                    continue
                x = pvt[left_m].values
                y = pvt[right_m].values
            else:
                # 退化：按每个方法在该 metric 内的出现顺序对齐（可能有风险，但聊胜于无）
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

            # 依据箱线图 cap 顶部来放置不等长支架
            capL = cap_tops.get((metric, left_m), np.nan)
            capR = cap_tops.get((metric, right_m), np.nan)
            if not (np.isfinite(capL) and np.isfinite(capR)):
                # 如果某一侧没有 cap（该侧没有数据），就跳过标注
                continue

            xL = centers[metric] + offsets[left_m]
            xR = centers[metric] + offsets[right_m]

            # 先确保 ylim 足够高（避免被裁剪）
            ymin_now, ymax_now = ax.get_ylim()
            yr = ymax_now - ymin_now
            # 预计 y_top 会再往上加一点，提前抬高上限
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
    # === 配对显著性（只比较一对） ===
    sig_pair=None,               # 例如 ("Evo2HiC","HiC-only")
    test="wilcoxon",             # 'wilcoxon'（默认）|'ttest_rel'|'sign'
    alternative="greater",     # 'two-sided'|'greater'|'less'（ttest_rel 总是双侧）
    show_p_text=False,           # 显示 p 数值还是星号
    bracket_lw=0.5,
    log_panel=None,
    log_metric=None,
):
    ax.clear()
    ax.set_aspect('auto')
    if colors is None:
        colors = sns.color_palette("Set2", n_colors=len(methods))

    # 箱线图
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

    # 叠加散点
    sns.stripplot(
        data=df[methods],
        ax=ax,
        jitter=jitter,
        alpha=alpha,
        size=point_size,
        palette=colors,
        edgecolor=None
    )

    # 轴 & 标题
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

    # === 配对显著性（只一对） ===
    if sig_pair is not None:
        a, b = sig_pair
        if a in methods and b in methods:
            i, j = methods.index(a), methods.index(b)

            # 成对对齐
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
                cap = bp["caps"][2*idx + 1]  # 上端 cap
                return float(np.max(cap.get_ydata()))

            # … 前面完成检验后 …
            # 获取两组箱线图各自 cap 顶部 y 值
            y_cap_i = _upper_cap_y(bp, i)
            y_cap_j = _upper_cap_y(bp, j)

            # 先确保 ylim 足够（stripplot 后再取）
            ymin0, ymax0 = ax.get_ylim()

            # 用不等脚支架：两脚分别贴各自 cap 上方
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

            # 如有需要，抬高 y 上界，避免文本被裁剪
            ymin_after, ymax_after = ax.get_ylim()
            yr = ymax_after - ymin_after
            target_top = max(y_cap_i, y_cap_j) + (0.015 + 0.02 + 0.01) * yr
            if ymax_after < target_top:
                ax.set_ylim(ymin_after, target_top)
    
    if yticks_num is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=yticks_num))

    return ax
