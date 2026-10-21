"""Charts + static HTML report + summary.json for a run (consumed by the demo app and the PPT)."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constraints import KIND_ZH  # noqa: E402
from .optimizer import OP_ZH  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "violet": "#6250d6", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
SRC_COLORS = {"seed": C["blue"], "math_seed": C["orange"], "rule": C["aqua"], "self_if": C["yellow"],
              "self_math": C["magenta"], "self_seed": C["violet"]}
SRC_ZH = {"seed": "种子指令(外部)", "math_seed": "数学金标", "rule": "规则增强", "self_if": "自生成·指令",
          "self_math": "自生成·数学", "self_seed": "自蒸馏种子"}


def _style(ax):
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["ink2"], labelsize=9)
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def load_state(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))


def champion_series(st: dict) -> list[dict]:
    """Champion after each generation (step function)."""
    gens = {g["gen"]: g for g in st["generations"]}
    out, cur = [], None
    for g in sorted(gens):
        if gens[g].get("promoted"):
            cur = gens[g]
        out.append({"gen": g, "champ": cur["gen"], "dev": cur["dev"], "test": cur.get("test")})
    return out


def fig_progress(st: dict):
    ser = champion_series(st)
    metrics = [("composite", "综合分"), ("if_prompt_acc", "指令遵循严格准确率"), ("math_acc", "GSM8K 准确率")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    for ax, (k, title) in zip(axes, metrics):
        _style(ax)
        xs = [s["gen"] for s in ser]
        ax.step(xs, [s["dev"][k] for s in ser], where="post", color=C["blue"], lw=2, label="冠军 (dev)")
        tx = [s["gen"] for s in ser if s["test"]]
        ax.plot(tx, [s["test"][k] for s in ser if s["test"]], "o", ms=7, color=C["blue"], mfc="white", mew=2,
                label="冠军 (test, 不参与选择)")
        cx, cy = [], []
        for g in st["generations"]:
            for c in g.get("candidates", []):
                cx.append(g["gen"])
                cy.append(c["dev"][k])
        ax.scatter(cx, cy, s=36, color=C["orange"], alpha=0.85, zorder=3, edgecolor="white", lw=1, label="候选 (dev)")
        ax.set_title(title, fontsize=11, color=C["ink"], loc="left")
        ax.set_xlabel("代数", fontsize=9, color=C["ink2"])
        ax.set_xticks(xs)
        ax.set_ylim(0, max(0.05, max([s["dev"][k] for s in ser] + cy + [0]) * 1.15))
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_kinds(st: dict):
    gens = [g for g in st["generations"] if g.get("promoted")]
    kinds = sorted(gens[0]["dev"]["per_kind"], key=lambda k: gens[-1]["dev"]["per_kind"].get(k, 0))
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc0ee", "#2a78d6", "#0f3f80"])
    M = np.array([[g["dev"]["per_kind"].get(k, 0) for g in gens] for k in kinds])
    fig, ax = plt.subplots(figsize=(1.3 + 0.9 * len(gens), 0.36 * len(kinds) + 1.2), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(kinds)), [KIND_ZH.get(k, k) for k in kinds], fontsize=9)
    ax.set_xticks(range(len(gens)), [f"第{g['gen']}代" for g in gens], fontsize=9)
    for i in range(len(kinds)):
        for j in range(len(gens)):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 0.55 else C["ink"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("各约束类型通过率（冠军，dev）", fontsize=11, loc="left", color=C["ink"])
    fig.tight_layout()
    return fig


def fig_mix(st: dict):
    gens = [g for g in st["generations"] if g.get("candidates")]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    _style(ax)
    xs = list(range(len(gens)))
    bottom = [0] * len(gens)
    for src in SRC_COLORS:
        vals = [next(c for c in g["candidates"] if c["cid"] == g["chosen"])["mix"].get(src, 0) for g in gens]
        if not any(vals):
            continue
        ax.bar(xs, vals, bottom=bottom, color=SRC_COLORS[src], width=0.6, label=SRC_ZH[src], edgecolor=C["surface"], lw=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(xs, [f"第{g['gen']}代" for g in gens])
    ax.set_title("训练数据构成（每代最佳候选）", fontsize=11, loc="left", color=C["ink"])
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    return fig


def fig_bandit(run_dir: Path):
    p = Path(run_dir) / "bandit.json"
    s = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    items = sorted(s.items(), key=lambda kv: kv[1][0] / sum(kv[1]))
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    fig.patch.set_facecolor(C["surface"])
    _style(ax)
    ax.grid(axis="x", color=C["grid"])
    ax.grid(axis="y", visible=False)
    means = [a / (a + b) for _, (a, b) in items]
    trials = [a + b - 2 for _, (a, b) in items]
    ax.barh(range(len(items)), means, color=[C["blue"] if t > 0 else C["grid"] for t in trials], height=0.6)
    ax.set_yticks(range(len(items)), [f"{OP_ZH.get(k, k)} (n={t:.0f})" for (k, _), t in zip(items, trials)], fontsize=8)
    ax.axvline(0.5, color=C["ink2"], lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_title("元优化器：改动算子成功率后验均值", fontsize=11, loc="left", color=C["ink"])
    fig.tight_layout()
    return fig


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def summary(st: dict) -> dict:
    gens = st["generations"]
    g0 = gens[0]
    champ = next(g for g in gens if g["gen"] == st["champion"])
    rows = []
    for g in gens:
        best = next((c for c in g.get("candidates", []) if c["cid"] == g.get("chosen")), None)
        rows.append({
            "gen": g["gen"], "promoted": g.get("promoted"), "parent": g.get("parent"),
            "ops": [OP_ZH[o] for o in best["ops"]] if best else [],
            "dev_composite": (best or g)["dev"]["composite"],
            "dev_if": (best or g)["dev"]["if_prompt_acc"], "dev_math": (best or g)["dev"]["math_acc"],
            "p_better": best["boot"]["p_better"] if best else None,
            "test": g.get("test"), "seconds": g.get("seconds"), "reflection": g.get("reflection", []),
            "synth": g.get("synth"),
        })
    return {"run": st["run_name"], "base_model": st["base_model"], "champion": st["champion"],
            "baseline_test": g0["test"], "champion_test": champ.get("test"), "champion_dev": champ["dev"],
            "generations": rows}


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    st = load_state(run_dir)
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    figs = {"progress": fig_progress(st), "kinds": fig_kinds(st), "bandit": fig_bandit(run_dir)}
    if any(g.get("candidates") for g in st["generations"]):
        figs["mix"] = fig_mix(st)
    imgs = {}
    for k, f in figs.items():
        b = _png(f)
        (out / f"{k}.png").write_bytes(b)
        imgs[k] = base64.b64encode(b).decode()
    sm = summary(st)
    (out / "summary.json").write_text(json.dumps(sm, indent=2, ensure_ascii=False), encoding="utf-8")

    def pct(x):
        return "—" if x is None else f"{x:.1%}"

    trs = []
    for r in sm["generations"]:
        t = r["test"] or {}
        pb = "—" if r["p_better"] is None else f"{r['p_better']:.2f}"
        trs.append(
            f"<tr><td>第{r['gen']}代</td><td>{'✅ 晋升' if r['promoted'] else '✖ 保留旧冠军'}</td>"
            f"<td>{html.escape(' + '.join(r['ops']) or ('基线' if r['gen'] == 0 else '默认配方'))}</td>"
            f"<td>{r['dev_composite']:.3f}</td><td>{pct(r['dev_if'])}</td><td>{pct(r['dev_math'])}</td>"
            f"<td>{pb}</td>"
            f"<td>{pct(t.get('if_prompt_acc'))}</td><td>{pct(t.get('math_acc'))}</td></tr>")
    refl = "".join(
        f"<details{' open' if i == len(sm['generations']) - 1 else ''}><summary>第{r['gen']}代 反思日志</summary><ul>"
        + "".join(f"<li>{html.escape(x)}</li>" for x in r["reflection"]) + "</ul></details>"
        for i, r in enumerate(sm["generations"]))
    b, c = sm["baseline_test"], sm["champion_test"] or sm["champion_dev"]
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSI 进化报告</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}} h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:var(--ink2);margin-bottom:20px}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}}
.kpi{{border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .kpi b{{font-size:26px;display:block}}
.kpi span{{color:var(--ink2);font-size:13px}} img{{max-width:100%;border:1px solid var(--line);border-radius:10px;margin:8px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
.wrap{{overflow-x:auto}} details{{margin:6px 0}} summary{{cursor:pointer;font-weight:600}}
</style></head><body><main>
<h1>基于大模型后训练的指令微调自进化系统 RSI · 运行 {html.escape(sm['run'])}</h1>
<div class="sub">基座 {html.escape(sm['base_model'])} · 当前冠军：第{sm['champion']}代 · 测试集独立于选择过程</div>
<div class="kpis">
<div class="kpi"><span>指令遵循严格准确率 (test)</span><b>{pct(b['if_prompt_acc'])} → {pct(c['if_prompt_acc'])}</b></div>
<div class="kpi"><span>约束级准确率 (test)</span><b>{pct(b['if_instr_acc'])} → {pct(c['if_instr_acc'])}</b></div>
<div class="kpi"><span>GSM8K (test)</span><b>{pct(b['math_acc'])} → {pct(c['math_acc'])}</b></div>
<div class="kpi"><span>退化率 (test)</span><b>{pct(b['degenerate_rate'])} → {pct(c['degenerate_rate'])}</b></div>
</div>
<img src="data:image/png;base64,{imgs['progress']}" alt="各代指标">
<div class="wrap"><table><tr><th>代</th><th>决策</th><th>改动</th><th>dev综合</th><th>dev指令</th><th>dev数学</th><th>P(更优)</th><th>test指令</th><th>test数学</th></tr>{''.join(trs)}</table></div>
<img src="data:image/png;base64,{imgs['kinds']}" alt="约束热力图">
{f'<img src="data:image/png;base64,{imgs["mix"]}" alt="数据构成">' if 'mix' in imgs else ''}
<img src="data:image/png;base64,{imgs['bandit']}" alt="算子成功率">
<h2>反思日志</h2>{refl}
</main></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"


if __name__ == "__main__":
    import sys
    print(build_report(Path(sys.argv[1])))
