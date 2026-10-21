"""RSI 可演示系统 (Gradio).

    python app/demo.py [--run pilot] [--port 7860] [--no-model]

Tabs: 进化总览 / 能力诊断 / 样例对比 / 实时对话 / 进化控制台
--no-model skips loading weights (dashboard + sample browser still work on CPU-only machines).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from rsi import report as R  # noqa: E402
from rsi.constraints import KIND_ZH, KINDS, Constraint, render_prompt, sample_constraints  # noqa: E402
from rsi.data import read_jsonl  # noqa: E402
from rsi.paths import RUNS  # noqa: E402

ZH2KIND = {v: k for k, v in KIND_ZH.items()}


def runs() -> list[str]:
    return sorted(p.name for p in RUNS.iterdir() if (p / "state.json").exists())


def st_of(run):
    return R.load_state(RUNS / run)


# ---------------------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------------------
def kpi_md(run):
    st = st_of(run)
    sm = R.summary(st)
    b = sm["baseline_test"]
    c = sm["champion_test"] or sm["champion_dev"]
    n_gen = len(st["generations"]) - 1
    n_prom = sum(1 for g in st["generations"][1:] if g.get("promoted"))
    pool = read_jsonl(RUNS / run / "self_pool.jsonl")

    def row(name, k):
        return f"| {name} | {b[k]:.1%} | **{c[k]:.1%}** | {c[k] - b[k]:+.1%} |"

    return "\n".join([
        f"### 运行 `{run}` · 基座 `{st['base_model']}` · 当前冠军：**第{st['champion']}代**",
        f"已进化 **{n_gen}** 代，晋升 **{n_prom}** 次；自生成并通过验证的训练样本累计 **{len(pool)}** 条。"
        "以下为**测试集**（不参与任何选择）上的结果：", "",
        "| 指标 | 原始基座 | 当前冠军 | 变化 |", "|---|---|---|---|",
        row("指令遵循·严格准确率", "if_prompt_acc"), row("指令遵循·约束级准确率", "if_instr_acc"),
        row("GSM8K 数学准确率", "math_acc"), row("输出退化率(越低越好)", "degenerate_rate"),
    ])


def gen_table(run):
    sm = R.summary(st_of(run))
    rows = []
    for r in sm["generations"]:
        t = r["test"] or {}
        rows.append({"代": r["gen"], "决策": "晋升" if r["promoted"] else "保留旧冠军",
                     "改动": " + ".join(r["ops"]) or ("基线" if r["gen"] == 0 else "默认配方"),
                     "dev综合": round(r["dev_composite"], 4), "dev指令": round(r["dev_if"], 3),
                     "dev数学": round(r["dev_math"], 3),
                     "P(更优)": None if r["p_better"] is None else round(r["p_better"], 2),
                     "test指令": None if not t else round(t["if_prompt_acc"], 3),
                     "test数学": None if not t else round(t["math_acc"], 3),
                     "耗时(分)": None if not r["seconds"] else round(r["seconds"] / 60, 1)})
    return pd.DataFrame(rows)


def reflection(run, gen):
    st = st_of(run)
    g = next((x for x in st["generations"] if x["gen"] == int(gen)), None)
    if not g:
        return ""
    lines = [f"#### 第{g['gen']}代反思日志"] + [f"- {x}" for x in g.get("reflection", [])]
    for c in g.get("candidates", []):
        ops = " + ".join(R.OP_ZH[o] for o in c["ops"]) or "默认配方"
        lines.append(f"- 候选{c['cid']}【{ops}】 dev综合 {c['dev']['composite']:.4f}，P(更优) {c['boot']['p_better']:.2f}，"
                     f"训练 {c['train']['examples']} 条 / {c['train']['seconds']:.0f}s，门控："
                     + "；".join(c["gate_reasons"]))
    return "\n".join(lines)


def gen_choices(run):
    return [str(g["gen"]) for g in st_of(run)["generations"]]


def refresh_overview(run):
    st = st_of(run)
    gens = gen_choices(run)
    return (kpi_md(run), R.fig_progress(st), gen_table(run), gr.update(choices=gens, value=gens[-1]),
            reflection(run, gens[-1]))


def refresh_diag(run):
    st = st_of(run)
    mix = R.fig_mix(st) if any(g.get("candidates") for g in st["generations"]) else None
    return R.fig_kinds(st), R.fig_bandit(RUNS / run), mix


# ---------------------------------------------------------------------------------------
# sample browser (no GPU)
# ---------------------------------------------------------------------------------------
def sources(run) -> dict[str, dict]:
    """Every evaluated model of a run: label -> {eval_dir, adapter}.  Champions first, then all candidates."""
    st = st_of(run)
    out = {}
    for g in st["generations"]:
        if g.get("promoted"):
            lab = "第0代·原始基座" if g["gen"] == 0 else f"第{g['gen']}代·冠军"
            out[lab] = {"eval_dir": g["eval_dir"], "adapter": g.get("adapter")}
    for g in st["generations"]:
        for c in g.get("candidates", []):
            if g.get("promoted") and c["cid"] == g.get("chosen"):
                continue
            ops = "+".join(R.OP_ZH[o] for o in c["ops"]) or "默认配方"
            out[f"第{g['gen']}代·候选{c['cid']}（{ops}，未晋升）"] = {"eval_dir": c["dir"], "adapter": c["adapter"]}
    return out


def source_choices(run):
    labs = list(sources(run))
    champs = [l for l in labs if "冠军" in l or "基座" in l]
    right = champs[-1] if len(champs) > 1 else labs[-1]
    return (gr.update(choices=labs, value=labs[0]), gr.update(choices=labs, value=right))


def sample_ids(run, kind_filter):
    first = next(iter(sources(run).values()))
    rows = read_jsonl(Path(first["eval_dir"]) / "eval_dev.jsonl")
    ids = [r["id"] for r in rows if r["type"] == "if" and (not kind_filter or ZH2KIND[kind_filter] in r["kinds"])]
    ids += [r["id"] for r in rows if r["type"] == "math" and not kind_filter]
    return gr.update(choices=ids, value=ids[0] if ids else None)


def show_sample(run, sid, src_a, src_b):
    out = []
    srcs = sources(run)
    prompt = ""
    for lab in (src_a, src_b):
        s = srcs.get(lab)
        rows = {r["id"]: r for r in read_jsonl(Path(s["eval_dir"]) / "eval_dev.jsonl")} if s else {}
        r = rows.get(sid)
        if not r:
            out.append("")
            continue
        prompt = r["prompt"]
        if r["type"] == "if":
            verdict = " ".join(f"{'✅' if ok else '❌'}{KIND_ZH.get(k, k)}" for k, ok in zip(r["kinds"], r["checks"]))
        else:
            verdict = f"{'✅' if r['correct'] else '❌'} 预测={r['pred']} 标准={r['gold']}"
        flag = "　⚠️未正常结束" if not r["finished"] else ""
        out.append(f"**{lab}**　{verdict}{flag}\n\n```text\n{r['response'] or '(空)'}\n```")
    return f"```text\n{prompt}\n```", out[0], out[1]


# ---------------------------------------------------------------------------------------
# live chat
# ---------------------------------------------------------------------------------------
class Live:
    """One base model in VRAM; LoRA adapters are attached lazily and switched per request."""

    def __init__(self, run: str):
        self.run, self.model, self.tok, self.names = run, None, None, {}
        self.lock = threading.Lock()

    def _adapter(self, path: str) -> str:
        from peft import PeftModel
        if path not in self.names:
            name = f"a{len(self.names)}"
            if not self.names:
                self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name)
            else:
                self.model.load_adapter(path, adapter_name=name)
            self.model.eval()
            self.names[path] = name
        return self.names[path]

    def answer(self, prompt: str, label: str, max_new: int, temperature: float) -> str:
        from rsi.modeling import generate, load_model
        with self.lock:
            if self.model is None:
                st = st_of(self.run)
                self.model, self.tok = load_model(str((ROOT / st["base_model"]).resolve()))
            adapter = (sources(self.run).get(label) or {}).get("adapter")
            if adapter:
                name = self._adapter(adapter)  # may wrap self.model in a PeftModel: bind set_adapter afterwards
                self.model.set_adapter(name)
                ctx = _null()
            else:
                ctx = self.model.disable_adapter() if self.names else _null()
            with ctx:
                return generate(self.model, self.tok, [prompt], max_new_tokens=max_new, temperature=temperature,
                                log=None)[0][0]["text"]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


LIVE: dict[str, Live] = {}


def build_prompt(task, kinds_zh, seed):
    rng = random.Random(int(seed))
    cons = [sample_constraints(rng, 1, kinds=[ZH2KIND[k]])[0] for k in kinds_zh]
    return render_prompt(task, cons), json.dumps([c.to_dict() for c in cons], ensure_ascii=False)


def verify(text, cons_json):
    cons = [Constraint.from_dict(d) for d in json.loads(cons_json or "[]")]
    if not cons:
        return "（未附加可验证约束）"
    return " ".join(f"{'✅' if c.check(text) else '❌'}{KIND_ZH[c.kind]}" for c in cons)


def chat(run, task, kinds_zh, seed, gen_a, gen_b, max_new, temp, no_model):
    prompt, cons = build_prompt(task, kinds_zh, seed)
    if no_model:
        return prompt, "（以 --no-model 启动，未加载权重）", "", "", ""
    live = LIVE.setdefault(run, Live(run))
    a = live.answer(prompt, gen_a, int(max_new), float(temp))
    b = live.answer(prompt, gen_b, int(max_new), float(temp))
    return prompt, a, verify(a, cons), b, verify(b, cons)


# ---------------------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------------------
PROC: dict[str, subprocess.Popen] = {}


def start_evolve(run, n):
    p = PROC.get(run)
    if p and p.poll() is None:
        return f"运行 {run} 已在进化中 (pid={p.pid})"
    cfg = ROOT / "configs" / f"{run}.yaml"
    if not cfg.exists():
        return f"找不到配置 {cfg}"
    out = open(RUNS / run / "evolve_stdout.txt", "a", encoding="utf-8")
    PROC[run] = subprocess.Popen([sys.executable, "-m", "rsi.loop", "--config", str(cfg), "--generations", str(int(n))],
                                 cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    return f"已启动：再进化 {int(n)} 代 (pid={PROC[run].pid})。注意：进化与实时对话共用一块 GPU。"


def tail_log(run, n=60):
    p = RUNS / run / "log.txt"
    if not p.exists():
        return ""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


# ---------------------------------------------------------------------------------------
def ui(default_run: str, no_model: bool):
    with gr.Blocks(title="RSI 指令微调自进化系统") as demo:
        gr.Markdown("# 基于大模型后训练的指令微调自进化系统 RSI\n"
                    "冠军模型自己出题、自己作答、程序化验证、筛选成训练数据 → 元优化器提出新训练配方 → "
                    "训练候选 → 自动评测 → 统计门控决定是否晋升 → 反思并调整下一代策略。")
        run = gr.Dropdown(runs(), value=default_run, label="运行")
        with gr.Tab("进化总览"):
            kpi = gr.Markdown()
            prog = gr.Plot(label="各代指标")
            table = gr.Dataframe(label="每代决策", interactive=False)
            with gr.Row():
                gsel = gr.Dropdown(label="查看第几代反思", choices=[])
            refl = gr.Markdown()
            btn = gr.Button("刷新")
            btn.click(refresh_overview, run, [kpi, prog, table, gsel, refl])
            run.change(refresh_overview, run, [kpi, prog, table, gsel, refl])
            gsel.change(reflection, [run, gsel], refl)
            demo.load(refresh_overview, run, [kpi, prog, table, gsel, refl])
        with gr.Tab("能力诊断"):
            with gr.Row():
                heat = gr.Plot(label="约束类型热力图")
                band = gr.Plot(label="元优化器经验")
            mix = gr.Plot(label="训练数据构成")
            gr.Button("刷新").click(refresh_diag, run, [heat, band, mix])
            demo.load(refresh_diag, run, [heat, band, mix])
        with gr.Tab("样例对比"):
            with gr.Row():
                kf = gr.Dropdown([""] + list(KIND_ZH.values()), value="", label="按约束类型筛选（空=全部）")
                sid = gr.Dropdown(label="dev 样例")
            with gr.Row():
                ga = gr.Dropdown(label="左侧模型")
                gb = gr.Dropdown(label="右侧模型")
            sp = gr.Markdown()
            with gr.Row():
                sa, sb = gr.Markdown(), gr.Markdown()
            kf.change(sample_ids, [run, kf], sid)
            demo.load(sample_ids, [run, kf], sid)
            demo.load(source_choices, run, [ga, gb])
            run.change(source_choices, run, [ga, gb])
            for w in (sid, ga, gb):
                w.change(show_sample, [run, sid, ga, gb], [sp, sa, sb])
        with gr.Tab("实时对话"):
            gr.Markdown("输入任意任务并勾选约束，系统会把约束改写进提示词，两个模型分别作答，并当场程序化验证。")
            task = gr.Textbox("Write a short introduction to the Great Wall of China for tourists.", label="任务", lines=2)
            with gr.Row():
                kinds = gr.CheckboxGroup(list(KIND_ZH.values()), value=["<<标题>>", "精确条目数", "P.S.附言"], label="附加约束")
            with gr.Row():
                la = gr.Dropdown(label="左侧模型")
                lb = gr.Dropdown(label="右侧模型")
            demo.load(source_choices, run, [la, lb])
            run.change(source_choices, run, [la, lb])
            with gr.Row():
                seed = gr.Number(0, label="约束参数种子", precision=0)
                mx = gr.Slider(64, 1024, 384, step=32, label="最大生成长度")
                tp = gr.Slider(0, 1.2, 0, step=0.1, label="温度(0=贪心)")
            go = gr.Button("生成并验证", variant="primary")
            fp = gr.Textbox(label="最终提示词", lines=4)
            with gr.Row():
                with gr.Column():
                    oa = gr.Textbox(label="左侧回答", lines=14)
                    va = gr.Markdown()
                with gr.Column():
                    ob = gr.Textbox(label="右侧回答", lines=14)
                    vb = gr.Markdown()
            go.click(lambda *a: chat(*a, no_model), [run, task, kinds, seed, la, lb, mx, tp], [fp, oa, va, ob, vb])
        with gr.Tab("进化控制台"):
            with gr.Row():
                ng = gr.Number(1, label="继续进化代数", precision=0)
                sbtn = gr.Button("启动进化", variant="primary")
            msg = gr.Markdown()
            logbox = gr.Textbox(label="运行日志（尾部）", lines=24)
            sbtn.click(start_evolve, [run, ng], msg)
            gr.Button("刷新日志").click(tail_log, run, logbox)
            gr.Timer(10).tick(tail_log, run, logbox)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    rs = runs()
    default = a.run or ("pilot" if "pilot" in rs else rs[-1])
    ui(default, a.no_model).queue().launch(server_name="127.0.0.1", server_port=a.port, inbrowser=False)
