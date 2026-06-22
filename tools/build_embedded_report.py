from __future__ import annotations

import base64
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "assets" / "figures"
OUT_PATH = ROOT / "reports" / "vlm_nla_research_proposal_preliminary_results.html"

NLA_URL = "https://transformer-circuits.pub/2026/nla/index.html"
LOGIT_LENS_URL = "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens"
TUNED_LENS_URL = "https://arxiv.org/abs/2303.08112"
ATTENTION_LENS_URL = "https://arxiv.org/abs/2310.16270"
HEADLENS_URL = "https://arxiv.org/abs/2603.18523"
SAE_URL = "https://transformer-circuits.pub/2023/monosemantic-features/index.html"
EAZY_URL = "https://openaccess.thecvf.com/content/ICCV2025/papers/Che_Hallucinatory_Image_Tokens_A_Training-free_EAZY_Approach_to_Detecting_and_ICCV_2025_paper.pdf"
VLM_SHORTCUT_URL = "https://openreview.net/forum?id=ZPQU4uGMBA"
VLM_CIRCUIT_TRACING_URL = "https://arxiv.org/abs/2602.20330"

NLA = f'<a href="{NLA_URL}">NLA</a>'
NLA_FULL = f'<a href="{NLA_URL}">Natural Language Autoencoder</a>'
LOGIT_LENS = f'<a href="{LOGIT_LENS_URL}">Logit Lens</a>'
TUNED_LENS = f'<a href="{TUNED_LENS_URL}">Tuned Lens</a>'
ATTENTION_LENS = f'<a href="{ATTENTION_LENS_URL}">Attention Lens</a>'
HEADLENS = f'<a href="{HEADLENS_URL}">HeadLens</a>'
SAE = f'<a href="{SAE_URL}">SAE</a>'
SPARSE_AUTOENCODER = f'<a href="{SAE_URL}">Sparse Autoencoder</a>'
EAZY = f'<a href="{EAZY_URL}">EAZY</a>'
HALLUCINATORY_IMAGE_TOKENS = f'<a href="{EAZY_URL}">Hallucinatory Image Tokens</a>'


CAPTIONS = {
    "experiment4_metric_comparison.png": "LLaVA-1.5 baseline 的汇总图。这里的 1tok/4tok/8tok 表示 AV prompt 里放了几个用于注入 activation 的 special tokens；512 表示训练样本数；dualcontrast 表示训练时同时使用 activation-shuffle contrastive loss 和 response-contrastive loss。图的核心信息是：8 tokens 比 4 tokens 更依赖 activation，但整体 ranking 仍然弱，说明机制通了，性能还不够。",
    "4tok_512_dualcontrast_metrics.png": "LLaVA-1.5 4-token dual-contrastive run 的指标图。可以把它看作 8-token run 的弱 baseline：activation signal 有，但正确 response 往往排不到前面。",
    "8tok_512_dualcontrast_metrics.png": "LLaVA-1.5 8-token dual-contrastive run 的指标图。相比 4 tokens，matched-vs-shuffled gap 更大，说明多个 AV special tokens 确实改善 activation conditioning。",
    "4tok_512_dualcontrast_synthetic_00000.png": "synthetic_00000 是一张程序生成的简单几何图。candidate response 不是模型自由生成出来的，而是从候选 response pool 中逐个 teacher-forced 计算 NLL 后排序得到。这个例子展示 4-token 模型有时能把正确解释排得很靠前。",
    "4tok_512_dualcontrast_synthetic_00001.png": "LLaVA 4-token synthetic example。图中的正确 response 和 top candidates 用来观察模型到底错在颜色、形状、位置，还是整体模板 prior。",
    "4tok_512_dualcontrast_synthetic_00002.png": "LLaVA 4-token failure case。正确答案是两物体描述，但模型更偏好高频、容易输出的错误模板；这说明 NLL ranking 仍受 language prior 影响。",
    "4tok_512_dualcontrast_synthetic_00003.png": "LLaVA 4-token qualitative example。synthetic_00xxx 都是自动生成的 controlled visual concept 样本，用于快速验证 activation 是否携带颜色、形状、位置等信息。",
    "4tok_512_dualcontrast_synthetic_00004.png": "LLaVA 4-token qualitative example。这里的重点不是图片本身复杂，而是 controlled label 让我们可以精确知道正确 explanation。",
    "8tok_512_dualcontrast_synthetic_00000.png": "LLaVA 8-token example。8tok 代表 activation 被 adapter 展开到 8 个连续 special tokens，而不是只塞进 1 个 token embedding。",
    "8tok_512_dualcontrast_synthetic_00001.png": "LLaVA 8-token example。matched/shuffled gap 很大说明模型在使用 activation，但正确 response 仍可能被一些更短、更高 prior 的候选压过。",
    "8tok_512_dualcontrast_synthetic_00002.png": "LLaVA 8-token two-object failure。这个例子提醒我们：activation signal 存在不等于 natural-language decoder 已经可靠。",
    "8tok_512_dualcontrast_synthetic_00003.png": "LLaVA 8-token qualitative example，用于观察多 token 注入后 candidate ranking 的局部改善。",
    "8tok_512_dualcontrast_synthetic_00004.png": "LLaVA 8-token qualitative example。该组图总体说明 LLaVA 适合做机制验证，但不是当前最适合继续冲性能的 LVLM。",
    "qwen3vl_layer15_probe_targets.png": "Qwen3-VL 的 target probe。这里比较 last_prompt、single image token 和 image_mean 三种 layer-15 activation target。image_mean 的 micro-F1 和 exact-set 明显更高，说明 Qwen3-VL 的全局视觉语义在 image-token block 中更容易被线性读出。",
    "qwen3vl_llava_metric_comparison.png": "Qwen3-VL 和 LLaVA-1.5 的主结果对比。Qwen3-VL 在 sensitivity delta、mean rank、top1/top5 上都大幅超过 LLaVA，说明 LLaVA 的弱结果不只是 NLA 机制问题，也和模型结构及视觉表征质量有关。",
    "qwen3vl_raw_synthetic_00000.png": "Qwen3-VL synthetic example。raw NLL ranking 可以正确选择 response，即使 greedy generation 不一定完全正确。这里强调：本报告主要看 teacher-forced ranking 和 sensitivity，而不是只看自由生成一句话。",
    "qwen3vl_raw_synthetic_00001.png": "Qwen3-VL synthetic candidate ranking example。top candidates 是从候选池里来的，不是 beam search。正确 response 排在前面意味着该 activation 对这段解释的 NLL 最低。",
    "qwen3vl_raw_synthetic_00002.png": "Qwen3-VL synthetic visual concept example。synthetic images 用简单图形降低数据噪声，目的是先测 NLA-style AV 是否能读出 controlled visual semantics。",
    "qwen3vl_raw_synthetic_00005.png": "Qwen3-VL synthetic example。用于观察 candidate scores 与具体颜色/形状/位置错误之间的关系。",
    "qwen3vl_heldout_raw_synthetic_00000.png": "Qwen3-VL held-out synthetic example。held-out seed 表示图片和 label 不是训练那 512 个样本，能检查模型是否只是背了训练集。",
    "qwen3vl_heldout_raw_synthetic_00001.png": "Qwen3-VL held-out example。正确答案排得很靠前，top wrong candidate 往往只是颜色或位置细节相近，说明错误模式是细粒度竞争，不是完全无关。",
    "qwen3vl_gain_synthetic_00005.png": "activation-gain scoring 的例子。gain score 会减掉 reference activation 下的 NLL，试图削弱 response 本身的语言 prior；它是 diagnostic，不一定总比 raw NLL 更稳定。",
    "qwen3vl_coco_token_metrics.png": "COCO object-token 指标图。中心结论是 bbox local token group 比 single object-center token 更稳：local group 的 sensitivity delta 和 top5 都更高。",
    "bbox_train_0_coco_000000474021_445486.png": "COCO train-like bbox-token group example。红框是 COCO object bbox，蓝色格子是被选中并平均的 Qwen3-VL image tokens。这个 target 不是全图 image_mean，而是 object region 内的一小片 tokens。",
    "bbox_train_1_coco_000000455981_1093311.png": "COCO bbox-token group example。bbox 内多个 token 的平均更稳定，因为一个 object 的语义通常分布在多个相邻 image tokens 上。",
    "bbox_heldout_0_coco_000000042563_172991.png": "COCO held-out bbox-token group example。该例 target 是 bottom-region train。long-label 版本要求恢复 object + region + full COCO caption，过难；short-label 版本只要求 object + region，更接近“这些 image tokens 被理解成什么”。",
    "center_train_0_coco_000000474021_445486.png": "COCO single center-token example。蓝色单格表示 bbox 中心对应的一个 image token。单 token 有语义信号，但比 bbox group 更容易受 grid 对齐、物体大小、遮挡影响。",
    "center_train_1_coco_000000455981_1093311.png": "COCO center-token example。和 bbox-token view 对比可以看出：只解释一个中心 token 更尖锐，但也更脆弱。",
    "center_heldout_0_coco_000000042563_172991.png": "COCO held-out center-token example。该图对应 bbox held-out example；比较两者可以直观看到 single-token 与 local-token-group 的差别。",
}


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def link_methods(text: str) -> str:
    escaped = html.escape(text)
    replacements = [
        ("Hallucinatory Image Tokens", HALLUCINATORY_IMAGE_TOKENS),
        ("Sparse Autoencoder", SPARSE_AUTOENCODER),
        ("Natural Language Autoencoder", NLA_FULL),
        ("Attention Lens", ATTENTION_LENS),
        ("Tuned Lens", TUNED_LENS),
        ("Logit Lens", LOGIT_LENS),
        ("HeadLens", HEADLENS),
        ("EAZY", EAZY),
        ("SAE", SAE),
        ("NLA", NLA),
    ]
    for needle, repl in replacements:
        escaped = escaped.replace(needle, repl)
    return escaped


def figure_html(path: Path) -> str:
    caption = CAPTIONS.get(path.name, path.name.replace("_", " ").replace(".png", ""))
    rel = path.relative_to(ROOT)
    return f"""
      <figure>
        <img src="{data_uri(path)}" alt="{html.escape(path.stem)}">
        <figcaption>
          {link_methods(caption)}<br>
          <code>{html.escape(str(rel))}</code>
        </figcaption>
      </figure>
    """


def figures_for(subdir: str) -> str:
    paths = sorted((FIG_ROOT / subdir).glob("*.png"))
    return "\n".join(figure_html(path) for path in paths)


def main() -> None:
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM-NLA 开题报告与初步实验结果</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #596371;
      --line: #d8dee8;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --accent: #1b6fb8;
      --accent-2: #1f8a70;
      --soft: #f8fafc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 16px/1.72 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    header {{
      background: #101820;
      color: white;
      padding: 56px 24px 40px;
    }}
    header .wrap, main {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: clamp(34px, 6vw, 60px);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 48px 0 16px;
      font-size: 28px;
      line-height: 1.25;
      border-bottom: 2px solid var(--line);
      padding-bottom: 8px;
    }}
    h3 {{
      margin: 30px 0 10px;
      font-size: 21px;
      line-height: 1.3;
      color: #123c5c;
    }}
    h4 {{
      margin: 22px 0 8px;
      font-size: 17px;
      color: #1d3b54;
    }}
    main {{
      padding: 30px 24px 64px;
      background: var(--panel);
    }}
    p {{ margin: 10px 0; }}
    a {{ color: var(--accent); }}
    .subtitle {{
      max-width: 900px;
      margin: 0;
      font-size: 18px;
      color: #d9e3ee;
    }}
    .meta {{
      margin-top: 22px;
      color: #aebdcc;
      font-size: 14px;
    }}
    .abstract {{
      border-left: 5px solid var(--accent);
      background: #eef6fc;
      padding: 18px 20px;
      margin: 18px 0 24px;
    }}
    .callout {{
      border-left: 4px solid var(--accent-2);
      background: #effaf6;
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .note {{
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 8px;
      padding: 14px 16px;
      margin: 16px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 24px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f0f3f7; }}
    code {{
      background: #eef1f5;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 0.92em;
    }}
    pre {{
      overflow: auto;
      background: #0f1720;
      color: #e8eef6;
      padding: 14px;
      border-radius: 8px;
      line-height: 1.5;
    }}
    figure {{
      margin: 24px 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 0 auto;
      border-radius: 4px;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 16px 0;
    }}
    .mini {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fcfdff;
    }}
    .mini strong {{ color: #123c5c; }}
    ul {{ padding-left: 22px; }}
    li {{ margin: 7px 0; }}
    .footnote {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 36px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>VLM-{NLA}：用 {NLA_FULL} 解释 LVLM 内部视觉 activation</h1>
      <p class="meta">生成日期：2026-06-22。本文所有图片均以内嵌 base64 形式写入 HTML，可单文件打开。</p>
    </div>
  </header>
  <main>
    <section>
      <h2>1. 摘要</h2>
      <div class="abstract">
        <p>大型视觉语言模型（LVLM，例如 LLaVA、Qwen-VL）已经可以把图片和文本联合起来回答问题，但它们的内部视觉表征仍然很难被人直接理解。模型说“图里有一辆火车”时，我们通常只能看到最终 answer，却不知道是哪几个 image tokens 支持了这个判断，也不知道模型是否真的看到了火车、是否被背景知识 shortcut 诱导、是否有少数 high-impact image tokens 推动了 hallucination。这个问题对于 VLM hallucination、visual grounding、multimodal reasoning 和模型安全都很关键。</p>
        <p>本项目尝试把 {NLA_FULL}（{NLA}）从 text-only LLM activation 扩展到 LVLM activation。{NLA} 的基本思想是训练一个 activation verbalizer（AV）把 activation vector 转成自然语言解释，再训练一个 activation reconstructor（AR）从解释文本重构 activation。原始 {NLA} 主要解释 text model 的 residual-stream activation；本项目的核心问题是：LVLM 中由图像输入产生的 internal activation，尤其是某个 image token 或某个 object region 内的一组 image tokens，能不能也被 AV 用自然语言解释出来？如果可以，这种解释是否能帮助我们研究 hallucination、shortcut 和视觉 grounding？</p>
        <p>作为开题阶段的 preliminary study，本项目完成了三组实验。第一组用 LLaVA-1.5 验证机制可行性：HF 实现里 image features 经过 projector 后替换 <code>&lt;image&gt;</code> token embedding，再进入 language model，这和 {NLA} 的 <code>inputs_embeds</code> injection 机制非常接近。实验说明 LLaVA 上可以做多 token activation injection，但 AV 性能较弱。第二组切换到 Qwen3-VL，用 layer-15 的 <code>image_mean</code> activation 训练 8-token AV，在 synthetic visual concepts 上达到很强的 candidate ranking 和 matched-vs-shuffled sensitivity。第三组进一步使用真实 COCO 图片，不再只解释全图 <code>image_mean</code>，而是解释 single object-center image token 和 object bbox 内 4-8 个 local image tokens 的平均 activation。结果显示：single image token 有语义信号，但噪声较大；object bbox 内的 local token group 更稳定，held-out short-label top5 从 53.1% 提升到 65.6%，sensitivity delta 从 +0.156 提升到 +0.412。</p>
        <p>这些结果支持一个谨慎但有价值的结论：{NLA}-style activation-to-language 方法可以迁移到 LVLM；Qwen3-VL 比 LLaVA-1.5 更适合作为后续主线；解释 specific image token/group 是可行方向，但真正稳定的解释很可能需要 local token group、多层 activation、AR 闭环、hard-negative evaluation，以及与 hallucination/shortcut 数据集结合的 causal intervention。</p>
      </div>
    </section>

    <section>
      <h2>2. 背景与研究动机</h2>
      <h3>2.1 为什么不能只看最终输出</h3>
      <p>LVLM 的最终输出是一个高度压缩后的结果。一个回答可能正确，但内部可能用了错误的视觉证据；一个回答可能 hallucinate，但 hallucination 可能来自 language prior，也可能来自少数异常 image tokens。只看 final answer，无法区分这些情况。为了研究模型是否真正 grounded in image，我们需要把分析粒度从 output text 往前推进到 internal activation，尤其是视觉 token 对应的 activation。</p>
      <p>这个项目关心的问题不是“让模型再解释一遍自己的答案”，而是“让某个内部 activation 本身被解释”。这两者差别很大。前者容易变成 post-hoc rationalization；后者更接近 mechanistic interpretability：我们固定一个 activation，把它注入 AV，让 AV 说出这个 activation 最支持什么语义，再用 matched-vs-shuffled 或 AR reconstruction 检验这段解释是否真的依赖该 activation。</p>

      <h3>2.2 从 {LOGIT_LENS} 到 {TUNED_LENS}：把隐藏状态投影成可读信息</h3>
      <p>{LOGIT_LENS} 是早期很直观的一类方法：把 transformer 中间层 hidden state 直接乘以最终 unembedding matrix，看它在词表上偏向哪些 token。它的优点是简单、零训练、容易画出每层预测如何演化；缺点是中间层 hidden state 未必已经处在最终 unembedding 可直接读取的空间，因此解释可能偏粗。</p>
      <p>{TUNED_LENS} 在这个思路上加了一层训练出来的 affine translator，让每一层 hidden state 先被校准到更接近最终输出空间，再读出 token distribution。它比 raw {LOGIT_LENS} 更稳，但解释对象仍然主要是“下一 token prediction”，而不是“这个 activation 对应的自然语言概念”。</p>

      <h3>2.3 从 {ATTENTION_LENS}/{HEADLENS} 到 {SAE}：从部件贡献到特征字典</h3>
      <p>{ATTENTION_LENS} 和 {HEADLENS} 类方法把 attention head 的输出或贡献单独拿出来分析，问某个 head 在某个位置推动了哪些 token 或语义。这类方法有助于定位模型内部的局部计算部件，但通常仍然需要人为解释每个 head 的功能。</p>
      <p>{SPARSE_AUTOENCODER}（{SAE}）则走另一条路线：把 dense activation 分解成稀疏 feature dictionary。一个 activation 可以被表示成少量 feature 的组合，feature 往往比原始 neuron 更接近人类可理解的概念。{SAE} 的优势是可以做大规模 feature discovery 和 circuit tracing；局限是每个 feature 仍需要命名、聚合和语义验证，而且对 multimodal image tokens 的解释还需要结合视觉区域。</p>

      <h3>2.4 {NLA} 的位置：让 activation 用自然语言表达自己</h3>
      <p>{NLA} 可以看作上述路线的补充。{LOGIT_LENS}/{TUNED_LENS} 更像“把 hidden state 投影到词表”；{SAE} 更像“把 activation 拆成 feature”；{NLA} 则尝试直接学习 <code>activation -> explanation text -> activation</code> 的 autoencoder。AV 负责把 activation verbalize 成 explanation；AR 负责从 explanation reconstruct activation。解释质量不只靠人读起来像不像，而要看 explanation 是否能保留足够信息让 AR 找回原 activation。</p>
      <p>把 {NLA} 迁移到 LVLM 后，最吸引人的问题是：我们能否让某个 image token 或 object region 的 activation 说出“我在模型内部像什么”？如果可以，就能把 {NLA} 接到 hallucination image tokens、shortcut circuits、object grounding、OCR regions 等具体研究问题上。</p>

      <h3>2.5 与 hallucination 和 shortcut 文献的连接</h3>
      <p>{EAZY} / {HALLUCINATORY_IMAGE_TOKENS} 一类工作提示我们：LVLM hallucination 可能不是全图均匀导致的，而是少数 image tokens 对最终回答有不成比例的影响。如果 VLM-{NLA} 能解释这些 high-impact image tokens，就可以进一步问：它们是否 verbalize 成了不存在的物体？zero-out 前后它们的解释是否变化？</p>
      <p>VLM shortcut 相关工作则提醒我们：模型在 diagram reasoning、visual relation、spatial reasoning 中可能靠背景知识或语言 prior 得分，而不是读懂图像本身。VLM-NLA 可以提供一个中间层检查工具：当我们对图片做 counterfactual edit 时，object-token activation 的解释是否跟着 visual evidence 改变，还是仍然停留在 shortcut prior 上？</p>
    </section>

    <section>
      <h2>3. Preliminary Experiment：实验目的与设计</h2>
      <h3>3.1 开题阶段要回答的两个问题</h3>
      <p>这组实验不是最终系统，而是为了回答两个 feasibility questions。</p>
      <ul>
        <li><strong>问题一：{NLA} 能不能迁移到 LVLM？</strong> 也就是，能否从 LVLM 内部取出由图像输入产生的 activation，把它通过 special token embedding 注入语言模型，并训练 AV 输出解释文本。</li>
        <li><strong>问题二：AV special token 能不能有多个？</strong> 原始 {NLA} 常见设定是把一个 activation 注入一个 marker token。本项目测试把一个 activation 通过 adapter 展开到多个连续 special tokens，观察是否能缓解 one-token bottleneck，并为未来多层 activation 同时优化做准备。</li>
      </ul>

      <h3>3.2 测试的 LVLM 及结构差异</h3>
      <table>
        <tr><th>模型</th><th>为什么选它</th><th>和 {NLA} injection 的关系</th><th>实验中的经验结论</th></tr>
        <tr><td>LLaVA-1.5 7B</td><td>结构简单，HF forward path 清楚，适合 first feasibility test。</td><td>图像经过 vision tower 和 projector 后，替换文本序列里的 <code>&lt;image&gt;</code> embeddings，再送入 language model。重复 <code>&lt;image&gt;</code> 可以得到多个 injection positions。</td><td>机制可行，但 AV ranking 较弱。适合作 baseline，不适合继续作为主力冲性能。</td></tr>
        <tr><td>Qwen3-VL 8B Instruct</td><td>视觉能力更强，image-token 机制更丰富，更适合研究 visual token representation。</td><td>使用 <code>&lt;|vision_start|&gt;</code>、<code>&lt;|image_pad|&gt;</code>、<code>&lt;|vision_end|&gt;</code> 作为视觉占位。AV prompt 中把一个 <code>&lt;|image_pad|&gt;</code> 扩展成 8 个连续 tokens。</td><td>synthetic image_mean 上效果很强；COCO object-token 上 local token group 有可靠信号。</td></tr>
      </table>

      <h3>3.3 Activation target：到底解释什么</h3>
      <p>实验都先固定在 language model layer 15。选择 layer 15 的直觉是：它不是太早的低级视觉特征，也不是太晚已经高度贴近输出 token 的状态，比较适合作为 semantic activation 的初步目标。</p>
      <table>
        <tr><th>target</th><th>含义</th><th>为什么测</th></tr>
        <tr><td><code>last_prompt</code></td><td>图文 prompt 最后一个文本 token 的 layer-15 activation。</td><td>它像是“模型准备回答前”的压缩状态，在 LLaVA first test 中最容易接到 text decoder。</td></tr>
        <tr><td><code>image</code></td><td>一个固定位置或中间位置的 image token activation。</td><td>用于测试 single image token 是否已经含有可解释视觉语义。</td></tr>
        <tr><td><code>image_mean</code></td><td>所有 image tokens 的 layer-15 activation 平均。</td><td>用于验证 Qwen3-VL 中全图视觉语义是否能被 AV 读出；这是强 feasibility target，但不是最终 local interpretability target。</td></tr>
        <tr><td><code>object_center</code></td><td>COCO bbox 中心对应的单个 Qwen3-VL image token。</td><td>直接回答“能不能解释特定一个 image token”。</td></tr>
        <tr><td><code>object_bbox_mean</code></td><td>COCO bbox 内 4-8 个 local image tokens 的平均。</td><td>测试一个 object 的语义是否更稳定地分布在 local token group 中。</td></tr>
      </table>

      <h3>3.4 数据设计</h3>
      <p>synthetic 数据是自动生成的简单图片，包含 1-2 个彩色几何图形，例如 blue triangle、purple square，并带有粗位置标签，例如 left、center、right。这类数据不追求真实复杂度，而是用于快速、可控地验证 activation 是否编码了颜色、形状、位置。图名里的 <code>synthetic_00000</code>、<code>synthetic_00001</code> 就是样本编号。</p>
      <p>COCO 数据来自 MSCOCO val2017。每张图选一个 prominent object annotation，用 bbox 映射到 Qwen3-VL image-token grid。红框是 COCO bbox，蓝色格子是被解释的 image token 或 token group。COCO 部分有两种 label 版本：</p>
      <ul>
        <li><strong>long-label version：</strong><code>These image tokens represent a train in the bottom region of the image. The full COCO caption is: ...</code>。这是第一版设计，想测试 local tokens 是否能支持 object + region + image-level caption。但这个目标偏难，也混入了全图 caption，不是最贴近“局部 token 被理解成什么”的问题。</li>
        <li><strong>short-label version：</strong><code>These image tokens represent a train in the bottom region of the image.</code>。这是后来修正后的版本，只保留 object category + coarse region，更适合解释 specific image token/group。</li>
      </ul>
      <div class="note">
        <p><strong>为什么 response 里会出现 “The full COCO caption is...”?</strong> 这不是 Qwen3-VL 自己生成的奇怪格式，而是 COCO long-label 数据构造时人为写进 target response 的。对应代码在 <code>extract_qwen3vl_coco_object_tokens.py</code> 的 <code>build_description</code>。后来为了更符合“解释局部 image token”的目标，我又用 <code>make_qwen3vl_coco_short_label_parquet.py</code> 生成了 short-label parquet，把 full caption 去掉了。因此报告里 COCO 的主结论应优先看 short-label 结果。</p>
      </div>

      <h3>3.5 AV prompt 与多 token 注入</h3>
      <p>Qwen3-VL synthetic 的 AV prompt 是：</p>
      <pre>&lt;|im_start|&gt;user
You are a careful interpreter of Qwen3-VL internal activations.

The activation is inserted here: &lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;
Explain the visual concept encoded by this activation inside &lt;explanation&gt; tags.
&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant</pre>
      <p>训练时如果设置 <code>num_injection_tokens=8</code>，脚本会把中间那个 <code>&lt;|image_pad|&gt;</code> 扩展成 8 个连续 <code>&lt;|image_pad|&gt;</code>。activation adapter 是一个 linear layer，把一个 4096-d activation 映射成 <code>8 x 4096</code> 的 injected embeddings。初始化时使用 repeated identity / sqrt(8)，让总能量和单 token 注入大致可比。</p>
      <pre>4096-d activation -> Linear -> 8 x 4096-d injected token embeddings</pre>
      <p>COCO 的 prompt 类似，只是说明文字改成 “selected Qwen3-VL image-token activations from a real COCO image”，强调解释对象是某个 image token 或 local token group。</p>

      <h3>3.6 训练目标</h3>
      <p>AV training 使用 LoRA fine-tuning 加 activation adapter。核心 loss 包含三部分：</p>
      <ul>
        <li><strong>SFT NLL：</strong>给正确 activation 和正确 response，要求模型对 response 的 token-level NLL 低。</li>
        <li><strong>activation-shuffle contrastive loss：</strong>把同一个 response 配上别的样本 activation，要求 wrong activation 的 NLL 至少比 matched activation 更高。</li>
        <li><strong>response-contrastive loss：</strong>给正确 activation 但换成别的样本 response，要求 wrong response 的 NLL 更高。</li>
      </ul>
      <p>文件名里的 <code>8tok_512_dualcontrast</code> 就是这个配置的缩写：<code>8tok</code> 表示 8 个 AV injection tokens；<code>512</code> 表示训练 512 个 synthetic samples；<code>dualcontrast</code> 表示同时用了 activation-shuffle contrastive 和 response-contrastive 两类对比约束。</p>
    </section>

    <section>
      <h2>4. Evaluation metrics：这些数字到底怎么看</h2>
      <h3>NLL 是什么</h3>
      <p>NLL 是 negative log likelihood，也就是在 teacher forcing 下，模型给目标 response 每个 token 的平均负 log probability。越低越好。低 NLL 表示：在给定 prompt 和 injected activation 的情况下，模型认为这段 response 更自然、更可能。</p>
      <p>这里不用 greedy generation 作为主要指标，因为 greedy output 容易受 decoding 和模板 prior 影响。我们更关心：如果把候选解释逐个喂给模型，正确解释在 correct activation 下是否真的更低 NLL。</p>

      <h3>matched / shifted / shuffled 是什么</h3>
      <ul>
        <li><strong>matched NLL：</strong>response 和 activation 来自同一个样本。这应该最低。</li>
        <li><strong>shifted NLL：</strong>response 不变，activation 换成相邻或错位样本。这是 structured mismatch。</li>
        <li><strong>shuffled NLL：</strong>response 不变，activation 随机换成别的样本。这是 random mismatch。</li>
      </ul>
      <p><code>sensitivity delta = shuffled_mean_nll - matched_mean_nll</code>。这个值越大越好。它表示“换错 activation 后，正确 response 变得多难预测”。如果 delta 接近 0，说明模型可能只是学会了 response 模板，而没有真正用 activation。</p>

      <h3>candidate ranking / top5 是什么</h3>
      <p>candidate ranking 的做法是：给定一个 query activation，从一个候选 response pool 中取出 64 或 128 个候选解释，对每个候选都算 teacher-forced NLL，然后按 NLL 从低到高排序。正确 response 排第 1 就是 top1；排进前 5 就是 top5。</p>
      <p>top5 candidate response 不是模型自由生成出来的 5 条，而是候选池中 NLL 最低的 5 条。这个设计可以减少自由生成的不稳定性，但也有一个限制：如果很多样本的 short label 一样，例如多个 <code>person/center</code>，按 sample id 排名会低估语义正确性。</p>

      <h3>activation-gain ranking 是什么</h3>
      <p>activation-gain 是一个 diagnostic score。它会比较 candidate response 在 query activation 下和 reference activation 下的 NLL 差异，试图减掉 response 本身的 language prior。它有助于诊断某些 response 是否只是因为短、常见、模板容易而得分高；但它不是唯一主指标，有时 raw NLL ranking 更稳定。</p>
    </section>

    <section>
      <h2>5. 初步结果与图像解读</h2>
      <h3>5.1 LLaVA-1.5：机制成立，但 AV 还不够强</h3>
      <table>
        <tr><th>run</th><th>target</th><th>AV tokens</th><th>sensitivity delta</th><th>mean rank</th><th>top1</th><th>top5</th></tr>
        <tr><td>LLaVA-1.5 best</td><td>L15 <code>last_prompt</code></td><td>8</td><td>+0.0790</td><td>27.13 / 128</td><td>3.1%</td><td>15.6%</td></tr>
      </table>
      <p>LLaVA 的结论要分开看。机制层面，它非常适合验证 {NLA} migration：<code>&lt;image&gt;</code> 是单 token marker，重复它就可以构造多个 injection positions，activation adapter 也能把一个 layer-15 activation 展开到 4 或 8 个 embeddings。实验上，8 tokens 比 4 tokens 更好，matched-vs-shuffled gap 变大，说明模型确实开始依赖 injected activation。</p>
      <p>但性能层面，LLaVA 离“可靠解释 activation”还很远。top1 只有 3.1%，top5 只有 15.6%，mean rank 仍是 27.13/128。这说明 LLaVA AV 虽然感知 activation，但 candidate selection 仍受 response language prior 和模板偏好影响。这个结果验证了多 token AV 的方向，但也暴露出 LLaVA-1.5 不是最合适的主模型。</p>
      {figures_for("llava15")}

      <h3>5.2 Qwen3-VL synthetic：全图 image-token 表征可以被强力 verbalize</h3>
      <table>
        <tr><th>split</th><th>target</th><th>AV tokens</th><th>sensitivity delta</th><th>mean rank</th><th>top1</th><th>top5</th></tr>
        <tr><td>train pool</td><td>L15 <code>image_mean</code></td><td>8</td><td>+1.1089</td><td>1.02 / 128</td><td>98.4%</td><td>100.0%</td></tr>
        <tr><td>held-out seed</td><td>L15 <code>image_mean</code></td><td>8</td><td>+1.1497</td><td>1.06 / 128</td><td>93.8%</td><td>100.0%</td></tr>
      </table>
      <p>Qwen3-VL 的 synthetic 结果非常强。train pool 上，matched NLL 大约 0.066，而 shuffled NLL 大约 1.175，delta 超过 +1.10。直观理解是：正确 activation 让正确 explanation 非常容易预测；一旦 activation 被换错，response 的 NLL 立刻升高很多。candidate ranking 里正确 response 几乎总是排第 1 或第 2。</p>
      <p>held-out seed 的结果也很强，top1 93.8%，top5 100%。这说明模型不是只背训练样本，而是学到了 synthetic visual activation 与 explanation 之间的映射。当然，<code>image_mean</code> 是所有 image tokens 的平均，更像全图 summary，不是最终最想要的 local interpretability target。它的意义是证明 Qwen3-VL 上 {NLA}-style AV 确实可行，而且明显强于 LLaVA。</p>
      {figures_for("qwen3vl_synthetic")}

      <h3>5.3 Qwen3-VL COCO：specific image token 和 local token group</h3>
      <table>
        <tr><th>held-out short-label target</th><th>sensitivity delta</th><th>matched better</th><th>mean rank</th><th>top1</th><th>top5</th></tr>
        <tr><td>single object-center token</td><td>+0.156</td><td>90.6%</td><td>11.31 / 64</td><td>25.0%</td><td>53.1%</td></tr>
        <tr><td>local bbox token group</td><td>+0.412</td><td>98.4%</td><td>5.19 / 64</td><td>34.4%</td><td>65.6%</td></tr>
      </table>
      <p>COCO 实验是这份报告里最接近真正 research question 的部分，因为它不再解释全图 average，而是解释某个 object region 的 image-token activation。结果说明：single object-center token 不是没有信息，它在 held-out 上 sensitivity delta 仍为 +0.156，matched better 90.6%。但它不稳定，mean rank 11.31/64，top5 53.1%。</p>
      <p>bbox local token group 明显更强：sensitivity delta +0.412，matched better 98.4%，top5 65.6%。这支持一个很重要的假设：在 Qwen3-VL 中，一个 object 的语义通常不是干净地压在一个 image token 上，而是分布在 bbox 内的一小片 tokens 上。因此后续如果想解释真实图像里模型如何理解某个 object，local token group 是比 single token 更稳的基本单位。</p>
      <p>需要特别注意 long-label 和 short-label 的差别。long-label 要求 local token 解释完整 COCO caption，这对局部 token 来说过难，也会把局部 object semantics 和全图 caption 混在一起。short-label 只解释 category + region，更符合“这个/这些 image tokens 被模型理解成什么”的目标。因此本报告把 short-label 结果作为 COCO 主结论。</p>
      {figures_for("qwen3vl_coco")}
    </section>

    <section>
      <h2>6. 阶段性结论：现在到底说明了什么</h2>
      <p>第一，{NLA}-style AV 可以迁移到 LVLM。LLaVA-1.5 和 Qwen3-VL 都支持通过 <code>inputs_embeds</code> 或等价路径把 activation-derived embeddings 注入 language model。重复 special token 并用 adapter 输出多个 injected embeddings 是可行的。</p>
      <p>第二，模型选择很关键。LLaVA-1.5 的结构最方便，但视觉表征和 decoder 行为导致 AV ranking 很弱。Qwen3-VL 的 image-token representation 更适合这条线，尤其是 <code>image_mean</code> 和 object bbox local token group。</p>
      <p>第三，多 token AV 不是 cosmetic change。它让一个 activation 不必被挤进一个 embedding，而是被展开成一小段 token-level information channel。LLaVA 中 8 tokens 优于 4 tokens；Qwen3-VL 中 8-token AV 已经可以取得很强 synthetic performance。</p>
      <p>第四，解释 single image token 是可行但噪声较大。COCO 结果显示 single center token 有语义信号，但 local bbox token group 更稳。这对于后续研究很重要：如果目标是解释模型对一个 object 的内部理解，应该优先解释 object-region token group，再逐步细化到 individual token。</p>
      <p>第五，目前的 evaluation 已经能区分“模型只是会说模板”和“模型真的用了 activation”。matched-vs-shuffled delta、candidate ranking、activation-gain diagnostic 共同说明 Qwen3-VL AV 确实依赖 activation。但完整 {NLA} 还需要 AR 闭环，目前 Qwen3-VL local token 的 AR 尚未完成。</p>
    </section>

    <section>
      <h2>7. 当前局限与风险</h2>
      <h3>7.1 实验规模仍然小</h3>
      <p>synthetic 只有 512 train rows 和 held-out seed；COCO 只有 128 train-like images 和 64 held-out images。这个规模足够做 feasibility，但不能证明方法已经在真实复杂分布上稳定。</p>

      <h3>7.2 image_mean 结果强，但解释粒度偏粗</h3>
      <p>Qwen3-VL synthetic 的最强结果来自 <code>image_mean</code>，它平均了所有 image tokens。这个 target 很适合证明模型视觉表征能被 AV 读出，但它不回答“特定一个 image token 被模型怎么理解”的问题。COCO local-token 实验才是更接近最终目标的方向。</p>

      <h3>7.3 candidate ranking 会受重复 label 和 language prior 影响</h3>
      <p>COCO short-label 中很多 response 语义重复，例如多个样本都是 <code>person/center</code>。按 sample id 排名时，即使模型选中了语义等价的 response，也会被算作 wrong candidate。因此后续需要 unique-label ranking 或 semantic match accuracy。另一方面，NLL 也会偏好短、常见、模板化 response，所以需要 hard-negative 和 activation-gain diagnostic。</p>

      <h3>7.4 long-label 设计混入了全图 caption</h3>
      <p>第一版 COCO response 中加入 “The full COCO caption is...” 是为了测试更严格的局部 token 到全图 caption 映射，但这个目标不够纯。局部 image tokens 不应该被要求恢复完整 caption。后续主线应使用 short label、object attributes、region description、或者人工/模型辅助生成的局部描述。</p>

      <h3>7.5 还缺少完整 AR 与 causal intervention</h3>
      <p>当前主要是 AV evaluation。要更接近原始 {NLA}，需要训练 Qwen3-VL local-token AR，让 explanation text 可以 reconstruct 原 activation，并报告 round-trip error。除此之外，解释本身还需要和 causal intervention 结合，例如 patch/zero-out 对应 image tokens 后观察 answer 和 explanation 是否同步变化。</p>
    </section>

    <section>
      <h2>8. 后续研究方向</h2>
      <h3>方向 A：{HALLUCINATORY_IMAGE_TOKENS} + VLM-{NLA}</h3>
      <p>基于 {EAZY} / {HALLUCINATORY_IMAGE_TOKENS} 的思路，先定位对 hallucinated object 有高影响的 image tokens，再用 VLM-{NLA} 解释这些 tokens。关键问题包括：这些 tokens 的 AV explanation 是否提到了不存在的物体？zero-out 或 patch 这些 tokens 后，hallucination 是否下降？解释文本是否随 intervention 改变？</p>

      <h3>方向 B：VLM shortcut analysis</h3>
      <p>构造 visual evidence 与 background prior 冲突的图片，例如 diagram、spatial relation、counterfactual object layout。比较 original / counterfactual / patched image 下 object-token activation 的 {NLA} explanation。如果模型输出没变、activation explanation 也不跟着视觉证据变，就可能说明模型依赖 shortcut。</p>

      <h3>方向 C：Multi-layer local-token {NLA}</h3>
      <p>把 target 从单层 activation 扩展到多层 tuple，例如 <code>L10 object_bbox_mean</code>、<code>L15 object_bbox_mean</code>、<code>L20 object_bbox_mean</code>。每层分配一组 AV tokens 或一个 layer-specific adapter，观察不同层分别编码 object identity、position、context、answer bias 的程度。AR 端也可以为每层设置 readout head，报告 per-layer reconstruction。</p>

      <h3>方向 D：更公平的 semantic evaluation</h3>
      <p>对 COCO short-label 做 unique-label ranking，把重复 label 合并；把 response 拆成 object category、region、attribute 三部分分别算 accuracy；引入 hard negatives，例如同类 object 不同位置、同位置不同 object、视觉相似但语义不同的 object。这会比现在的 random candidate pool 更能测试解释是否精细。</p>

      <h3>方向 E：从 region token 到 circuit</h3>
      <p>VLM-{NLA} 可以和 {SAE}/<a href="{VLM_CIRCUIT_TRACING_URL}">circuit tracing</a> 结合。先用 {NLA} 给某个 region activation 生成自然语言摘要，再用 {SAE} 或 attribution graph 找到支撑这个摘要的 features 和 attention paths。这样可以把“这个 token group 代表 train/bottom region”进一步拆成“哪些 feature、哪些 layer、哪些 head 支撑了这个表示”。</p>
    </section>

    <section>
      <h2>9. 相关工作与阅读路线</h2>
      <ul>
        <li><a href="{NLA_URL}">Natural Language Autoencoders</a>：本项目的直接出发点，提供 AV/AR 框架。</li>
        <li><a href="{LOGIT_LENS_URL}">Logit Lens</a>：把中间 hidden state 直接投影到词表，适合理解预测如何逐层演化。</li>
        <li><a href="{TUNED_LENS_URL}">Tuned Lens</a>：为每层训练 translator，使中间层预测更可读、更校准。</li>
        <li><a href="{ATTENTION_LENS_URL}">Attention Lens</a> / <a href="{HEADLENS_URL}">HeadLens</a>：分析 attention head 或局部部件对输出 token 的贡献；HeadLens 在 LVLM counting circuits 场景中提出。</li>
        <li><a href="{SAE_URL}">Sparse Autoencoder / Monosemanticity</a>：把 dense activation 分解成稀疏 feature dictionary，是 circuit-level analysis 的重要工具。</li>
        <li><a href="{EAZY_URL}">Hallucinatory Image Tokens / EAZY</a>：提示 hallucination 可能与少数 image tokens 强相关，是 VLM-{NLA} 很自然的下游方向。</li>
        <li><a href="{VLM_SHORTCUT_URL}">Do Vision-Language Models Really Understand Visual Language?</a>：强调 VLM 可能依赖 background knowledge shortcut，而非真正视觉理解。</li>
        <li><a href="{VLM_CIRCUIT_TRACING_URL}">Circuit Tracing in Vision-Language Models</a>：展示 VLM circuit tracing 的更底层路线，可与 {NLA} 的自然语言解释互补。</li>
      </ul>
    </section>

    <section>
      <h2>10. 本地材料</h2>
      <pre>scripts/qwen3vl/                  Qwen3-VL extraction, AV training, evaluation, COCO object-token extraction
scripts/llava15/                  LLaVA-1.5 baseline scripts
results/qwen3vl/synthetic/        Qwen3-VL synthetic image_mean metrics
results/qwen3vl/coco_object_tokens/ COCO object-token metrics
assets/figures/                   source figures; all are also embedded into this HTML
reports/*.md                      exploratory notes
docs/nla_to_vlm_multilayer_design.md multi-layer/multi-token design notes</pre>
    </section>

    <p class="footnote">本文由 <code>tools/build_embedded_report.py</code> 生成。所有 figure 均已嵌入当前 HTML 文件。</p>
  </main>
</body>
</html>
"""
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
