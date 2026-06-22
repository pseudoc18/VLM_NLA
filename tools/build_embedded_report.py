from __future__ import annotations

import base64
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "assets" / "figures"
OUT_PATH = ROOT / "reports" / "vlm_nla_research_proposal_preliminary_results.html"


CAPTIONS = {
    "experiment4_metric_comparison.png": (
        "LLaVA-1.5 baseline: 8 AV tokens improve activation sensitivity, but ranking remains weak.",
        "LLaVA-1.5 baseline：8 个 AV tokens 提升了 activation sensitivity，但 candidate ranking 仍然不强。",
    ),
    "8tok_512_dualcontrast_synthetic_00001.png": (
        "A LLaVA qualitative case: the model uses the injected activation, but exact decoding is still dominated by language priors.",
        "一个 LLaVA qualitative case：模型确实用到了 injected activation，但 exact decoding 仍容易被 language prior 影响。",
    ),
    "qwen3vl_layer15_probe_targets.png": (
        "Qwen3-VL target probe: image_mean is much easier to decode than last_prompt or a single image token on synthetic concepts.",
        "Qwen3-VL target probe：在 synthetic concepts 上，image_mean 比 last_prompt 或 single image token 更容易 decode。",
    ),
    "qwen3vl_llava_metric_comparison.png": (
        "Qwen3-VL versus LLaVA-1.5: Qwen3-VL reaches much stronger activation-conditioned ranking.",
        "Qwen3-VL 对比 LLaVA-1.5：Qwen3-VL 的 activation-conditioned ranking 明显更强。",
    ),
    "qwen3vl_raw_synthetic_00000.png": (
        "Qwen3-VL synthetic example: teacher-forced ranking can be correct even when greedy generation is imperfect.",
        "Qwen3-VL synthetic example：即使 greedy generation 不完美，teacher-forced ranking 仍可能正确。",
    ),
    "qwen3vl_raw_synthetic_00001.png": (
        "Qwen3-VL synthetic candidate ranking example.",
        "Qwen3-VL synthetic candidate ranking 示例。",
    ),
    "qwen3vl_raw_synthetic_00002.png": (
        "Qwen3-VL synthetic visual concept example.",
        "Qwen3-VL synthetic visual concept 示例。",
    ),
    "qwen3vl_raw_synthetic_00005.png": (
        "Qwen3-VL synthetic example with activation-conditioned candidate scores.",
        "Qwen3-VL synthetic 示例，展示 activation-conditioned candidate scores。",
    ),
    "qwen3vl_heldout_raw_synthetic_00000.png": (
        "Held-out synthetic example for Qwen3-VL.",
        "Qwen3-VL held-out synthetic 示例。",
    ),
    "qwen3vl_heldout_raw_synthetic_00001.png": (
        "Held-out synthetic example where the best wrong candidate is semantically close.",
        "一个 held-out synthetic 示例：top wrong candidate 与正确答案语义很接近。",
    ),
    "qwen3vl_gain_synthetic_00005.png": (
        "Activation-gain scoring is useful as a diagnostic, but does not always match raw NLL ranking.",
        "Activation-gain scoring 是有用的 diagnostic，但不总是等同于 raw NLL ranking。",
    ),
    "qwen3vl_coco_token_metrics.png": (
        "COCO object-token metrics: bbox-local token groups beat single center tokens.",
        "COCO object-token metrics：bbox-local token groups 明显强于 single center token。",
    ),
    "bbox_train_0_coco_000000474021_445486.png": (
        "COCO bbox-token group example from the train-like split.",
        "COCO train-like split 中的 bbox-token group 示例。",
    ),
    "bbox_train_1_coco_000000455981_1093311.png": (
        "COCO bbox-token group example: red box is the object bbox; blue cells are explained image tokens.",
        "COCO bbox-token group 示例：红框是 object bbox，蓝色格子是被解释的 image tokens。",
    ),
    "bbox_heldout_0_coco_000000042563_172991.png": (
        "Held-out COCO bbox-token group example.",
        "Held-out COCO bbox-token group 示例。",
    ),
    "center_train_0_coco_000000474021_445486.png": (
        "COCO single object-center token example from the train-like split.",
        "COCO train-like split 中的 single object-center token 示例。",
    ),
    "center_train_1_coco_000000455981_1093311.png": (
        "COCO center-token example corresponding to the same object as the bbox-token view.",
        "COCO center-token 示例，与 bbox-token view 对应同一个 object。",
    ),
    "center_heldout_0_coco_000000042563_172991.png": (
        "Held-out COCO center-token example.",
        "Held-out COCO center-token 示例。",
    ),
}


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def figure_html(path: Path) -> str:
    cap_en, cap_zh = CAPTIONS.get(
        path.name,
        (
            path.name.replace("_", " ").replace(".png", ""),
            path.name.replace("_", " ").replace(".png", ""),
        ),
    )
    rel = path.relative_to(ROOT)
    return f"""
      <figure>
        <img src="{data_uri(path)}" alt="{html.escape(path.stem)}">
        <figcaption>
          <strong>{html.escape(cap_en)}</strong><br>
          <span>{html.escape(cap_zh)}</span><br>
          <code>{html.escape(str(rel))}</code>
        </figcaption>
      </figure>
    """


def figures_for(subdir: str) -> str:
    paths = sorted((FIG_ROOT / subdir).glob("*.png"))
    return "\n".join(figure_html(path) for path in paths)


def main() -> None:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM-NLA Research Proposal and Preliminary Results</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5b6470;
      --line: #d9dee7;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --accent: #1b6fb8;
      --accent-2: #1f8a70;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 16px/1.58 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans", "Helvetica Neue", sans-serif;
    }}
    header {{
      background: #101820;
      color: white;
      padding: 56px 24px 36px;
    }}
    header .wrap, main {{
      max-width: 1080px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(34px, 6vw, 64px);
      line-height: 1.04;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 42px 0 14px;
      font-size: 28px;
      line-height: 1.2;
      border-bottom: 2px solid var(--line);
      padding-bottom: 8px;
    }}
    h3 {{
      margin: 28px 0 8px;
      font-size: 20px;
      line-height: 1.25;
      color: #123c5c;
    }}
    main {{
      padding: 28px 24px 60px;
      background: var(--panel);
    }}
    p {{ margin: 10px 0; }}
    a {{ color: var(--accent); }}
    .subtitle {{
      max-width: 860px;
      margin: 0;
      font-size: 18px;
      color: #d9e3ee;
    }}
    .meta {{
      margin-top: 22px;
      color: #aebdcc;
      font-size: 14px;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #eef6fc;
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .callout.green {{
      border-left-color: var(--accent-2);
      background: #effaf6;
    }}
    .zh {{
      color: #28313d;
      background: #fafbfc;
      border-left: 3px solid #ccd6e2;
      padding: 10px 12px;
      margin: 10px 0 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 22px;
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
    li {{ margin: 6px 0; }}
    .footnote {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 32px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>VLM-NLA Research Proposal and Preliminary Results</h1>
      <p class="subtitle">A bilingual, plain-language report on adapting Natural Language Autoencoders to LLaVA-1.5 and Qwen3-VL internal activations.</p>
      <p class="meta">Generated 2026-06-22. All figures are embedded as base64 data URIs; this HTML file is self-contained.</p>
    </div>
  </header>
  <main>
    <section>
      <h2>1. Executive Summary / 摘要</h2>
      <div class="callout">
        <p><strong>Main finding.</strong> NLA-style <code>activation -> text</code> verbalization is feasible for VLMs. LLaVA-1.5 proves the mechanism, but Qwen3-VL gives much stronger performance. On real COCO images, a local group of image tokens inside an object bbox is more interpretable and more stable than a single center image token.</p>
      </div>
      <div class="zh">
        <p><strong>一句话结论。</strong> 把 NLA 用到 VLM 上是可行的。LLaVA-1.5 证明了机制可以跑通，但 Qwen3-VL 的效果明显更好。在真实 COCO 图片上，解释 object bbox 里面的一小组 image tokens，比只解释单个 center image token 更稳定。</p>
      </div>
      <p>The current system is not a final NLA checkpoint. It is a research prototype showing that visual activations can be injected through multiple AV special tokens and evaluated with matched-vs-shuffled sensitivity and candidate ranking.</p>
      <div class="zh">
        <p>现在这套东西还不是最终版本的 NLA checkpoint，而是一个 research prototype。它说明：visual activation 可以通过多个 AV special tokens 注入模型，并且可以用 matched-vs-shuffled sensitivity 和 candidate ranking 来评估解释是否真的依赖 activation。</p>
      </div>
    </section>

    <section>
      <h2>2. Motivation / 为什么做这个</h2>
      <p>A VLM can answer a question about an image, but the output alone does not tell us which internal visual tokens the model used, which tokens were ignored, or whether the answer came from the image or from a shortcut in background knowledge.</p>
      <div class="zh">
        <p>一个 VLM 可以回答图片问题，但只看输出文本，我们不知道模型到底用了哪些 internal visual tokens，也不知道哪些 tokens 被忽略了，更不知道答案是真的来自 image evidence，还是来自 background knowledge shortcut。</p>
      </div>
      <p>NLA gives a natural interface for this problem: make an internal activation "speak" in natural language, then test whether that explanation preserves enough information to reconstruct or select the original activation.</p>
      <div class="zh">
        <p>NLA 提供了一个很自然的接口：让一个 internal activation 用 natural language 说出自己代表什么，然后检验这段 explanation 是否真的保留了 activation 里的信息。</p>
      </div>
    </section>

    <section>
      <h2>3. Background / 背景</h2>
      <h3>Natural Language Autoencoder</h3>
      <p>Original NLA has two directions: <strong>AV</strong> maps an activation vector to text, and <strong>AR</strong> maps the text back to an activation vector. The round trip asks whether the explanation captures the information in the vector rather than merely sounding plausible.</p>
      <div class="zh">
        <p>原始 NLA 有两个方向：<strong>AV</strong> 把 activation vector 变成 text，<strong>AR</strong> 把 text 再 reconstruct 回 activation vector。这个 round trip 的意义是：不是看 explanation 好不好听，而是看它有没有真正保留 vector 里的信息。</p>
      </div>
      <h3>Why VLMs are a good target</h3>
      <p>In many VLMs, visual features eventually enter the language model as token-like embeddings. That makes VLMs a natural place to try NLA: replace special visual placeholder embeddings with activation-derived embeddings and ask the language model to verbalize them.</p>
      <div class="zh">
        <p>很多 VLM 最终都会把 visual features 转成类似 token embedding 的东西，再送进 language model。这和 NLA 的 input-embeds injection 非常接近：我们可以把 special visual placeholder embeddings 换成 activation-derived embeddings，然后让 language model 解释它们。</p>
      </div>
    </section>

    <section>
      <h2>4. Method / 方法</h2>
      <div class="grid">
        <div class="mini"><strong>Activation target</strong><br>Layer-15 hidden state from LLaVA-1.5 or Qwen3-VL.</div>
        <div class="mini"><strong>AV injection</strong><br>Map one activation to <code>K</code> special token embeddings, usually <code>K=8</code>.</div>
        <div class="mini"><strong>Training loss</strong><br>SFT loss plus activation-shuffle and response-contrastive losses.</div>
        <div class="mini"><strong>Evaluation</strong><br>Matched-vs-shuffled NLL sensitivity and teacher-forced candidate ranking.</div>
      </div>
      <p>For Qwen3-VL, the injection prompt uses visual placeholders:</p>
      <pre>&lt;|vision_start|&gt;&lt;|image_pad|&gt; x 8 &lt;|vision_end|&gt;</pre>
      <p>The activation adapter maps one 4096-dimensional activation into eight 4096-dimensional injected embeddings:</p>
      <pre>4096-d activation -> 8 x 4096-d injected embeddings</pre>
      <div class="zh">
        <p>通俗说，就是不要把一个 activation 硬塞进一个 token 里，而是把它展开成 8 个 special token embeddings。这样 AV decoder 有更宽的 bottleneck，可以更好地读出 activation 里的信息。</p>
      </div>
    </section>

    <section>
      <h2>5. Preliminary Results / 初步结果</h2>
      <h3>5.1 LLaVA-1.5: mechanism works, performance is weak</h3>
      <table>
        <tr><th>Run</th><th>Target</th><th>AV tokens</th><th>Sensitivity delta</th><th>Mean rank</th><th>Top-1</th><th>Top-5</th></tr>
        <tr><td>LLaVA-1.5 best</td><td>L15 <code>last_prompt</code></td><td>8</td><td>+0.0790</td><td>27.13 / 128</td><td>3.1%</td><td>15.6%</td></tr>
      </table>
      <p>LLaVA-1.5 is easy to instrument because it uses a single-token <code>&lt;image&gt;</code> marker. Repeating that marker gives multiple AV injection sites. However, even with 8 tokens and contrastive training, exact candidate ranking remains weak.</p>
      <div class="zh">
        <p>LLaVA-1.5 很适合作为第一步，因为它有单 token 的 <code>&lt;image&gt;</code> marker，重复这个 marker 就能得到多个 AV injection sites。但即使使用 8 个 tokens 和 contrastive training，exact candidate ranking 还是不够强。</p>
      </div>
      {figures_for("llava15")}

      <h3>5.2 Qwen3-VL synthetic: strong global visual activation verbalization</h3>
      <table>
        <tr><th>Split</th><th>Target</th><th>AV tokens</th><th>Sensitivity delta</th><th>Mean rank</th><th>Top-1</th><th>Top-5</th></tr>
        <tr><td>Train pool</td><td>L15 <code>image_mean</code></td><td>8</td><td>+1.1089</td><td>1.02 / 128</td><td>98.4%</td><td>100.0%</td></tr>
        <tr><td>Held-out seed</td><td>L15 <code>image_mean</code></td><td>8</td><td>+1.1497</td><td>1.06 / 128</td><td>93.8%</td><td>100.0%</td></tr>
      </table>
      <p>Qwen3-VL is much stronger. The global <code>image_mean</code> target is not the final interpretability target, but it proves that Qwen3-VL visual activations can drive faithful natural-language selection.</p>
      <div class="zh">
        <p>Qwen3-VL 明显更强。<code>image_mean</code> 是全图 average，不是最终我们最想解释的 local token target，但它证明了 Qwen3-VL visual activation 确实可以控制 natural-language explanation。</p>
      </div>
      {figures_for("qwen3vl_synthetic")}

      <h3>5.3 Qwen3-VL COCO: specific image tokens and local token groups</h3>
      <table>
        <tr><th>Held-out short-label target</th><th>Sensitivity delta</th><th>Matched better</th><th>Mean rank</th><th>Top-1</th><th>Top-5</th></tr>
        <tr><td>Single object-center token</td><td>+0.156</td><td>90.6%</td><td>11.31 / 64</td><td>25.0%</td><td>53.1%</td></tr>
        <tr><td>Local bbox token group</td><td>+0.412</td><td>98.4%</td><td>5.19 / 64</td><td>34.4%</td><td>65.6%</td></tr>
      </table>
      <p>The COCO result is more important for interpretability than the synthetic <code>image_mean</code> result. It shows that we can target a specific object region and ask what those image tokens mean. Single tokens work, but local bbox token groups are clearly more stable.</p>
      <div class="zh">
        <p>COCO 结果比 synthetic <code>image_mean</code> 更接近真正的 interpretability 问题。它说明我们可以盯着某个 object region，问这片 image tokens 在模型内部代表什么。单个 token 有信号，但 bbox 里的 local token group 明显更稳。</p>
      </div>
      {figures_for("qwen3vl_coco")}
    </section>

    <section>
      <h2>6. Interpretation / 现在怎么理解</h2>
      <p>The current evidence supports a layered interpretation:</p>
      <ul>
        <li>LLaVA-1.5 confirms that the NLA injection mechanism transfers to VLMs.</li>
        <li>Qwen3-VL confirms that a stronger VLM can support high-quality activation-conditioned AV.</li>
        <li>COCO object-token results show that local visual semantics are distributed across small image-token neighborhoods.</li>
      </ul>
      <div class="zh">
        <p>现在可以分三层理解：LLaVA-1.5 证明机制能迁移；Qwen3-VL 证明强 VLM 上 activation-conditioned AV 可以很强；COCO object-token 说明 local visual semantics 通常分布在一小片 image-token neighborhood 里，而不是一个 token 干净地对应一个 object。</p>
      </div>
      <p>The strongest statement we can make today is:</p>
      <div class="callout green">
        <p><strong>Qwen3-VL image-token activations can be verbalized in an NLA-style setup, and local multi-token object targets are a promising path toward explaining what VLMs internally understand from specific image regions.</strong></p>
      </div>
    </section>

    <section>
      <h2>7. Limitations / 当前限制</h2>
      <ul>
        <li>These are small research runs, not large-scale released checkpoints.</li>
        <li>The strongest synthetic result uses global <code>image_mean</code>, which is useful but less localized.</li>
        <li>COCO short-label ranking underestimates semantic accuracy when multiple samples share the same label, such as <code>person/center</code>.</li>
        <li>AR is not yet fully integrated for Qwen3-VL local token targets.</li>
        <li>Multi-layer AV/AR has a clear design but has not yet been fully run.</li>
      </ul>
      <div class="zh">
        <p>限制也要说清楚：目前还是小样本实验；synthetic 最强结果是 global <code>image_mean</code>；COCO short-label 有重复 label，所以 sample-id ranking 会低估 semantic accuracy；Qwen3-VL local token 的完整 AR 闭环还没做；multi-layer AV/AR 还只是设计清楚，没完整跑完。</p>
      </div>
    </section>

    <section>
      <h2>8. Proposal / 下一步研究计划</h2>
      <h3>A. Hallucinatory Image Tokens + VLM-NLA</h3>
      <p>The EAZY / Hallucinatory Image Tokens line argues that a small number of high-impact image tokens can drive object hallucinations. VLM-NLA can add a missing piece: instead of only detecting or zeroing those tokens, verbalize them.</p>
      <div class="zh">
        <p>EAZY / Hallucinatory Image Tokens 这条线说明：少数 high-impact image tokens 可能直接驱动 object hallucination。VLM-NLA 可以补上一块：不只是 detect 或 zero-out 这些 tokens，而是让这些 tokens 说出它们在模型内部像什么。</p>
      </div>
      <p>Concrete experiments:</p>
      <ul>
        <li>Generate hallucinated object responses on COCO/Hall-COCO style data.</li>
        <li>Use attention or EAZY-style interventions to select candidate Hallucinatory Image Tokens.</li>
        <li>Run VLM-NLA on those tokens before and after zero-out/patching.</li>
        <li>Ask whether AV explanations mention hallucinated object concepts that are absent from the image.</li>
      </ul>

      <h3>B. VLM shortcut analysis</h3>
      <p>The diagram-understanding and shortcut-learning papers motivate a different question: does the VLM answer from visual evidence, or from memorized/background knowledge?</p>
      <div class="zh">
        <p>VLM shortcut 方向问的是另一个问题：模型的答案到底来自 image evidence，还是来自 memorized/background knowledge？这和 diagram reasoning、spatial relation、OCR-like layouts 特别相关。</p>
      </div>
      <p>Concrete experiments:</p>
      <ul>
        <li>Create paired images where visual evidence conflicts with common priors.</li>
        <li>Extract activations from object tokens, relation tokens, and final answer tokens.</li>
        <li>Use VLM-NLA to compare explanations under original, counterfactual, and patched images.</li>
        <li>Measure whether internal explanations track the visual edit or stay fixed to the shortcut prior.</li>
      </ul>

      <h3>C. Multi-layer local-token NLA</h3>
      <p>Move from one activation vector to activation tuples such as <code>L10/L15/L20 object_bbox_mean</code>. Give each layer its own AV token block or adapter head, then compare which layer carries object identity, spatial position, and hallucination-prone concepts.</p>
      <div class="zh">
        <p>把目标从一个 activation vector 扩展到 activation tuple，例如 <code>L10/L15/L20 object_bbox_mean</code>。每一层可以有自己的 AV token block 或 adapter head，然后比较哪一层更携带 object identity、spatial position、以及 hallucination-prone concepts。</p>
      </div>

      <h3>D. Better evaluation</h3>
      <ul>
        <li>Use unique-label ranking for repeated COCO labels.</li>
        <li>Add semantic match metrics with object/category/region decomposition.</li>
        <li>Mine hard negatives from the current model's top wrong candidates.</li>
        <li>Add listwise ranking loss instead of only one shuffled response.</li>
        <li>Train AR for Qwen3-VL object-token targets and report round-trip reconstruction.</li>
      </ul>
    </section>

    <section>
      <h2>9. Related Work / 相关工作</h2>
      <ul>
        <li><a href="https://transformer-circuits.pub/2026/nla/index.html">Natural Language Autoencoders</a>: the original AV/AR framework for text-only LLM activations.</li>
        <li><a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Che_Hallucinatory_Image_Tokens_A_Training-free_EAZY_Approach_to_Detecting_and_ICCV_2025_paper.pdf">EAZY / Hallucinatory Image Tokens</a>: studies image tokens tied to hallucination behavior and motivates token-level hallucination diagnostics.</li>
        <li><a href="https://openreview.net/forum?id=ZPQU4uGMBA">Do Vision-Language Models Really Understand Visual Language?</a>: shows that diagram-reasoning performance can come from background-knowledge shortcuts rather than genuine relation understanding.</li>
        <li><a href="https://arxiv.org/abs/2602.20330">Circuit Tracing in Vision-Language Models</a>: studies VLM internal mechanisms with transcoders, attribution graphs, attention methods, feature steering, and circuit patching.</li>
      </ul>
      <div class="zh">
        <p>这些相关工作和 VLM-NLA 的关系是：NLA 给出 activation-to-language 的接口；EAZY 给出 hallucination image tokens 的候选目标；shortcut work 给出要诊断的行为现象；circuit tracing 给出更底层的 causal/mechanistic analysis 路线。</p>
      </div>
    </section>

    <section>
      <h2>10. Local Artifacts / 本地材料</h2>
      <pre>scripts/qwen3vl/                  Qwen3-VL extraction, AV training, evaluation, COCO token extraction
scripts/llava15/                  LLaVA-1.5 baseline scripts
results/qwen3vl/synthetic/        lightweight JSON metrics for synthetic image_mean runs
results/qwen3vl/coco_object_tokens/ lightweight JSON metrics for COCO local token runs
assets/figures/                   source figures also embedded in this HTML
reports/*.md                      exploratory notes
docs/nla_to_vlm_multilayer_design.md multi-layer/multi-token design notes</pre>
      <p>This HTML file embeds all figures from <code>assets/figures/</code>. The image files are also kept separately in the repository for README/GitHub browsing.</p>
      <div class="zh">
        <p>这个 HTML 已经内嵌了 <code>assets/figures/</code> 里的所有图。图片文件本身也保留在 repo 里，方便 GitHub README 或单独查看。</p>
      </div>
    </section>

    <p class="footnote">End of report. This file is generated by <code>tools/build_embedded_report.py</code>.</p>
  </main>
</body>
</html>
"""
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
