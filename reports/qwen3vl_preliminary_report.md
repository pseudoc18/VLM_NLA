# Qwen3-VL NLA Preliminary Check

Date: 2026-06-18

Remote workspace:

```bash
/common/users/lc1279/Projects/nla_llava15_experiment
```

## Question

Can we switch the LLaVA-1.5 NLA experiment to Qwen3-VL, and is LLaVA's weaker
visual backbone the reason the AV results were not close to ordinary NLA?

## Mechanism Smoke

Model:

```text
Qwen/Qwen3-VL-8B-Instruct
transformers: 5.6.0.dev0
class: Qwen3VLForConditionalGeneration
text layers: 36
hidden size: 4096
```

Qwen3-VL uses these visual placeholder tokens:

```text
<|vision_start|> id 151652
<|image_pad|>    id 151655
<|vision_end|>   id 151653
```

For a small synthetic image, the processor expanded the image into 64
`<|image_pad|>` tokens. For the 336x336 synthetic dataset, it used 100 image
tokens.

Smoke result:

```text
generated text:
The image displays a green square and a blue triangle side by side, with text below identifying them.

layer15 activation shape: [4096]
layer15 activation norm:  53.81
8-token AV injection positions: [27, 28, 29, 30, 31, 32, 33, 34]
finite original logits: true
finite AV injected logits: true
```

Conclusion: Qwen3-VL supports the NLA `inputs_embeds` route. The mechanism is
not only plausible; it has already passed a forward smoke test.

## Attribute Probe

I extracted 128 synthetic samples at layer 15 and ran the same ridge
activation-to-attributes probe used for LLaVA. The task is to recover
color/shape/position labels from the activation vector.

| model/target | activation | micro-F1 | attribute acc | exact-set acc | note |
|---|---|---:|---:|---:|---|
| LLaVA-1.5 L15 | last_prompt | 0.908 | 0.940 | 0.344 | previous best probe |
| Qwen3-VL L15 | last_prompt | 0.725 | 0.846 | 0.000 | shape/position good, color under-recalled |
| Qwen3-VL L15 | middle image token | 0.776 | 0.848 | 0.125 | position very strong, color still mixed |
| Qwen3-VL L15 | mean of all image tokens | 0.981 | 0.987 | 0.844 | very strong visual representation |

Files:

```bash
outputs/qwen3vl_smoke/qwen3vl_nla_smoke_summary.json
outputs/qwen3vl_attribute_probe_L15_last_prompt_n128.json
outputs/qwen3vl_attribute_probe_L15_image_n128.json
outputs/qwen3vl_attribute_probe_L15_image_mean_n128.json
```

Local copies:

```bash
outputs/qwen3vl_experiment
```

## Interpretation

The result partly supports the hypothesis, but with an important correction.

It is not simply that LLaVA's raw visual ability is too weak. In the previous
LLaVA run, a simple probe could already decode visual attributes from layer-15
last-prompt activations quite well.

The bigger architectural difference is where the visual information lives:

```text
LLaVA-1.5:
  image features are projected and replace one long block of <image> tokens,
  and last_prompt worked reasonably well as a compact target.

Qwen3-VL:
  image information is distributed across many <|image_pad|> tokens.
  A single last_prompt or single image token is not the best target.
  Mean-pooled image-token activation is extremely decodable.
```

So for Qwen3-VL, the right NLA target probably should not be "one last prompt
activation." A better first target is either:

```text
1. mean-pooled image-token activation at layer 15, or
2. multiple image-token activations / chunks mapped into multiple AV tokens.
```

This also connects directly to the earlier "multiple AV special tokens" idea:
Qwen3-VL naturally gives us 64 to 100 visual placeholder tokens, so NLA on
Qwen3-VL should use a multi-token activation representation from the start.

## Next Step

The next real AV experiment should be:

```text
Qwen3-VL L15 image_mean -> 8 or 16 <|image_pad|> AV tokens -> explanation text
```

Then compare against LLaVA's best run:

```text
LLaVA 8-token dual contrastive:
  raw mean rank: 27.13 / 128
  raw top5:      15.6%
```

If Qwen3-VL's much stronger `image_mean` probe translates into AV training, it
should beat the LLaVA ranking baseline. If it does not, then the main bottleneck
is not the vision backbone; it is the AV decoder/training objective.
