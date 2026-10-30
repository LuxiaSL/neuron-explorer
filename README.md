# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## kotodama-tsumugi backend

The explorer also runs on [kotodama-tsumugi](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b), a 1.4B base model
with looped depth (52 unique layers run as 72 effective positions; 39–58 reuse 19–38) and block attention residuals.
Code: [github.com/anima-research/kotodama](https://github.com/anima-research/kotodama).

    git clone https://github.com/anima-research/kotodama ../kotodama
    uv sync && uv pip install -e ../kotodama                  # Python >= 3.12
    export EXPLORER_BACKEND=koto EXPLORER_KOTO_CKPT=hf:LuxiaSL/kotodama-tsumugi-1b
    uv run python explorer/build_connections_koto.py --out koto-data/cache/connections.npz   # once
    uv run python explorer/server.py                          # http://localhost:8791

`EXPLORER_KOTO_CKPT` takes `hf:REPO[@REVISION]` (weights from the Hub, e.g. a step branch of
[the training trajectory](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-trajectory)) or a local kotodama
checkpoint. The kotodama backend keeps its caches, runs, stats and names under `koto-data/`
(`EXPLORER_KOTO_DATA`), apart from the shipped Qwen data. The first start builds the lens index and activation scales.
`EXPLORER_KOTO_SIZE` picks the preset (default `1b-tsumugi`).

What differs from Qwen3.5 (details in `explorer/runtime_koto.py`):
- **Layer = effective position.** A looped position shares its weights with the core, so neuron n at 25 and at 45 is
  the same weights on pass 1 and pass 2, with different activations. Reads and pushes are per pass; to push "the
  neuron" in both passes, push both positions.
- **Dose unit:** the median norm of the state entering the position's FFN (attention residuals have no single residual
  stream).
- **Connections** read through each position's own norm and ignore the input-dependent residual routing (weight-only
  leads). Pass-1 → pass-2 links, including a neuron to its own second-pass self, show up.
- **Prompts are raw text** (a base model; no chat template).
- **Position 0** carries massive-activation units (≈10²–10⁴ at the first token, 0 after), so reads, fires and
  activation views leave it out (`EXPLORER_SKIP_POS0=1`, default); reads keep the raw value as `pos0_act`.
- **Steered chat** has a sampling panel (temperature, top-p, repetition and frequency penalty, seed) with tsumugi's
  defaults. All three columns share one seed, so they differ only by the push.
- `GET /api/debug` shows the live model's current op and lock state.
- **The in-page Claude** (and any agent driving the explorer) gets `explorer/agent_prompt_koto.md` on this backend: the
  model's shape, its two looped passes, base-model prompting, the tools that exist here, and the binding questions it
  is studied for. It adapts the Qwen prompt's method; `agent_prompt.local.md` still appends.

**The tsumugi neuron compendium** ships in `koto/`: `compendium-tsumugi-release.json` (what the **lib** tab loads on
the kotodama backend until a library is uploaded with `POST /api/compendium`) and a readable
`COMPENDIUM-tsumugi-release.md`. It holds the named neurons found so far on the release checkpoint (each with its
pole, both loop positions where it matters, what it fires on, steering examples, a J-lens verdict and a confidence
mark), multi-neuron recipes to try, and the imposters: neurons that fired on a concept without carrying it. Every entry
was found with the method in `agent_prompt_koto.md` (matched contrasts, steering on raw openings, break tests, pass 1
vs pass 2) and spot-checked on a fresh opening.

**Draft names** come from an automatic naming pass over every neuron (clusters of neurons, each with a category and an
agreement grade). They are read-only and sit under the hand names: where a neuron has no hand name, lists, search
results, tooltips and connections show its draft, muted behind a `draft` chip, and the neuron panel shows the draft's
cluster, category, family, agreement and label apart from the editable name. A hand name always wins. Lens search also
matches hand names, draft labels and cluster names (`src` `name` / `draft`). The file is `EXPLORER_DRAFTS` (default
`drafts.json` beside the names file, so `koto-data/drafts.json` on this backend), re-read when it changes; without it
the explorer shows no drafts:

    {"meta": {...},                         # free-form: how the pass was made
     "neurons": {"L25/N1234": {"cluster": 17, "cluster_name": "...", "category": "...", "family": "...",
                               "agreement": "high", "label": "..."}}}     # agreement high | medium | low; label may be null

`/api/units` rows and `/api/unit` carry the entry as `draft` (null for none); Claude's tools see it too.

**The measured neuron graph** lists, for each neuron, the earlier neurons that drive it most on real text and the
later neurons that it drives. An edge is activation × gradient summed over a text corpus, and it counts every path
between the two neurons. The weight-only connections, by contrast, follow the direct path alone. For kotodama-tsumugi
the graph is the public dataset
[LuxiaSL/kotodama-tsumugi-1b-neuron-graph](https://huggingface.co/datasets/LuxiaSL/kotodama-tsumugi-1b-neuron-graph);
its card defines every array. To turn it on, download `graph.safetensors` and point `EXPLORER_GRAPH` at it:

    huggingface-cli download --repo-type dataset LuxiaSL/kotodama-tsumugi-1b-neuron-graph graph.safetensors --local-dir koto-data
    export EXPLORER_GRAPH=koto-data/graph.safetensors

Without the variable, or with a file that is missing, unreadable or measured on another shape of model, the explorer
shows no graph. The file is memory-mapped, and a neuron's rows are read only when that neuron is opened. The neuron
panel's Details tab then shows its 12 strongest inputs and outputs. Each one is a signed share of the neuron's total
input or output, and each one opens that neuron on click. A bar shows how much the listed edges carry together. A badge
gives how well the input list reproduces across two halves of the corpus (split-half ρ: ≥ .8, .6–.8, < .6). An edge
too large for the file's storage is shown as at least the largest finite edge in its list (`≥`). `GET /api/graph?layer=
&n=&k=` returns the same data, and Claude's `neuron_graph` tool reads it.

The attention-head tools (head slabs on the map, head rows and patterns, the agent's head tools) read the Qwen
backend's head captures and answer "not available" on kotodama. The sweep (map, list and embeddings views) is not
ported to kotodama yet; neuron pages, steered chat, lenses,
connections, `fires` and live inference work. Serving beyond localhost: `EXPLORER_HOST`, `EXPLORER_ALLOW_HOSTS` and
`EXPLORER_CLIENTS` allowlist hosts and client addresses.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.
