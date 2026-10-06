# mlplat: why is my GPU waiting?

mlplat watches a machine-learning training run from the operating system's point of view, tells you in plain language what slowed it down, and predicts where the full training run would be fastest and cheapest.

**[Take the 1-minute tour of the live demo →](https://ml-experimentation-platform-demo-lcsqrdpo2ggnamdakqfexg.streamlit.app/tour?step=1)**

## Why I built it

When you train a model, the GPU is the expensive part, so it should always be busy. In practice it often isn't. It sits idle while the CPU is still preparing the next batch of data (loading images, resizing them, augmenting them). Most tools only show you the loss curve, so you can't see the waiting at all.

Students, me included, usually guess where to train: their own GPU, free Colab, Kaggle, or a paid cloud GPU. Sometimes they pay for a faster GPU that doesn't help, because the GPU was never the slow part. I wanted a tool that answers three questions with measurements instead of guesses:

1. **Where did the time go?**
2. **What should I fix first?**
3. **Where should I train the full run, how long will it take, and what will it cost?**

## What it does

- **Measures every process in a training run once a second**: CPU, memory, disk and GPU, including each data-loader worker separately. A small C++ program does this by reading Linux's built-in process information.
- **Explains the result in plain words**, for example: *"Your GPU sat empty for 6.3 seconds of 8.0 seconds, waiting for data."* Behind the sentence is a set of explicit rules with the numbers that triggered them.
- **Predicts other setups** from one short profiling run: other worker counts, other machines, total time and cost.
- **Shows it all in a dashboard** that uses a bakery as the analogy: the GPU is the oven, data-loader workers are bakers preparing dough, and a training step is one tray coming out of the oven.

## Results on a real training run

The numbers below come from a small image-classification model trained on an NVIDIA RTX 4060 with a deliberately slow data pipeline.

| | 1 data-loader worker | 12 data-loader workers | Change |
|---|---|---|---|
| Training steps per second | 27 | 154 | **5.7× faster** |
| Time per step | 36.8 ms | 6.3 ms | |
| GPU busy | 33% | 77% | +45 points |

**What mlplat said about the slow run:** the data-loader worker was 100% busy while the GPU was waiting, so the fix was more workers, not a bigger GPU. After that fix, it said the next bottleneck was the training loop syncing with the GPU after every step. Syncing every 10 steps instead raised it to **169 steps per second with the GPU 87% busy**.

**What the planner predicted for a 20,000-step job:** about 13 minutes as it was, about 3 minutes with the fix, and essentially no gain from a 2× faster rented GPU without the fix, because the GPU wasn't the slow part. These are estimates, and the prices and speeds for other machines are placeholders you fill in.

**How accurate the predictions are:** from one short profiling run with 1 worker, the predicted step times for 6 and 12 workers were within **12% and 25%** of the measured ones (both on the cautious side), on the same machine.

**How much the watching costs:** about **0.15% of one CPU core** on average (0.20% at most) across five runs, with GPU monitoring on, and about 24 MB of memory.

## How it works

```text
your training command
        │
        ▼
  ┌───────────┐   starts training and the agent, stops both cleanly
  │ Launcher  │   (Python)
  └─────┬─────┘
        ▼
  ┌───────────┐   samples every process once a second
  │   Agent   │   (C++17: Linux /proc, pidfd, NVIDIA's NVML)
  └─────┬─────┘
        ▼
  ┌───────────┐   turns samples into a verdict with evidence,
  │ Analyzer  │   and predicts time and cost elsewhere (Python)
  └─────┬─────┘
        ▼
  ┌───────────┐   the story of each run in plain language,
  │ Dashboard │   with the technical details one click away (Streamlit)
  └───────────┘
```

A few details, in plain terms:

- **/proc**: Linux keeps a live record of every running program under `/proc`. The agent reads it directly, so it needs no special permissions and no changes to your training code.
- **pidfd**: a Linux feature that notifies the agent the instant training ends, so the final measurement isn't missed.
- **NVML loaded at runtime**: the NVIDIA monitoring library is only loaded when the agent starts. The same program runs on machines without an NVIDIA GPU and simply records that there's no GPU.

Each part writes plain files (JSON lines) that the next part reads, so every run can be re-analyzed later.

## Try it

**In your browser, no install:** the [live demo](https://ml-experimentation-platform-demo-lcsqrdpo2ggnamdakqfexg.streamlit.app/) shows the real runs from the results above. Start with the [1-minute tour](https://ml-experimentation-platform-demo-lcsqrdpo2ggnamdakqfexg.streamlit.app/tour?step=1), then explore each run's story or compare runs before and after the fix.

**On your own training code:** profiling your own runs needs a local install. You need Linux or WSL2, CMake, a C++17 compiler and Python 3.10 or newer. An NVIDIA GPU is optional.

```bash
cmake -S agent -B agent/build && cmake --build agent/build -j
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,examples]"

mlplat run --name first-run -- python examples/train_image_classifier.py --workers 1 --steps 1500
mlplat analyze

mlplat plan --name where-to-train --total-steps 20000 -- python examples/train_image_classifier.py --workers 1 --steps 100000

mlplat dashboard
```

To have your own training script show up with step timing, add one line inside the loop:

```python
import mlplat
mlplat.log(step=step, loss=loss, data_time=data_time, compute_time=compute_time)
```

`data_time` is how long the loop waited for the next batch. It's optional, but it makes the diagnosis much more precise.

Tests: 44 C++ unit tests, 8 agent integration tests (one on a real GPU when available) and 104 Python tests.

## The rest of the platform

This repository started as a small ML experimentation platform. That part is still here:

- **Spark / Databricks ETL** validates raw data, quarantines bad rows and writes versioned Delta tables.
- **Dataset registry** pins each experiment to an exact dataset version, so results can be reproduced.
- **FastAPI** accepts experiment requests; **Postgres** stores job state; **Kafka** queues training jobs for workers.
- **MLflow** stores parameters, metrics and models. mlplat runs can be mirrored into MLflow too.

```text
Raw data → Spark ETL → versioned Delta table → FastAPI request → Postgres job → Kafka → worker → MLflow
```

## Credits

Bakery art in the dashboard: [Cozy Bakery & Food Icons Pack](https://jimal-art.itch.io/bakery-food-asset-pack-vector) by Jimal. It's used with attribution and isn't included in this repository.

Built by Kirpa Chandok, CS @ Waterloo.
