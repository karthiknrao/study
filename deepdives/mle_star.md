# MLE-STAR Implementations

## Overview

MLE-STAR (Machine Learning Engineering via Search and Targeted Refinement) is a multi-agent system from Google Cloud that automates ML pipeline development. It uses web search to find state-of-the-art approaches, then iteratively refines solutions through targeted code block modifications guided by ablation studies. It achieved medals in 64% of Kaggle competitions on MLE-bench Lite.

**Paper**: [MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692)

**Authors**: Jaehyun Nam, Jinsung Yoon, Jiefeng Chen, Jinwoo Shin, Sercan Ö. Arık, Tomas Pfister

**Conference**: NeurIPS 2025

---

## Official Implementations

| Repository | Description |
|------------|-------------|
| [google/adk-samples (MLE)](https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering) | **Official** open-source implementation built with Google's Agent Development Kit (ADK) |
| [jaehyun513/MLE-STAR](https://github.com/jaehyun513/MLE-STAR) | Author's repository with code examples |

---

## Unofficial/Alternative Implementations

| Repository | Description |
|------------|-------------|
| [WalkingDevFlag/MLE-STAR-Open](https://github.com/WalkingDevFlag/MLE-STAR-Open) | **Google-free reimplementation** using OpenAI-compatible LLMs (OpenRouter/Ollama) + DuckDuckGo search |
| [ruvnet/claude-flow (MLE-STAR Workflow)](https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow) | MLE-STAR workflow integrated into Claude-Flow orchestration |
| [sfc-gh-fshu/MLE-STAR](https://github.com/sfc-gh-fshu/MLE-STAR) | Faithful reimplementation for research baseline use |
| [aslansd/MLE-Agent-GoogleADK-Gradio](https://github.com/aslansd/MLE-Agent-GoogleADK-Gradio) | Gradio interface wrapper |
| [robiscoding/mle-star-recipes](https://github.com/robiscoding/mle-star-recipes) | Recipe-based runner for MLE-STAR |
| [zsavmak/machine-learning-engineering](https://github.com/zsavmak/machine-learning-engineering) | Implementation based on the paper |
| [MLSysOps/MLE-agent](https://github.com/MLSysOps/MLE-agent) | Related MLE agent with arxiv/paper integration |

---

## Key Resources

- [Paper (arXiv)](https://arxiv.org/abs/2506.15692)
- [Google Research Blog](https://research.google/blog/mle-star-a-state-of-the-art-machine-learning-engineering-agents/)
- [OpenReview (NeurIPS 2025)](https://openreview.net/forum?id=vS1M06Px6u)
- [Author's Page](https://jaehyun513.github.io/)

---

## Methodology Summary

1. **Web Search**: Leverages search engine to retrieve effective models from the web to form initial solutions
2. **Targeted Refinement**: Iteratively refines solutions by exploring strategies targeting specific ML components
3. **Ablation Studies**: Analyzes impact of individual code blocks to guide exploration
4. **Ensembling**: Novel ensembling method using effective strategies suggested by MLE-STAR
