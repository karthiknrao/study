# Deep Research: A Systematic Survey - Analysis

## 1. Survey Scope & Motivation

### Precise Scope
**Included:**
- End-to-end research workflows where LLMs act as autonomous agents
- Four core components: Query Planning, Information Acquisition, Memory Management, Answer Generation
- Three capability phases: Agentic Search → Integrated Research → Full-stack AI Scientist

**Explicitly Excluded:**
- Standard RAG techniques (treated as a distinct, simpler paradigm)
- Static, single-shot prompting approaches
- Web agents without research-oriented synthesis capabilities

### Why This Survey Is Needed Now
- **Catalyst**: Rapid emergence of commercial "Deep Research" products (OpenAI DeepResearch, Gemini DeepResearch, Perplexity Deep Research) in late 2024-2025
- **Gap**: No prior comprehensive survey systematically analyzing DR's technical landscape
- **Confusion**: Field lacks unified terminology across rapidly proliferating systems

---

## 2. Taxonomy & Organization

```
Deep Research System
├── Query Planning
│   ├── Parallel Planning (one-shot decomposition)
│   ├── Sequential Planning (iterative, dependency-aware)
│   └── Tree-based Planning (MCTS, DAG exploration)
├── Information Acquisition
│   ├── Retrieval Tools (text, multimodal)
│   ├── Retrieval Timing (adaptive strategies)
│   └── Information Filtering (selection, compression)
├── Memory Management
│   ├── Consolidation, Indexing, Updating, Forgetting
│   └── Graph-based, timeline-based approaches
└── Answer Generation
    ├── Evidence integration, coherence maintenance
    └── Multimodal presentation (slides, videos)
```

---

## 3. Historical Evolution

| Year | Milestone | Significance |
|------|-----------|--------------|
| 2022 | WebGPT | First LLM trained to browse web with human feedback |
| 2022 | ReAct | Unifies reasoning + acting |
| 2023 | Toolformer | Self-supervised tool use learning |
| 2024 | Search-R1 | First end-to-end RL for search-augmented reasoning |
| 2025 | OpenAI/Gemini DeepResearch | Commercial deployment |

**Paradigm Shifts:** Static RAG → Adaptive Retrieval → Multi-agent orchestration → RL-optimized planning

---

## 4. Core Concepts (Key Papers)

| Component | Key Papers | Current State |
|-----------|-----------|---------------|
| **Query Planning** | Least-to-Most Prompting, Search-R1, RAG-Star | Active (RL approaches leading) |
| **Information Acquisition** | Self-RAG, ColBERT-v2, HtmlRAG | Mature for text, active for multimodal |
| **Memory Management** | HippoRAG, MemGPT, Memory-R1 | Active (graph-based emerging) |
| **Answer Generation** | LongWriter, RAPID, RioRAG | Active (multimodal is frontier) |

### Category Details

#### Query Planning
- **Definition**: Transforming complex questions into structured sequences of executable sub-queries
- **Technical Approach**: Parallel (independent), Sequential (dependency-aware), Tree-based (MCTS)
- **Strengths**: Enables multi-step reasoning; handles complex queries
- **Limitations**: Parallel ignores dependencies; sequential has latency; tree-based is complex

#### Information Acquisition
- **Definition**: Retrieving and filtering external information via tools/APIs
- **Technical Approach**: Lexical (BM25), semantic (dense encoders), commercial web search
- **Strengths**: Accesses up-to-date information; grounds outputs in evidence
- **Limitations**: Retrieved content may be noisy; API cost/latency

#### Memory Management
- **Definition**: Governing lifecycle of task-solving context across long-horizon investigations
- **Technical Approach**: Consolidation, indexing, updating, forgetting
- **Strengths**: Long-horizon coherence; continual learning support
- **Limitations**: Long-term credit assignment is challenging

#### Answer Generation
- **Definition**: Synthesizing accumulated evidence into coherent, verifiable responses
- **Technical Approach**: Integrate upstream info, resolve conflicts, structure narrative
- **Strengths**: Comprehensive, citeable outputs; logical coherence
- **Limitations**: Long-form coherence challenging; novelty vs. hallucination boundary unclear

---

## 5. Comparative Analysis

| Dimension | Standard RAG | Agentic Search | Integrated Research | AI Scientist |
|-----------|--------------|----------------|--------------------|--------------|
| Search Engine | Limited | Full | Full | Full |
| Tool Use | None | APIs | APIs + Code | Full |
| Memory | None | Basic | Advanced | Full |
| Output | Short answer | Short answer | Report | Academic paper |
| Novelty | None | None | Limited | Yes |

### Trade-offs
- **Efficiency vs. Quality**: Sequential/tree-based planning improves quality but increases latency
- **Flexibility vs. Reliability**: More autonomous systems are harder to verify
- **Memory vs. Compute**: Structured memory improves reasoning but adds overhead

---

## 6. Benchmarks & Evaluation Landscape

| Scenario | Key Benchmarks |
|----------|----------------|
| Multi-hop QA | HotpotQA, MuSiQue, GAIA |
| Web Interaction | WebArena, BrowseComp |
| Report Generation | AutoSurvey, DeepResearch Bench |
| AI for Research | PaperBench, Scientist-Bench |

### Standard Metrics
- **Agentic Search**: Exact Match, F1, Recall@k, Citation Accuracy
- **Report Generation**: LLM-as-Judge scores (factuality, coherence, coverage)
- **AI for Research**: Novelty scores, reproducibility, expert evaluation

### Evaluation Problems
1. **LLM-as-Judge bias**: Prefers longer responses, affected by ordering
2. **Logical evaluation gap**: No standardized metrics for argumentative coherence
3. **Novelty vs. hallucination**: Unclear boundary

---

## 7. Open Problems & Future Directions

### Explicitly Stated Challenges
1. **Retrieval Timing** - Need fine-grained knowledge-gap detection
2. **Memory Evolution** - Proactive personalization, cognitive-inspired structuring
3. **Training Instability** - Entropy collapse, echo trap, void turns in multi-turn RL
4. **Evaluation** - Logical coherence, novelty-hallucination boundary, LLM-as-judge bias

### Under-explored Problems
- Long-term credit assignment for intermediate memory decisions
- Cold-start methods that preserve exploration
- Denser reward signals for GRPO-based algorithms

### "Elephant in the Room" Issues Not Discussed
- Economic incentives (engagement over truth)
- Copyright and attribution in synthesis
- Environmental costs of multi-turn, multi-agent systems

---

## 8. Practical Takeaways

### For Practitioners

**Default Choice:**
- **Prototyping**: Workflow Prompting (e.g., Anthropic's multi-agent framework)
- **Production**: Pre-trained models with tool use (GPT-4, Claude with search tools)

**When It Fails:**
- Tasks requiring domain-specific knowledge graphs
- Scenarios with strict latency budgets
- Cases needing guaranteed factual accuracy

### Available Tools/Libraries

| System | Type | Notes |
|--------|------|-------|
| OpenAI DeepResearch | Commercial API | Full multimodal output |
| Perplexity Deep Research | Commercial API | Web-focused |
| LangChain/LangGraph | Open-source | Build custom DR workflows |
| LlamaIndex | Open-source | RAG + agent capabilities |
| CAMEL-AI OWL | Open-source | Multi-agent research |

### Computational Requirements
- **Phase I (Agentic Search)**: Single GPU inference; 10-30s per query
- **Phase II (Integrated Research)**: Multi-turn retrieval; 1-5 min per report
- **Phase III (AI Scientist)**: Multi-agent orchestration; potentially hours per output

---

## 9. Paper Quality Assessment

- **Coverage**: Comprehensive (500+ references); limited on failure modes, non-English DR
- **Writing**: Well-organized with helpful "Takeaway" boxes
- **Timeliness**: Published November 2025, includes 2025 papers
- **Fairness**: Generally balanced; acknowledges limitations of each approach

---

## 10. Reading Roadmap (Applications/Systems focus)

### Top 3 Papers to Read Next

1. **Self-RAG** (ICLR 2024) → Adaptive retrieval foundation
2. **Search-R1** (2024) → End-to-end RL training paradigm
3. **Anthropic Deep Research** (2025) → Production multi-agent design

### Reading Order
Self-RAG → Search-R1 → Anthropic Deep Research

### Sections to Revisit
- **Section 3.2.2 (Retrieval Timing)** after Self-RAG
- **Section 4.3 (End-to-End Agentic RL)** after Search-R1
- **Section 4.1 (Workflow Prompting)** after Anthropic Deep Research

---

## 11. Quick Reference Summary

**Deep Research (DR)** = LLMs as autonomous research agents that iteratively plan queries, acquire evidence, manage memory, and synthesize verifiable outputs. Unlike RAG's static retrieve-then-generate, DR uses closed-loop reasoning with tool augmentation.

### Key Terms

| Term | Definition |
|------|------------|
| **Deep Research (DR)** | End-to-end research workflow where LLMs autonomously plan, search, synthesize, and report |
| **Agentic Search** (Phase I) | Finding correct sources and extracting concise answers |
| **Integrated Research** (Phase II) | Producing coherent reports integrating heterogeneous evidence |
| **Full-stack AI Scientist** (Phase III) | Generating hypotheses, conducting experiments, proposing novel perspectives |
| **Query Planning** | Decomposing complex questions into structured sequences of sub-queries |
| **Adaptive Retrieval** | Determining when to retrieve based on model confidence/knowledge boundaries |
| **Memory Consolidation** | Transforming transient information into stable, long-term representations |
| **LLM-as-Judge** | Using LLMs to evaluate generated outputs on quality dimensions |

### Mental Map

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PLANNING                           │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐             │
│  │ Parallel │  │ Sequential│  │ Tree-based   │             │
│  └──────────┘  └───────────┘  └──────────────┘             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               INFORMATION ACQUISITION                        │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Retrieval Tools│  │ Timing       │  │ Filtering     │   │
│  └────────────────┘  └──────────────┘  └───────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 MEMORY MANAGEMENT                           │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │
│  │ Consolidate │  │ Index    │  │ Update  │  │ Forget   │  │
│  └─────────────┘  └──────────┘  └─────────┘  └──────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  ANSWER GENERATION                          │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────┐   │
│  │ Integrate   │  │ Synthesize &  │  │ Structure &     │   │
│  │ Upstream    │  │ Maintain      │  │ Present         │   │
│  │ Info        │  │ Coherence     │  │ (multimodal)    │   │
│  └─────────────┘  └───────────────┘  └─────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                    Final Report
                    (with citations)
```

---

*Source: arXiv:2512.02038 - "Deep Research: A Systematic Survey" (November 2025)*
