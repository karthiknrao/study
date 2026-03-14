# Paper Analysis: A Self-Improving Coding Agent

## Metadata
- **Title**: A Self-Improving Coding Agent
- **Authors**: Maxime Robeyns (University of Bristol/iGent AI), Martin Szummer (iGent AI), Laurence Aitchison (University of Bristol)
- **Venue/Year**: arXiv preprint, May 2025 (v2)
- **Code**: https://github.com/MaximeRobeyns/self_improving_coding_agent
- **arXiv**: 2504.15228

---

## Summary
**One-sentence contribution**: A coding agent that can edit its own Python codebase to improve its performance on benchmark tasks, demonstrating non-gradient-based learning through code updates and achieving 17%→53% improvement on SWE-Bench Verified.

**Problem statement**: Traditional agent systems require manual design of prompting strategies and orchestration code. Can an agent autonomously improve its own implementation?

**Key innovation**: Eliminates the distinction between meta-agent and target-agent—**a single agent that serves as both**, editing its full Python codebase (not a DSL) based on benchmark performance.

---

## Paper Type: **Agent Paper** (with LLM elements)

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVING CODING AGENT                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│   │ Main Agent   │  │ Sub-Agents   │  │   Async Overseer     │    │
│   │ (Orchestrator)│  │ - Coding     │  │   (Safety Monitor)   │    │
│   │              │  │ - Problem    │  │                      │    │
│   │              │  │   Solver     │  │   Runs every 30s     │    │
│   │              │  │ - Reasoning  │  │   or on key events   │    │
│   │              │  │   (o3-mini)  │  │                      │    │
│   └──────────────┘  └──────────────┘  └──────────────────────┘    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          TOOLS                                      │
│   open_file | close_file | overwrite_file | execute_command        │
│   submit_answer | return_result | early_exit | archive_analysis   │
│   calculator | diff_patch_edit (self-discovered)                  │
├─────────────────────────────────────────────────────────────────────┤
│                    META-IMPROVEMENT LOOP                            │
│                                                                     │
│   ┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌─────────┐    │
│   │ Agent i │───▶│ Benchmarks  │───▶│ Archive │───▶│ Agent   │    │
│   │         │    │ Evaluation  │    │ Store   │    │ i+1     │    │
│   └─────────┘    └─────────────┘    └─────────┘    └─────────┘    │
│        ▲                                                    │       │
│        └────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Planning** | Main orchestrator agent delegates to sub-agents | Decomposes tasks, routes to specialists |
| **Memory** | Archive of previous agent iterations + benchmark results | Provides historical context for improvement decisions |
| **Tool Use** | File operations, shell execution, diff editing (self-discovered) | Code manipulation and testing |
| **Coordination** | Sub-agent calls with context management | Handles long-horizon tasks (~1 hour for meta-improvement) |
| **Oversight** | Async LLM-based overseer (separate thread) | Detects loops, deviations, can cancel/notify agents |

---

## Environment

- **Domain**: Software engineering, coding tasks
- **Action space**:
  - File operations (open, close, edit, overwrite)
  - Shell command execution
  - Sub-agent invocation
  - Archive analysis
- **Observation space**:
  - Problem statements
  - File contents and directory trees
  - Tool/sub-agent responses
  - Overseer notifications
- **Reward signal**: Utility function combining benchmark score, cost, and time
  - `U = 0.5×pscore + 0.25×(1-cost/$10) + 0.25×(1-time/300s)`
  - Timeout penalty: 50% reduction

---

## Self-Improvement Mechanism

**Algorithm 1: Self-Referential Agent Improvement**
```
Input: Benchmarks B, iteration count n
1. Initialize agent A₀
2. For i = 0 to n-1:
   a. Evaluate Aᵢ on benchmarks → store pᵢ
   b. Select best agent from archive: î = argmax(p₀...pᵢ)
   c. Run Aî to generate Aᵢ₊₁ (edits own codebase)
3. Return Aₙ
```

**Key difference from ADAS**: The meta-agent IS selected from the archive of improved agents, enabling compounding improvements.

---

## Results

### Benchmark Performance (15 iterations)

| Iteration | File Edit | Symbol Loc | SWE-Bench V | LiveCodeBench | Cost ($) | Time (s) |
|-----------|-----------|------------|-------------|---------------|----------|----------|
| 0 (base) | 0.82 | 0.35 | **0.17** | 0.65 | 1.91 | 130.2 |
| 9 | 0.88 | 0.40 | 0.47 | 0.61 | 2.03 | 126.3 |
| 14 | 0.94 | 0.34 | **0.53** | 0.67 | 2.20 | 117.1 |
| 15 | 0.91 | 0.40 | **0.51** | **0.71** | 1.70 | 114.5 |

**Key improvement**: **17% → 53% on SWE-Bench Verified** (3.1× improvement)

### Self-Discovered Improvements

| Iteration | Innovation | Impact |
|-----------|------------|--------|
| Early | "Smart Edit" tool (diff-based editing) | Faster file edits, token efficiency |
| 6 | Code Context Summarization | Better context management |
| 9 | AST Symbol Locator | Improved codebase navigation |
| Later | File Edit Verification | Quality assurance on edits |
| Later | Hybrid Symbol Locator | Combined AST + search |

### Reasoning Tasks (AIME 2024, GPQA Diamond)

| Metric | Base Agent | o3-mini alone (high) | After Improvement |
|--------|------------|---------------------|-------------------|
| Mean Accuracy | 76% | 83% (87% + 79%) | ~77% |

**Finding**: Little improvement on reasoning-heavy tasks where base models already perform well. Scaffolding may interfere with "reasoning models" (o1, o3-mini, R1).

---

## Critical Evaluation

### Strengths
1. **Truly self-improving**: Single agent serves as both meta and target, enabling compounding
2. **Full Python codebase**: No domain-specific language constraints
3. **Practical improvements**: Real gains on real benchmarks (SWE-Bench, LiveCodeBench)
4. **Safety-conscious**: Async overseer, observability mechanisms
5. **Sample-efficient**: Non-gradient learning through code reflection

### Weaknesses
1. **Path dependency**: Initial feature ideas heavily influence subsequent suggestions
2. **High cost**: ~$7,000 for 15-iteration run
3. **Limited reasoning improvement**: Scaffolding can interfere with reasoning models
4. **Short timeouts**: 5-minute limits may understate base model capabilities
5. **Novelty challenge**: LLM struggles to generate truly innovative improvements

### Reproducibility
- **Code**: Available at GitHub (linked in paper)
- **Experiments**: Well-documented with detailed iteration logs
- **Benchmarks**: Standard (SWE-Bench Verified, LiveCodeBench) + synthetic tasks

### Impact Potential
- **High for agent design**: Demonstrates automated agent optimization is viable
- **Foundation for future work**: Framework for jointly training LLMs with agent systems
- **Safety implications**: Important stepping stone for self-modifying AI systems

---

## Safety Considerations

The paper dedicates Section 6 to safety:

1. **Observability**: Web interface + async overseer provide rich insight into agent behavior
2. **Evaluation-based updates**: Safety benchmarks can be included in evaluation loop
3. **No weight updates**: Steganography risk lower without model fine-tuning
4. **Mitigations may be insufficient**: For recursively self-improving systems with weight updates

---

## Personal Takeaways

### Key Insight
**The most valuable improvements were tooling innovations** (diff-based editing, AST symbol location) rather than prompting strategies. This suggests agent improvement may be more about capability expansion than prompt optimization.

### Related Work
- **ADAS** (Hu et al. 2024): Separate meta/target agents, DSL-based
- **Gödel Agent** (Yin et al. 2024): Self-modifying but not general-purpose coding
- **AlphaEvolve** (Google): Scientific discovery via coding agents
- **STOP** (Zelikman et al. 2024): Self-taught optimizer for algorithmic tasks

### Action Items
1. **Explore the codebase**: Understanding the implementation details
2. **Compare with Claude Code**: How does this relate to current coding assistants?
3. **Read ADAS paper**: For context on automated agent design
4. **Consider applications**: What domains could benefit from self-improving agents?

### Relevance to My Work
This paper demonstrates that **LLM reflection + code execution = non-gradient learning**. The key insight is that an agent can improve by:
1. Running on benchmarks
2. Analyzing results
3. Editing its own code
4. Repeating

This is fundamentally different from gradient descent—more sample-efficient but potentially less thorough.

---

## Open Questions
1. How to prevent path dependency in improvement trajectories?
2. Can this approach scale to longer runs (100+ iterations)?
3. How would joint LLM + agent system training work?
4. What's the theoretical limit of scaffolding-based improvement?

---

## Appendix: Agent Prompts

The paper includes detailed prompts in Appendices:

### Main Agent (Orchestrator)
- Routes requests to sub-agents
- Synthesizes results into final answer
- Has file viewing tools for context, but no substantive editing tools

### Coding Sub-Agent
- Professional programmer persona
- 6-step approach: understand context → prototype → code → test → cleanup
- Can call reasoning agent when stuck

### Problem Solver Sub-Agent
- Simple, fast execution
- Plan → Execute → Verify workflow

### Overseer Prompt
- Monitors execution in real-time
- Can notify agents or cancel them
- XML-formatted judgements with fields for: making_progress, is_looping, needs_notification, force_cancel_agent
