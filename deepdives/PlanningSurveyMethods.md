# Large Language Models for Planning: A Comprehensive and Systematic Survey - Analysis

## 1. Survey Scope & Motivation

### Precise Scope
**Included:**
- LLM-based planning methodologies across all domains (web navigation, embodied AI, travel, games)
- Three main approaches: External Module Augmented, Finetuning-based, Searching-based
- Theoretical foundations (MDP/POMDP, high/low-level, open/closed-loop planning)
- Evaluation frameworks, benchmarks, and underlying mechanisms

**Explicitly Excluded:**
- Non-LLM planning methods (symbolic planners as comparison points only)
- General LLM capabilities not specifically related to planning
- Planning in non-agent contexts

### Why This Survey Is Needed Now
- **Catalyst**: LLMs showing promise in planning tasks but systematic understanding lacking
- **Gap**: Previous surveys either outdated (Feb 2024) or insufficiently comprehensive
- **Confusion**: Rapid proliferation of methods needs unified taxonomy and comparison

### Target Audience
- AI/ML researchers and practitioners
- Assumes familiarity with LLMs, reinforcement learning, and basic planning concepts

## 2. Taxonomy & Organization

```
LLM-based Planning Methods
├── External Module Augmented Methods
│   ├── Planner Enhanced (LLM+P, LLM-DP, PDDLEGO, etc.)
│   └── Memory Enhanced (MemoryBank, TiM, MemGPT, etc.)
├── Finetuning-based Methods
│   ├── Imitation Learning-based (Agent-FLAN, FireAct, KnowAgent, etc.)
│   └── Feedback-based (A3T, SPPO, Step-DPO, etc.)
└── Searching-based Methods
    ├── Decomposition-based (HuggingGPT, RaDA, UrbanLLM, etc.)
    ├── Exploration-based (ToT, SearChain, AoT, etc.)
    └── Decoding-based (contrastive, predictive, grounded decoding)
```

**Intuitiveness**: Highly intuitive classification based on technical approach
**Edge Cases**: Some methods combine categories (e.g., search with external modules)
**Comparison**: More granular than prior surveys focusing only on agent planning

## 3. Historical Evolution

**Timeline:**
- **Pre-2020**: Symbolic planning dominates (PDDL framework)
- **2020-2022**: Early LLM planning experiments, simple prompting approaches
- **2023**: Systematic methods emerge (ToT, AoT, ReAct)
- **2024**: Specialized finetuning, hybrid approaches, large-scale benchmarks
- **2025**: Maturity in taxonomy, interpretability studies, commercial applications

**Paradigm Shifts:**
1. Symbolic → Neural planning
2. Static prompting → Interactive planning
3. Single-agent → Multi-agent collaborative planning

**Cyclical Trends**: Search-based methods rediscover classical AI techniques (MCTS, A*) but with LLM heuristics

## 4. Core Concepts Deep Dive

### External Module Augmented Methods
- **Definition**: Combine LLMs with planners/memory for enhanced capabilities
- **Key Papers**: LLM+P (2023), MemoryBank (2024), MemGPT (2024)
- **Technical Approach**: LLM generates high-level plans → symbolic planner executes/details
- **Strengths**: Formal correctness guarantees, memory efficiency
- **Limitations**: Integration overhead, symbolic-LLM translation challenges
- **Current State**: Active research area with practical applications

### Finetuning-based Methods
- **Definition**: Specialize LLMs for planning via supervised/reinforcement learning
- **Key Papers**: Agent-FLAN (2024), A3T (2024), Step-DPO (2025)
- **Technical Approach**: Fine-tune on planning trajectories with imitation/reward learning
- **Strengths**: End-to-end optimization, better sample efficiency
- **Limitations**: Expensive training, domain specificity
- **Current State**: Mainstream approach for specialized applications

### Searching-based Methods
- **Definition**: Explore planning space using search algorithms with LLM guidance
- **Key Papers**: ToT (2023), SearChain (2024), AoT (2024)
- **Technical Approach**: Tree/graph search with LLM as expansion/scoring heuristic
- **Strengths**: Systematic exploration, handles uncertainty
- **Limitations**: Computational overhead, requires careful search design
- **Current State**: Highly active with many algorithmic innovations

## 5. Comparative Analysis

| Approach | Performance | Scalability | Data Efficiency | Interpretability |
|----------|-------------|-------------|----------------|------------------|
| External Module | Medium-High | High | High | High |
| Finetuning | High | Medium | Low | Medium |
| Searching | Medium | Low | High | Medium-High |

**Trade-offs:**
- External: Correctness vs. flexibility
- Finetuning: Performance vs. training cost
- Searching: Quality vs. computational cost

**Hybrid Approaches**: Many methods combine categories (e.g., search with memory, fine-tuning with decomposition)

## 6. Benchmarks & Evaluation Landscape

**Dominant Benchmarks:**
- Web navigation: WebShop, WebArena
- Embodied AI: ALFRED, BabyAI, MineDojo
- Games: TextWorld, ScienceWorld
- Travel planning: TravelPlanner

**Standard Metrics:**
- Success rate, task completion
- Path length/optimality ratio
- Human evaluation scores
- Computational efficiency

**Evaluation Issues:**
- Fragmentation across domains
- Insufficient real-world testing
- Under-measured: generalization, robustness, safety

## 7. Open Problems & Future Directions

**Explicitly Stated Challenges:**
1. Long-horizon planning with partial observability
2. Real-time planning in dynamic environments
3. Multi-agent coordination and communication
4. Sample-efficient learning from limited demonstrations

**Under-explored Areas:**
- Planning under resource constraints
- Ethical/safety constraints in planning
- Cross-domain transfer learning

**"Elephant in the Room":**
- LLMs' fundamental planning limitations vs. symbolic systems
- Evaluation vs. real-world deployment gap

**Field Advancement Needs:**
1. Unified evaluation framework
2. Open-source planning benchmarks
3. Theoretical understanding of LLM planning capabilities

## 8. Practical Takeaways

**For Practitioners:**
- Start with searching-based methods (ToT/AoT) for prototyping
- Use finetuning for domain-specific, performance-critical applications
- Consider external modules when formal correctness is required

**Default Choices:**
- General tasks: Exploration-based search (ToT variant)
- Web/UI automation: Memory-enhanced methods
- Robotics/embodied: Finetuning with simulation data

**Tools/Libraries:**
- LangChain, AutoGPT (agent frameworks)
- Custom implementations for specialized methods

**Computational Resources:**
- Search methods: High inference cost, moderate memory
- Finetuning: High training cost, efficient inference
- External modules: Low compute, integration complexity

## 9. Paper Quality Assessment

**Comprehensiveness:**
- Excellent coverage of LLM planning literature (250+ papers)
- Missing: Very recent 2025 papers (published May 2025, so up-to-date as of publication)

**Organization:**
- Clear logical structure, well-defined taxonomy
- Good use of figures and tables for visualization

**Fairness:**
- Balanced discussion of pros/cons for each approach
- Acknowledges limitations of LLM-based planning

**Currency:**
- Published May 2025, includes most 2024 literature
- Rapidly evolving field means new methods emerging

## 10. Reading Roadmap

**For Theory Focus:**
1. **ToT (2023)** - Foundation of search-based planning
2. **LLM+P (2023)** - Hybrid symbolic-neural approach  
3. **Agent-FLAN (2024)** - State-of-the-art finetuning

**For Applications Focus:**
1. **MemGPT (2024)** - Practical memory management
2. **SearChain (2024)** - Web navigation applications
3. **KnowAgent (2024)** - Embodied AI planning

**Reading Order:** Theoretical → Hybrid → Applied
**Revisit Sections:** 3.1 (External modules) after theoretical papers, 3.3 (Searching) after application papers

## 11. Quick Reference Summary

**Current State:** LLMs show significant promise for planning tasks but face challenges in long-horizon reasoning, robustness, and real-world deployment. The field has converged on three main technical approaches, each with distinct trade-offs between performance, efficiency, and correctness.

**Key Terms:**
- **MDP/POMDP**: Planning formalisms (fully/partially observable)
- **Open/Closed-loop**: One-shot vs. interactive planning
- **External Augmentation**: LLMs + symbolic planners/memory
- **Search-based**: Systematic exploration of planning space
- **Finetuning**: Specialized training for planning tasks

**Mental Map:**
```
Planning Problems → Formalization (MDP/POMDP) → Method Selection:
  - Need correctness? → External modules
  - Have data? → Finetuning  
  - Need exploration? → Search-based
  → Evaluation → Real-world deployment
```