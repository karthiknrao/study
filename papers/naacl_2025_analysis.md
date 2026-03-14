# Research Paper Analysis: "Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools"

**Paper Type**: LLM Planning / Agent Tool-Use Paper (Combining LLMs with formal verification)

## Step 1: Bibliographic Information

- **Title**: Large Language Models Can Solve Real-World Planning Rigorously with Formal Verification Tools
- **Authors**: Yilun Hao (MIT), Yongchao Chen (MIT/Harvard), Yang Zhang (MIT-IBM Watson AI Lab), Chuchu Fan (MIT)
- **Venue**: NAACL 2025 (Conference of the Nations of the Americas Chapter of the ACL)
- **Year**: 2025
- **Project Page**: https://sites.google.com/view/llm-rwplanning
- **Pages**: 3434-3483 (50 pages)

## Step 2: Core Analysis

### Quick Summary
- **One-sentence contribution**: This paper proposes a framework that combines LLMs with SMT solvers to formalize and solve complex multi-constraint planning problems, achieving 93.9% success rate on travel planning benchmarks and strong zero-shot generalization to new domains.
- **Problem statement**: LLMs struggle with complex multi-constraint planning problems (e.g., travel planning with budget, transportation, timing constraints) due to lack of rigorous logical reasoning, while algorithm-based solvers (SMT/SAT) handle constraints rigorously but can't parse natural language.
- **Key innovation**: Using LLMs to translate natural language queries into formal constraint satisfaction problems (SMT encoding), then leveraging sound and complete SMT solvers to find solutions, with interactive repair for unsatisfiable queries.

### Methodology
- **Approach**: Two-stage framework: 1) LLM translates natural language query → formal steps → SMT code, 2) SMT solver finds solution or provides unsatisfiability proof.
- **Key components**:
  1. **Query-Step Generation**: LLM converts natural language to structured problem formulation steps
  2. **Step-Code Generation**: LLM converts steps to SMT solver code (using Z3)
  3. **SMT Solver Execution**: Z3 solves constraint satisfaction problem
  4. **Interactive Plan Repair**: If unsatisfiable, LLM analyzes core constraints, suggests modifications, interacts with user
- **Algorithm**: Uses in-context learning with 3 examples to teach LLM pattern of converting queries to SMT constraints.

### Results
- **Main findings**: Framework achieves 93.3%/93.9% final pass rates on TravelPlanner validation/test sets, vs. 10% for best LLM-only baseline (o1-preview) and 65% for LLM-Modulo framework.
- **Benchmarks**: TravelPlanner dataset (Xie et al., 2024) with 180 validation, 1000 test queries; UnsatChristmas dataset; new multi-constraint tasks (Block Picking, Task Allocation, TSP, Warehouse).
- **Comparisons**: Significantly outperforms all baselines: Greedy Search (0%), TwoStage GPT-4 (0.6%), Direct GPT-4 (4.4%), Direct o1-preview (10%).

### Limitations
- **Acknowledged**: Need for careful prompt design of instruction steps and corresponding codes; compute cost of LLM+SMT; potential hallucination in step generation.
- **Assumptions**: Access to flight/hotel/attraction APIs for real data; SMT solver can encode all constraints.
- **Failure cases**: LLM may generate incorrect steps/code; SMT solver may time out for very complex problems.

## Step 3: Type-Specific Analysis (LLM/Agent Planning)

### Framework Architecture
```
┌─────────────────────────────────────────────────────────┐
│                  USER NATURAL LANGUAGE QUERY             │
├─────────────────────────────────────────────────────────┤
│  LLM: Query → Steps → Code (SMT encoding)               │
├─────────────────────────────────────────────────────────┤
│  SMT SOLVER: Solve constraints / Provide unsat core     │
├─────────────────────────────────────────────────────────┤
│  IF SAT: Return plan                                    │
│  IF UNSAT: LLM analyzes, suggests modifications         │
│           (optional: interact with user)                │
│           Update code, retry                            │
└─────────────────────────────────────────────────────────┘
```

### Key Components Table
| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Planning** | LLM generates problem formulation steps | Translate natural language to structured problem |
| **Tool Use** | SMT solver (Z3) | Rigorously solve constraint satisfaction |
| **Memory** | Not explicitly included | - |
| **Interaction** | Unsatisfiable core analysis + suggestions | Handle infeasible queries, provide feedback |

### Environment & Evaluation
- **Domain**: Travel planning (primary), plus Block Picking, Task Allocation, TSP, Warehouse
- **Action space**: Selecting cities, transportation, attractions, restaurants, accommodations subject to constraints
- **Observation space**: Natural language queries with multiple constraints
- **Evaluation metrics**: Delivery rate, commonsense constraint pass rate, hard constraint pass rate, final pass rate

### Results Detail

**Table 1: TravelPlanner Performance**
- **Ours (Claude-3)**: 93.3% validation, 93.9% test final pass rate
- **Ours (GPT-4)**: 93.3% validation, 90.2% test
- **Baselines**: Direct o1-preview (10%), LLM-Modulo (65% with o1-preview)

**Table 2: Zero-shot Generalization to New Tasks**
- **Block Picking**: 92% optimal (vs 4% for TwoStage GPT-4o)
- **Task Allocation**: 92% optimal (vs 0%)
- **TSP**: 100% optimal (vs 0%)
- **Warehouse**: 72% optimal (vs 0%)

**Interactive Plan Repair** (Tables 3 & 4):
- **UnsatChristmas**: 85.0% average success (vs 63.7% no reason baseline)
- **Modified TravelPlanner**: 91.7% success with 20 iterations

### Prompting Strategy
- **Technique**: In-context learning with 3 examples
- **Example structure**: Query → Steps (9 sections by constraint type) → SMT code
- **Robustness**: Tested with paraphrased prompts - 86.7% pass rate with GPT-4 (vs 93.3% original)

## Step 4: Critical Evaluation

### Strengths
1. **Elegant combination**: Leverages LLMs for natural language understanding and SMT solvers for rigorous constraint solving - addresses weaknesses of both approaches.
2. **Strong empirical results**: 93.9% success vs 10% for best LLM-only baseline on challenging TravelPlanner benchmark.
3. **Zero-shot generalization**: Framework generalizes to unseen constraint types and completely new domains with minimal prompt modification.
4. **Interactive repair**: Handles unsatisfiable queries by identifying core conflicts and suggesting modifications.
5. **Prompt robustness**: Not overly sensitive to prompt wording (tested with paraphrased prompts).

### Weaknesses
1. **Prompt engineering required**: Need carefully designed instruction steps and code examples (though only 3 examples needed).
2. **Compute cost**: LLM inference + SMT solving is more expensive than LLM-only approaches.
3. **Potential hallucination**: LLM may generate incorrect steps/code, though SMT solver provides verification.
4. **Limited scalability**: For extremely complex problems, SMT solver may time out.
5. **API dependencies**: Real-world deployment requires access to flight/hotel/attraction data APIs.

### Reproducibility
- **Code availability**: Project page suggests code will be released (not yet available in paper).
- **Data**: Uses public TravelPlanner dataset and newly created datasets.
- **Experiments**: Well-documented with detailed evaluation metrics and ablation studies.
- **Verifiability**: Framework uses standard components (LLMs + Z3), should be reproducible.

### Impact Potential
- **Field influence**: Bridges gap between LLM-based planning and formal verification communities.
- **Follow-up work**: Could inspire similar combinations for other combinatorial optimization problems.
- **Practical applications**: Real-world travel planning, logistics, scheduling systems.
- **Research direction**: Demonstrates value of hybrid neuro-symbolic approaches.

## Step 5: Personal Takeaways

### Key Insight
LLMs excel at parsing natural language and structuring problems, while formal solvers excel at rigorous constraint satisfaction. Combining them creates a system greater than the sum of its parts for complex real-world planning.

### Related Work
- **LLM-Modulo Framework** (Kambhampati et al., 2024): Also combines LLMs with external critics/verifiers, but achieves only 65% on TravelPlanner.
- **TravelPlanner benchmark** (Xie et al., 2024): Established the challenge of multi-constraint travel planning.
- **Code generation for planning** (Liu et al., 2023; Guan et al., 2023): Similar ideas of using LLMs to generate planning code.

### Action Items
1. **Implementation**: Try to reproduce with open-source LLMs (Llama, Mixtral) and Z3.
2. **Extension**: Apply framework to other domains like project scheduling, resource allocation.
3. **Optimization**: Reduce prompt engineering via few-shot learning or fine-tuning.
4. **Evaluation**: Test on more real-world planning problems with diverse constraints.

### Relevance
- **To LLM/agent research**: Demonstrates effective tool use pattern for complex reasoning tasks.
- **To planning/scheduling**: Provides bridge between natural language interfaces and formal planning systems.
- **To my work**: Hybrid neuro-symbolic approaches for reliable AI systems.

---

**Analysis completed**: 2026-03-05
**Paper**: NAACL 2025, pages 3434-3483
**Key metric**: 93.9% success on TravelPlanner vs 10% LLM-only baseline