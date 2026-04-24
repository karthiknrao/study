# AI Agents & Simulations: Comprehensive Research Compilation
## Based on Stanford CS222 (Fall 2024) — Joon Sung Park

---

## Table of Contents
1. [Foundational Papers (Course References)](#1-foundational-papers)
2. [Generative Agents: Core Architecture & Extensions](#2-generative-agents)
3. [Cognitive Architectures for Language Agents](#3-cognitive-architectures)
4. [Agent Memory Systems](#4-agent-memory-systems)
5. [Generative Agent-Based Models (GABMs)](#5-generative-agent-based-models)
6. [Large-Scale Social Simulation Platforms](#6-large-scale-social-simulation-platforms)
7. [Multi-Agent LLM Systems & Emergent Behavior](#7-multi-agent-llm-systems)
8. [Opinion Dynamics & Political Simulation](#8-opinion-dynamics)
9. [Validation, Believability & Accuracy](#9-validation)
10. [Ethics, Bias & Representation](#10-ethics)
11. [Silicon Samples & LLM-as-Human-Subjects](#11-silicon-samples)
12. [Applications: Clinical, Education, Digital Twins](#12-applications)
13. [Generative Ghosts & AI Afterlives](#13-generative-ghosts)
14. [Surveys & Meta-Analyses](#14-surveys)
15. [Key Repositories & Resources](#15-resources)

---

## 1. Foundational Papers (Course References)

These are the readings directly assigned in CS222:

| Paper | Authors | Year | Topic |
|-------|---------|------|-------|
| Dilemmas in a General Theory of Planning | Rittel & Webber | 1973 | Wicked problems |
| Micromotives and Macrobehavior (Ch. 1) | T.C. Schelling | 1978 | Emergent macro behavior |
| Social Simulacra | Park, Popowski, Cai, Morris, Liang, Bernstein | 2022 | Populated prototypes for social computing |
| Out of One, Many: Using LMs to Simulate Human Samples | Argyle, Busby, Fulda, Gubler, Rytting, Wingate | 2023 | LLMs as population samples |
| Précis of Unified Theories of Cognition | A. Newell | 1992 | Cognitive architecture foundations |
| A Gentle Introduction to Soar | Lehman et al. | 2006 | Soar cognitive architecture |
| Generative Agents: Interactive Simulacra of Human Behavior | Park, O'Brien, Cai, Morris, Liang, Bernstein | 2023 | Core generative agent architecture |
| Cognitive Architectures for Language Agents | Sumers, Yao, Narasimhan, Griffiths | 2024 | CoALA framework |
| LLMs Generate Structurally Realistic Social Networks | Chang, Chaszczewicz, Wang, Josifovska, Pierson, Leskovec | 2024 | Social network simulation |
| Roleplay-Doh | Louie, Nandi, Fang, Chang, Brunskill, Yang | 2024 | Domain-expert LLM patient creation |
| The Role of Emotion in Believable Agents | Bates | 1994 | Believability in agents |
| Predicting Results of Social Science Experiments Using LLMs | Ashokkumar, Hewitt, Ghezae, Willer | 2024 | LLM prediction of experiments |
| Agent-Based Models in Empirical Social Research | Bruch & Atwell | 2015 | ABM methodology |
| Growing Artificial Societies (Ch. 1-2) | Epstein & Axtell | 1996 | Bottom-up social simulation |
| The Essence of Chaos | E.N. Lorenz | 1993 | Chaos theory |
| The Nash Equilibrium: A Perspective | Holt & Roth | 2004 | Game theory equilibria |
| D3: Data-Driven Documents | Bostock, Ogievetsky, Heer | 2011 | Visualization language |
| NetLogo User Manual | Wilensky | 2021 | ABM platform |
| LLMs Cannot Replace Human Participants | Wang, Morgenstern, Dickerson | 2024 | Ethics of representation |
| Whose Opinions Do LLMs Represent? | Santurkar, Durmus, Ladhak, Lee, Liang, Hashimoto | 2023 | LLM opinion bias |
| Generative Ghosts | Morris & Brubaker | 2024 | AI afterlives |
| Mobility Network Models of COVID-19 | Chang et al. | 2021 | Network-based epidemic modeling |
| The Computer for the 21st Century | Weiser | 1991 | Ubiquitous computing vision |

---

## 2. Generative Agents: Core Architecture & Extensions

### 2.1 The Foundational Paper
- **Generative Agents: Interactive Simulacra of Human Behavior**
  Park, O'Brien, Cai, Morris, Liang, Bernstein | UIST 2023
  *25 agents in a Sims-like sandbox exhibiting emergent social behaviors — memory stream, reflection, planning architecture*

### 2.2 Major Extensions (2024-2026)

- **Generative Agent Simulations of 1,000 People**
  Park, Zou et al. | arXiv 2024 (arXiv:2411.10109)
  *Simulates 1,052 real individuals using LLMs grounded in qualitative interviews. Achieves 85% accuracy matching human responses in social science experiments. Interview-grounded agents outperform survey-based approaches.*

- **Generative Agents in Agent-Based Modeling: Overview, Validation, and Challenges**
  IEEE Trans. on Artificial Intelligence, Vol. 6, No. 12, December 2025
  *Comprehensive review of generative agent use in ABM with hybrid validation bridging quantitative metrics and qualitative believability assessments.*

- **Towards Realistic User Simulations with Generative Agents**
  ResearchGate 2024
  *Addresses limitations of simulation approaches for conversational recommender systems using generative agents.*

- **Generative Agents in the Streets: Exploring the Use of Large Language Models**
  ScienceDirect 2025
  *Explores LLM-based agents adopting characters with personalities, preferences, and backgrounds for real-world behavioral simulation.*

- **LLM Agents Grounded in Self-Reports Enable General-Purpose Simulations**
  (Same as "Simulations of 1,000 People" above — this is the arXiv title)

---

## 3. Cognitive Architectures for Language Agents

- **Cognitive Architectures for Language Agents (CoALA)**
  Sumers, Yao, Narasimhan, Griffiths | TMLR 2024 (arXiv:2309.02427)
  *Proposes the CoALA framework: LLM as core reasoning engine, memory (working + long-term), action space (internal + external), structured decision-making loop. Retrospectively surveys and organizes a large body of LLM agent work.*

- **The Rise and Potential of Large Language Model Based Agents: A Survey**
  Xi et al. | Science China Information Sciences 2025
  *Comprehensive survey covering construction frameworks, application scenarios, and multi-agent societies built on LLMs.*

- **Large Language Model Agent: A Survey on Methodology, Applications, and Challenges**
  arXiv 2025 (arXiv:2503.21460)
  *Systematically deconstructs LLM agent systems through methodology-centered taxonomy, linking architecture to capabilities.*

- **Cognitive LLMs: Toward Human-Like Artificial Intelligence**
  SAGE Journals 2025
  *Advances multi-agent LLM frameworks toward human-like cognition, building on CoALA and classical cognitive science.*

- **Agentic, Multimodal LLM with Behavioural Capabilities**
  Architectural Science Review 2025
  *Examines behavioral simulation capabilities of multimodal LLM agents in architectural and spatial contexts.*

- **A Comprehensive Survey of the LLM-Based Agent: Contextual Cognition**
  Preprints 2026
  *Highlights the gap in contextual cognition across LLM agent surveys, proposing new evaluation dimensions.*

---

## 4. Agent Memory Systems

- **A Survey on the Memory Mechanism of LLM-Based Agents**
  Zhang & Dai | ACM Computing Surveys 2025 (doi:10.1145/3748302)
  *Systematic review of "what is" and "why we need" memory in LLM agents, covering design, evaluation, and applications.*

- **Memory in the Age of AI Agents: A Survey**
  Liu et al. | arXiv 2025 (arXiv:2512.13564)
  *Comprehensive survey covering episodic, semantic, and procedural memory paradigms. 1000+ GitHub stars, featured as HuggingFace Daily Paper #1.*

- **Memory OS of AI Agent**
  ACL/EMNLP 2025
  *Operating-system-like memory management for LLM agents with short-term, mid-term, and long-term memory tiers.*

- **LLM Agent Memory: A Survey from Unified Representation-Management Perspectives**
  Preprints 2026
  *Decouples memory abstractions from model-specific methods, categorizing into natural language tokens, intermediate representations, and parameters.*

- **Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Challenges**
  arXiv 2025
  *Proposes generalized blueprint with working, episodic, semantic, and procedural stores interacting through a central executive.*

- **AriGraph: Learning Knowledge Graph World Models with Episodic Memory**
  IJCAI 2025
  *Integrates semantic knowledge graphs with episodic vertices/edges, enhancing LLM agent performance in text-based games.*

- **Memory in LLM-Based Multi-Agent Systems: Mechanisms, Challenges, and Collective Intelligence**
  ResearchGate 2024
  *Examines memory sharing, conflicts, and collective intelligence in multi-agent settings with AriGraph-style memory graphs.*

---

## 5. Generative Agent-Based Models (GABMs)

- **LLMs and Generative Agent-Based Models for Complex Systems Research**
  ScienceDirect 2024 (doi:10.1016/j.cosrev.2024.1001386)
  *Reviews LLMs' disruptive role in network science, evolutionary game theory, social dynamics, and epidemic modeling. Assesses predicting social behavior, enhancing cooperation, and modeling disease propagation.*

- **Large Language Models Empowered Agent-Based Modeling and Simulation: A Survey**
  Gao et al. | Nature Humanities & Social Sciences Communications 2024
  *Interdisciplinary survey covering ABM background, LLM agent capabilities (autonomy, social ability, reactivity, proactiveness), and future directions.*

- **Integrating LLM in Agent-Based Social Simulation**
  arXiv 2025 (arXiv:2507.19364)
  *Argues hybrid approaches integrating LLMs into established platforms (GAMA, NetLogo) offer the best compromise between capability and reliability.*

- **LLM-Augmented Agent-Based Modelling for Social Simulations**
  ResearchGate 2024
  *Enhancements include long-context modeling, expert constraints, and interactive feedback for more realistic social ABMs.*

- **Scaling LLM-Guided Agent Simulations**
  MIT Media Lab | AAMAS 2025
  *Fundamental breakthrough enabling LLM-guided agents to scale from small to large simulations with theoretical bounds.*

- **Value-Based LLM Agent Simulation for Mutual Fund Trading**
  Nature Scientific Reports 2025
  *Applies value-based decision-making in LLM agents to simulate mutual fund trading behaviors.*

- **LLM-Based Multi-Agent Systems are Scalable Graph Generative Models**
  ACL Findings 2025
  *Demonstrates LLM-based multi-agent systems can generate structurally realistic social graphs.*

- **Forecasting Research Ideas through Multi-Agent LLM Simulations**
  ACM CI 2025
  *Uses multi-agent LLM simulation to predict future research directions, instantiated as author role-playing agents.*

---

## 6. Large-Scale Social Simulation Platforms

- **AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents**
  Piao, Yan, Zhang et al. | arXiv 2025 (arXiv:2502.08691)
  *Large-scale social simulator integrating LLM-driven agents with emotional, need-based, and cognitive processes. Uses Ray/asyncio for scalable urban, social, and economic environment modeling. Experiments: reducing polarization, evaluating UBI, mitigating misinformation.*

- **SocioVerse: A World Model for Social Simulation Powered by LLM Agents and 10M Real-World Users**
  Zhang, Lin et al. | arXiv 2025 (arXiv:2504.10157)
  *World model with four alignment components and a user pool of 10 million real individuals. Validated across politics (elections), news, and economics domains with high accuracy.*

- **SoAgent: A Real-World Data Empowered Agent Pool**
  VLDB Workshop 2025
  *Framework generating agents from real census-like data. Outperforms synthetic-profile agents in questionnaire simulation and conspiracy theory diffusion studies.*

- **From Individual to Society: A Survey on Social Simulation Driven by LLM-Based Agents**
  Fudan DISC | arXiv 2024
  *Comprehensive survey from Fudan University on the progression from individual agents to societal-scale simulation.*

- **Synthetic Social Media Influence Experimentation via Agentic Simulation**
  JASSS 2025 (Journal of Artificial Societies and Social Simulation)
  *Creates social influence simulation engines with agent-based modeling for key social research questions.*

- **Chirper: Collective Behaviors of LLM-Based Agents**
  ScienceDirect 2025
  *Examines emergent collective behavior in Chirper, a social simulation platform exclusively inhabited by AI agents.*

- **SocioVerse (Code)**
  GitHub: FudanDISC/SocioVerse — Open-source implementation of the 10M-user social simulation framework.

---

## 7. Multi-Agent LLM Systems & Emergent Behavior

- **Emergent Coordination in Multi-Agent Language Models**
  arXiv 2025 (arXiv:2510.05174)
  *Reviews multi-agent LLM systems since ChatGPT, showing emergence of rich social behaviors (friendships, economies) in Sims-like environments.*

- **Cooperative Profiles Predict Multi-Agent LLM Team Performance**
  arXiv 2025 (arXiv:2604.20658)
  *Shows that cooperative behavioral profiles predict team performance in multi-agent LLM settings.*

- **Emergent Social Conventions and Collective Bias in LLM Populations**
  Science Advances 2025 (doi:10.1126/sciadv.adu9368)
  *Demonstrates emergence of conventions and collective biases in LLM populations — foundational for understanding multi-agent dynamics.*

- **Simulating Cooperative Prosocial Behavior with Multi-Agent LLMs**
  ACM 2025 (doi:10.1145/3708359.3712149)
  *Replicates human behavior from public goods game experiments with three treatments: priming, transparency, varying endowments.*

- **ALYMPICS: LLM Agents Meet Game Theory**
  COLING 2025
  *Comprehensive benchmark evaluating LLM agents in rational strategic decision-making scenarios.*

- **Will Systems of LLM Agents Cooperate? Investigation into Social Dilemmas**
  arXiv 2025 (arXiv:2501.16173)
  *Investigates emergent cooperative tendencies when LLMs generate complete strategies for iterated social dilemmas.*

- **Emergent Cooperative Behavior in LLMs with Game Theory**
  ResearchGate 2025
  *Investigates LLM behavior through iterated prisoner's dilemma and trust games.*

- **LLMs Miss the Multi-Agent Mark**
  NeurIPS 2025 (Position Paper)
  *Highlights critical discrepancies between MAS theory and current LLM implementations across four key areas.*

- **LLM-Driven Social Influence for Cooperative Behavior in Multi-Agent Systems**
  IEEE 2025
  *Introduces formal model grounded in game theory and social networks, balancing direct benefits with LLM-guided social influence.*

- **Simulating Social Behavior of LLM-Based Autonomous Negotiator Agents**
  Taylor & Francis 2025
  *Evaluates LLM agents in buyer-seller bargaining games for commerce applications.*

- **Multi-Agent LLM Systems: From Emergent Collaboration to Self-Organization**
  Preprints 2025
  *Surveys the spectrum from collaborative multi-agent systems to self-organizing LLM populations.*

- **A Survey on LLM-Based Multi-Agent Systems: Workflow**
  Springer 2024
  *Comprehensive survey of LLM-based MAS with workflow organization and application scenarios.*

- **LLM-Based Multi-Agents: Progress and Challenges (IJCAI 2024)**
  IJCAI 2024
  *Survey covering fundamental concepts, latest trends, and applications of LLM-based multi-agent systems.*

---

## 8. Opinion Dynamics & Political Simulation

- **Simulating Opinion Dynamics with Grounded LLM Agents**
  arXiv 2025 (arXiv:2603.26701)
  *Addresses inherent LLM bias toward factual consensus, examining implications for political and economic decision-making simulation.*

- **Simulating Opinion Dynamics with Networks of LLM-Based Agents**
  NAACL Findings 2024
  *Proposes LLM-based opinion dynamics simulation. Reveals strong inherent bias toward accurate information, leading to consensus aligned with scientific reality.*

- **Modeling the Impact of LLMs on Opinion Dynamics**
  Engineering Applications of AI 2025
  *Millions of simulations investigating LLMs' unique positive effect on collective opinion difference.*

- **Simulating Influence Dynamics with LLM Agents**
  arXiv 2025 (arXiv:2503.08709)
  *Simulator for modeling competing influences in social networks with LLM agents, for influence propagation and counter-misinformation.*

- **LLM-Powered Simulations Revealing Polarization in Social Networks**
  COLING 2025
  *Language-based simulation providing explainable, dynamic environment for echo chamber studies.*

- **An Agent-Based Simulation of Politicized Topics Using LLMs**
  Springer 2025
  *100-agent simulation grounded in psychometric/demographic data from Serbian social media users, with LLM-generated content feeds.*

- **DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics**
  OpenReview 2025
  *Benchmark for evaluating role-playing LLM agents in multi-turn, multi-agent opinion exchange.*

- **Public Opinion Dissemination Simulation Based on LLMs**
  Nature Scientific Reports 2026
  *Models public opinion flow using large language model agents.*

- **MTOS: LLM-Driven Multi-Topic Opinion Simulation**
  arXiv 2025 (arXiv:2510.12423)
  *Framework integrating multi-topic contexts with LLMs for more realistic opinion simulation.*

- **Investigating Opinion Dynamics in LLM Interactions**
  EPJ Data Science 2025
  *Studies how opinions evolve, addressing polarization, radicalization, and consensus formation.*

---

## 9. Validation, Believability & Accuracy

- **Validation is the Central Challenge for Generative Social Simulation**
  Springer / Artificial Intelligence Review 2025 (doi:10.1007/s10462-025-11412-6)
  *Argues validation is THE key challenge. Reviews methods from traditional ABM validation, LLM benchmarks, and believability assessments. Proposes framework connecting individual-level accuracy to macro-level fidelity.*

- **Validating Generative Agent-Based Models**
  (Technical Report)
  *Bridges the gap between traditional ABM validation (Gräbner 2018) and LLM evaluation benchmarks (BigBench, HELM).*

- **Generative Agents in ABM: Overview, Validation, and Challenges**
  IEEE Trans. AI, Vol. 6, No. 12, Dec 2025
  *Hybrid validation combining quantitative metrics with qualitative believability. Covers 6 validation scenarios including social discussion.*

- **A Large-Scale Replication of Scenario-Based Experiments in Psychology Using LLMs**
  Nature Computational Science 2025
  *Replicated 156 psychological experiments using 3 SOTA LLMs. 73-81% main effect replication, 46-63% interaction effects. LLMs consistently produce larger effect sizes than humans.*

- **Finetuning LLMs for Human Behavior Prediction in Social Science Experiments**
  EMNLP 2025 (arXiv:2509.05830)
  *Demonstrates finetuning on individual-level responses from past experiments meaningfully improves simulation accuracy.*

- **Predicting Results of Social Science Experiments Using LLMs**
  Ashokkumar, Hewitt, Ghezae, Willer | 2024
  *Archive of 70 pre-registered nationally representative survey experiments (476 treatment effects, 105K participants).*

- **Establishing Reference Points for Artificial Social Agent Evaluation**
  Frontiers in Computer Science 2025
  *Develops benchmark standards for evaluating social agents using representative human samples.*

- **Evaluation and Benchmarking of LLM Agents: A Survey**
  ACM 2025 (doi:10.1145/3711896.3736570)
  *Two-dimensional taxonomy organizing LLM agent evaluation frameworks.*

- **Human Simulacra: Benchmarking the Personification of LLMs**
  OpenReview 2025
  *Benchmark for evaluating how well LLMs can personify and simulate individual human behavior.*

---

## 10. Ethics, Bias & Representation

- **Position: LLM Social Simulations Are a Promising Research Method**
  ICML 2025
  *Defends LLM social simulations while acknowledging simulation bias (systematic inaccuracies in representing particular social groups). Addresses social desirability bias and sycophancy.*

- **LLMs Cannot Replace Human Participants Because They Cannot Portray Identity Groups**
  Wang, Morgenstern, Dickerson | 2024
  *Critical paper arguing LLMs fail to faithfully represent diverse identity groups.*

- **Whose Opinions Do LLMs Represent?**
  Santurkar, Durmus, Ladhak, Lee, Liang, Hashimoto | ICML 2023
  *Shows LLM opinions are skewed toward specific demographic groups, raising concerns about representativeness.*

- **LLM Generated Persona is a Promise with a Catch**
  NeurIPS 2025
  *Examines the promises and pitfalls of LLM-generated personas for silicon-sample simulation.*

- **Representation Bias in Political Sample Simulations with LLMs**
  2024
  *Identifies systematic representation biases when using LLMs to simulate political samples.*

- **The Silicon Gaze: A Typology of Biases and Inequality in LLMs**
  SAGE Journals 2025
  *20.3 million query audit of ChatGPT mapping systematic biases in representations of countries, cities, neighborhoods.*

- **Performance and Biases of LLMs in Public Opinion Research**
  Nature Humanities & Social Sciences Communications 2024
  *Detailed demographic data analysis examining representation biases in AI simulations.*

- **LLM-Based Simulations of Human Behavior in Psychological Research**
  AAAI/AIES 2025
  *Examines philosophical problems (scientific representation, epistemic opacity) in using LLMs as models of the human mind.*

- **Large Language Models as Psychological Simulators**
  SAGE Journals 2025
  *Treats LLM simulations as hypothesis-generating tools requiring validation with human samples.*

- **AI, Ethics, and Cognitive Bias: An LLM-Based Synthetic Simulation**
  AI in Education 2025
  *Models six decision-making scenarios with probabilistic representations of key cognitive biases.*

---

## 11. Silicon Samples & LLM-as-Human-Subjects

- **Using LLMs to Generate Silicon Samples in Consumer and Marketing Research**
  Journal of Consumer & Marketing Research 2024 (doi:10.1002/mar.21982)
  *Recommends silicon samples for upstream research (qualitative pretesting, pilot studies) rather than replacing human subjects.*

- **Representativeness and Structural Consistency of Silicon Samples**
  arXiv 2025 (arXiv:2507.02919)
  *Criticizes earlier silicon sample work (Argyle 2023, Heyde 2024) for overstating representativeness.*

- **This Human Study Did Not Involve Human Subjects: Validating LLM-Augmented Surveys**
  Northwestern / MU Collective 2025
  *Combines human participants with LLM-predicted responses. Statistical methods prevent LLM bias in estimates.*

- **GPT-ology, Computational Models, Silicon Sampling: How Should We Think?**
  CogSci 2024
  *Framework for understanding whether LLMs serve as computational models of human cognition vs. simulation tools.*

- **1,000 AI Agents Predict Human Behavior with 85% Accuracy**
  Stanford HAI 2024
  *Stanford press coverage of Park et al.'s 1000-people simulation: 0.98 correlation with human responses.*

- **Social Science Researchers Use AI to Simulate Human Subjects**
  Stanford News 2025
  *Cautions that LLM simulations should augment, not replace, human experiments.*

---

## 12. Applications: Clinical, Education, Digital Twins

### Clinical & Patient Simulation
- **LLMs Forecast Patient Health Trajectories Enabling Digital Twins**
  Nature Digital Medicine 2025
  *LLM-based forecasting for digital twin applications in clinical trials, treatment selection, and adverse event mitigation.*

- **Simulated Patient Systems Powered by LLM-Based AI Agents**
  Nature Digital Medicine 2025
  *Comprehensive review of simulated patient systems for medical education and clinical decision-making.*

- **LLM-Based Virtual Patient Simulations in Medical Education**
  Applied Sciences (MDPI) 2025
  *Reviews 40 studies (2023-2025) mapping implementation approaches, data practices, and evaluation methods.*

- **Interactive LLM-Based Simulator for Dementia-Related Activities**
  arXiv 2025 (arXiv:2603.29856)
  *Framework for digital-twin-style modeling and assistive AI policy for dementia care.*

- **TWIN-GPT: Digital Twins for Clinical Trials via LLMs**
  ACM 2024 (doi:10.1145/3674838)
  *Establishes cross-dataset associations to generate personalized digital twins for different patients.*

- **LLMs Improve Clinical Decision Making**
  BMC Medical Education 2024
  *Uses LLMs to simulate patient-doctor interactions for clinical decision-making training.*

### Education
- **A Guide to Prompt Design for Simulation-Based Education**
  Frontiers in Medicine 2024
  *Bridge the gap for simulationists on LLM use in medical education, covering clinical scenario development and OSCE creation.*

### Digital Twins
- **Digital Twins in Healthcare: Comprehensive Review**
  PMC 2025
  *Reviews LLM-enabled digital twins across healthcare with verification and validation frameworks.*

- **Personalised LLMs and the Risks of the Digital Twin Metaphor**
  Philosophy & Technology 2026
  *Examines risks of digital twin metaphor for personalized LLMs, including griefbots and commercial clone applications.*

- **Healthcare Digital Twins: Methodological Review**
  ScienceDirect 2025
  *Multi-stage modeling review of digital twin applications in IoT-AI healthcare (2017-2025).*

### Policy & Organizations
- **Roleplay-Doh: Enabling Domain-Experts to Create LLM-Simulated Patients**
  Louie et al. | 2024
  *Allows domain experts (not ML engineers) to create simulated patients via principle elicitation.*

- **Simulating Human Behavior with AI Agents**
  Stanford HAI Policy Brief 2024
  *Policy brief on using generative agents as research tools for policy analysis.*

- **Leveraging LLM-Based Agents for Social Science Research**
  Nature Humanities & Social Sciences Communications 2025
  *CiteAgent framework for simulation-based social science research.*

---

## 13. Generative Ghosts & AI Afterlives

- **Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives**
  Morris, Brubaker | CHI 2025 (arXiv:2402.01662)
  *Design space for AI afterlives with 7 dimensions. Benefits: legacy preservation, emotional support. Risks: delayed grief, privacy breaches, security threats.*

- **The Code That Binds Us: Navigating Human-AI Assistant Relationships**
  Manzini, Keeling, Alberts, Vallor, Morris, Gabriel | AIES 2024
  *Examines appropriateness of human-AI assistant relationships across the spectrum from assistants to clones to ghosts.*

- **Griefbots, Deadbots, Postmortem Avatars: Responsible Applications in Digital Afterlife**
  Philosophy & Technology 2024
  *Examines responsible applications of generative AI in the digital afterlife industry.*

- **The Many Faces of Indeterminacy in Interactive Deadbots**
  PMC 2025
  *Philosophical analysis of indeterminacy in AI systems that simulate deceased individuals.*

---

## 14. Surveys & Meta-Analyses

- **Large Language Models Empowered ABM: A Survey**
  Gao et al. | Nature Humanities & Social Sciences Communications 2024
  *THE go-to survey for the intersection of LLMs and agent-based modeling.*

- **From Individual to Society: Social Simulation Driven by LLM-Based Agents**
  Fudan University | arXiv 2024
  *Systematic progression from individual role-playing agents to societal-scale simulation.*

- **The Rise and Potential of LLM-Based Agents: A Survey**
  Science China Information Sciences 2025
  *Broad survey of LLM agent construction frameworks, applications, and multi-agent societies.*

- **LLM-Based Multi-Agents: Progress and Challenges**
  IJCAI 2024
  *Overview of LLM-based multi-agent systems including fundamental concepts and applications.*

- **Cognitive Architectures for Language Agents (CoALA)**
  Sumers et al. | TMIR 2024
  *Canonical framework organizing LLM agent architectures around memory, reasoning, and action.*

- **Memory in the Age of AI Agents: A Survey**
  arXiv 2025 (arXiv:2512.13564)
  *Most comprehensive survey on agent memory systems to date.*

- **A Survey on the Memory Mechanism of LLM-Based Agents**
  ACM Computing Surveys 2025
  *Detailed review of memory design and evaluation in LLM agents.*

- **Validation is the Central Challenge for Generative Social Simulation**
  Springer 2025
  *Critical survey on validation methods and challenges in LLM-based social simulation.*

- **Awesome LLM-Based Human Simulation (ICLR 2025 Blog Post)**
  GitHub: Persdre/awesome-llm-human-simulation
  *Curated list spanning psychology, economics, education, political science, and AI safety.*

---

## 15. Key Repositories & Resources

| Resource | URL | Description |
|----------|-----|-------------|
| Awesome Agent Papers | github.com/luo-junyu/Awesome-Agent-Papers | Comprehensive collection of LLM agent papers |
| SocialAgent (Fudan) | github.com/FudanDISC/SocialAgent | Resources for LLM-based social simulation |
| SocioVerse | github.com/FudanDISC/SocioVerse | 10M-user social simulation framework |
| Agent Memory Paper List | github.com/Shichun-Liu/Agent-Memory-Paper-List | Papers on agent memory systems |
| LLM Multi-Agents Survey Papers | github.com/taichengguo/LLM_MultiAgents_Survey_Papers | Categorized multi-agent papers |
| LLM ABM Simulation | github.com/tsinghua-fib-lab/LLM-Agent-Based-Modeling-and-Simulation | Tsinghua's ABM+LLM survey & code |
| LLM Agents Simulation Framework | github.com/PRAISELab-PicusLab/LLM-Agents-Simulation-Framework | GABM framework for emergent behaviors |
| Awesome LLM-Psychometrics | github.com/ValueByte-AI/Awesome-LLM-Psychometrics | LLMs in psychological measurement |
| Awesome LLM in Social Science | github.com/ValueByte-AI/Awesome-LLM-in-Social-Science | LLM applications in social science |
| Awesome LLM Human Simulation | github.com/Persdre/awesome-llm-human-simulation | LLM-based human simulation papers |
| LLM Agent Benchmark List | github.com/zhangxjohn/LLM-Agent-Benchmark-List | Benchmarks for LLM agent evaluation |
| CS222 Course Page | joonspk-research.github.io/cs222-fall24 | Stanford course page with slides |

---

## Research Landscape Summary

### Key Trends (2024-2026)
1. **Scale**: Moving from 25 agents (Park 2023) → 1,000 (Park 2024) → 10M (SocioVerse 2025)
2. **Grounding**: Shift from synthetic personas to real interview/census data
3. **Validation**: Growing emphasis on rigorous validation bridging micro/macro fidelity
4. **Memory**: Mature frameworks for episodic, semantic, procedural memory in agents
5. **Multi-agent emergence**: Social conventions, norms, and collective bias emerge spontaneously
6. **Ethics debate**: Active tension between "LLMs can simulate humans" vs "LLMs cannot represent identity groups"
7. **Platforms**: Multiple large-scale platforms (AgentSociety, SocioVerse) enabling reproducible research
8. **Applications**: Rapid adoption in clinical simulation, policy analysis, and education

### Open Challenges
- **Validation gap**: No consensus methodology for validating generative social simulations
- **Representation bias**: LLMs systematically misrepresent certain populations
- **Effect size inflation**: LLMs produce larger effects than human studies
- **Scalability-cost tradeoff**: Large-scale simulations remain expensive
- **Chaos vs. equilibrium**: Understanding when GABMs converge vs. diverge
- **Ethical boundaries**: Consent, privacy, and autonomy in simulating real individuals
