# GDELT + LLMs/Agents Research Compilation

A comprehensive overview of research papers, tools, and approaches combining the GDELT database with Large Language Models and AI agents for geopolitical analysis, event forecasting, and knowledge extraction.

---

## Table of Contents

1. [Academic Papers](#academic-papers)
   - [MIRAI: Benchmarking LLM Agents as Geopolitical Forecasters](#mirai-benchmarking-llm-agents-as-geopolitical-forecasters)
   - [Do Large Language Models Know Conflict?](#do-large-language-models-know-conflict)
   - [Talking to GDELT Through Knowledge Graphs](#talking-to-gdelt-through-knowledge-graphs)
   - [Project Oracle](#project-oracle)
   - [GPT4MTS](#gpt4mts)
2. [GDELT Official Experiments](#gdelt-official-experiments)
3. [Tools & Platforms](#tools--platforms)
4. [Key Research Themes](#key-research-themes)
5. [References](#references)

---

## Academic Papers

### MIRAI: Benchmarking LLM Agents as Geopolitical Forecasters

**Paper:** arXiv:2407.01231
**Institutions:** UCLA, Caltech
**URL:** https://arxiv.org/abs/2407.01231

#### Overview
MIRAI is the first comprehensive benchmark for evaluating LLM-based autonomous agents as geopolitical event forecasters. The benchmark uses GDELT as the primary data source for historical events and real-time monitoring.

#### Key Contributions
- **Agent Architecture:** ReAct-style agents with two action types:
  - "Single Function": Direct tool invocations
  - "Code Block": Multi-step programmatic workflows
- **Evaluation Framework:** Systematic assessment of forecasting accuracy across different geopolitical event types
- **Data Integration:** Direct querying of GDELT's realtime API for current event context

#### Findings
- **Best Performance:** GPT-4o achieved 29.6% F1 score (highest among tested models)
- **RAG Benefits:** Retrieval-Augmented Generation approaches consistently outperformed parametric-only methods
- **Action Type Impact:** "Code Block" actions enabled more complex multi-source analysis
- **Temporal Awareness:** Agents with access to historical GDELT data showed improved forecasting

#### Implications
- Demonstrates feasibility of autonomous geopolitical analysis systems
- Highlights importance of real-time data access for accurate forecasting
- Establishes baseline metrics for future agent development

---

### Do Large Language Models Know Conflict?

**Paper:** arXiv:2505.09852
**Focus:** Parametric vs. RAG-based conflict forecasting

#### Overview
This paper investigates whether LLMs can predict conflict events using their parametric knowledge alone versus when augmented with retrieved GDELT data.

#### Key Questions
1. How much geopolitical knowledge is encoded in LLM parameters?
2. Does RAG with GDELT data improve conflict prediction?
3. What are the limitations of each approach?

#### Findings
- **Parametric Knowledge:** LLMs contain substantial implicit knowledge about geopolitical patterns but struggle with recent events beyond training cutoff
- **RAG Improvement:** Augmenting with GDELT data significantly improves:
  - Recency-aware predictions
  - Geographic specificity
  - Event type classification accuracy
- **Hybrid Approaches:** Combining parametric reasoning with retrieved context yields best results

#### Methodology
- Systematic comparison of predictions with/without retrieval
- Evaluation against ground truth GDELT events
- Analysis across different conflict types and regions

---

### Talking to GDELT Through Knowledge Graphs

**Paper:** arXiv:2503.07584
**Focus:** LLM-based KG construction and querying

#### Overview
Presents an approach for constructing and querying Knowledge Graphs from GDELT data using LLMs, enabling natural language interaction with the database.

#### Architecture
1. **KG Construction:** LLMs extract entities and relationships from GDELT event records
2. **Schema Design:** Ontology aligned with CAMEO taxonomy
3. **Natural Language Interface:** Users query KG in plain English
4. **Reasoning Layer:** Multi-hop inference over connected events

#### Key Features
- **Entity Resolution:** Disambiguates actors across different name variations
- **Temporal Reasoning:** Tracks relationship evolution over time
- **Causal Inference:** Identifies potential cause-effect chains in event sequences
- **Subgraph Extraction:** Retrieves relevant context for specific queries

#### Applications
- "What events led to the escalation between X and Y?"
- "Show me cooperation trends between countries A and B over the past year"
- "Identify potential conflict hotspots based on recent protest patterns"

---

### Project Oracle

**Institution:** Stanford University
**Focus:** Autoregressive event prediction using Transformers

#### Overview
Project Oracle applies transformer architectures to GDELT's historical event sequences for next-event prediction and trend forecasting.

#### Approach
- **Sequence Modeling:** Treats event streams as sequences for autoregressive prediction
- **Multi-scale Analysis:** Captures patterns at daily, weekly, monthly granularities
- **Event Embedding:** Learns dense representations of CAMEO event types

#### Results
- Competitive performance on next-event prediction
- Ability to capture long-range dependencies in conflict dynamics
- Interpretable attention patterns correlating with real-world dynamics

---

### GPT4MTS

**Venue:** AAAI 2024
**Focus:** Multimodal time-series forecasting with GDELT

#### Overview
Combines GDELT event data with other modalities (text, numerical indicators) for enhanced time-series forecasting.

#### Innovation
- **Multimodal Fusion:** Integrates GDELT events with economic indicators, news text, and social signals
- **Cross-modal Attention:** Learns relationships between event types and outcome variables
- **Temporal Encoding:** Specialized embeddings for event timestamps

---

## GDELT Official Experiments

The GDELT Project blog has published several experiments combining GDELT with LLMs:

### Blog Series Highlights

1. **Natural Language Querying**
   - Experiments with using LLMs to translate natural language into BigQuery SQL
   - Enables non-technical users to explore GDELT data

2. **Event Summarization**
   - LLM-generated summaries of daily/weekly event clusters
   - Automated situation reports from raw event streams

3. **Tone Analysis Enhancement**
   - Combining GDELT's AvgTone with LLM sentiment analysis
   - More nuanced emotional interpretation of coverage

4. **Entity Extraction Validation**
   - Using LLMs to validate and expand GDELT's actor extraction
   - Cross-referencing with external knowledge bases

---

## Tools & Platforms

### GDELT Cloud
- Cloud-native access to GDELT data
- API integrations for real-time applications
- Compatible with LLM tool use patterns

### GDELT Guru
- Experimental LLM-powered interface
- Natural language querying of GDELT
- Conversational exploration of event data

### VANTAGE
- Visual analytics platform
- Integrates GDELT with LLM-generated insights
- Dashboard for geopolitical monitoring

### Warcast
- Conflict forecasting tool
- Uses GDELT as primary data source
- LLM-augmented analysis and reporting

### Academic Tools
- Various research prototypes for KG construction
- Custom RAG pipelines for GDELT querying
- Agent frameworks built on GDELT API

---

## Key Research Themes

### 1. Forecasting & Prediction
- Next-event prediction using sequence models
- Long-term trend forecasting
- Conflict escalation/de-escalation prediction
- Economic/political outcome forecasting

### 2. Event Extraction & Classification
- Improving CAMEO classification accuracy
- Fine-grained sub-event detection
- Cross-lingual event alignment
- Duplicate event resolution

### 3. Knowledge Graph Construction
- Entity-relationship extraction
- Temporal KG reasoning
- Multi-hop inference
- Subgraph retrieval for RAG

### 4. Autonomous Agents
- ReAct-style reasoning agents
- Multi-source information gathering
- Self-correcting analysis pipelines
- Tool-augmented forecasting

### 5. Natural Language Interfaces
- Text-to-SQL for GDELT BigQuery
- Conversational querying
- Automated report generation
- Multi-document summarization

---

## References

### Papers
1. MIRAI: Benchmarking LLM Agents as Geopolitical Forecasters - https://arxiv.org/abs/2407.01231
2. Do Large Language Models Know Conflict? - https://arxiv.org/abs/2505.09852
3. Talking to GDELT Through Knowledge Graphs - https://arxiv.org/abs/2503.07584
4. GPT4MTS - AAAI 2024

### Resources
- GDELT Project: https://gdeltproject.org/
- GDELT Documentation: https://www.gdeltproject.org/data.html
- GDELT Blog: https://blog.gdeltproject.org/
- CAMEO Codebook: https://parusanalytics.com/eventdata/cameo.html

---

*Last updated: March 2026*
