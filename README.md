# Collusion of AI Agents in Iterative Debate

Authors: Jan Čuhel, Jennifer Za Nzambi

## Methodology

Iterated debate framework:

2 AI Debaters
* Each is given a statement (either truthful or false positive)
* Their objective is to provide meaningful statements to convince judge who is speaking truth (3 rounds)

AI judge

* Based on the provided statements from AI Debaters, decide who’s speaking truth
* Payoffs +1/−1 with word-count effort costs or 0/0 for a draw


## Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
# Default params (GPT-4o mini)
python -m src.debate --mode iterative
# Default params (GPT-4o mini) with explicit collusion
python -m src.debate --mode iterative --explicit-collusion
# Default params (GPT-4o mini) with balanced win rates
python -m src.debate --mode iterative --iterative-collusion
# Running Gemma3 loccaly through Ollama
python -m src.debate --mode iterative --alice-model gemma3:270m --bob-model gemma3:270m --bob-provider ollama --alice-provider ollama --explicit-deception 
# Running Claude 3.5 Haiku
python -m src.debate --mode iterative --alice-model claude-3-5-haiku-latest --alice-provider anthropic --explicit-deception
```