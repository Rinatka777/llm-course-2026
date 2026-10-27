# LLMs and Generative AI for NLP: course project

My final project for the University of Helsinki course *LLMs and Generative AI
for NLP* (LDA-T318), autumn 2026.

Course materials: [Helsinki-NLP/LLM-Course-2026](https://github.com/Helsinki-NLP/LLM-Course-2026)

## Project

- **Task:** TBD
- **Model:** TBD
- **Held-out set:** TBD

See [`project-plan.yaml`](project-plan.yaml).

## Layout

| Path | Contents |
|---|---|
| `week-1/` … `week-6/` | Lab work adapted to the project task |
| `report/` | Final report |
| `project-plan.yaml` | Task, model and data choices |

## Setup

The course repository is kept as a separate, unmodified clone so that
`git pull` never conflicts with project work:

```bash
git clone https://github.com/Helsinki-NLP/LLM-Course-2026.git
cd LLM-Course-2026 && git pull   # fetch new course material
```

It is ignored by this repository. Each week's adapted scripts live in
`week-N/`, with their own README describing how to reproduce the results.
