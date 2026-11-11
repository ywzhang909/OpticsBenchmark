# Optis Benchmark — Dataset Directory

This directory holds all the datasets needed to evaluate optical-design agents.
It currently covers four task datasets, in either the ready or the
work-in-progress state.

---

## Directory Structure

```
dataset/
├── paper_info_extract/           # [Ready] Paper information extraction task
│   ├── dataset_json/
│   │   ├── dataset_v1.json      # 15 evaluation samples (paper title + PDF path)
│   │   └── gold_answer_v1.json  # Structured gold answers with 13 fields (full annotation for each paper)
│   └── data_v1/                 # 15 original PDF papers (2015–2026, mostly metasurface/metamaterial)
├── info_extraction/              # [Ready] Extended data for information extraction
│   └── AO/                      # Adaptive Optics domain
│       └── *_analysis.txt       # AI-generated analyses of 274 papers (mixed Chinese + English)
├── paper_review/                 # [WIP] Paper review task
│   ├── dataset_json/            # (empty, awaiting JSON annotations)
│   └── data_v1/                 # (empty, awaiting the original PDFs)
├── optics_question_answer/       # [WIP] Optics question-answering task
│   └── (empty directory)        # To be generated
└── README.md                     # This file
```

---

## Dataset Details

### 1. `paper_info_extract` — Paper Information Extraction (✅ Ready)

Extracts 13 structured fields from optics papers, used to evaluate the
information-extraction ability of LLMs.

| Item | Description |
|------|-------------|
| **Sample size** | 15 papers |
| **Paper domain** | Metasurface, metamaterial, nanophotonics (2015–2026, mainly from *Advanced Functional Materials*, *Scientific Reports*, *Nanoscale* and similar journals) |
| **Extracted fields** | title, Publication Year, DOI, Journal, Ten Keywords, Authors, Contact Authors, Affiliations, Abstract, Objective, Novelty, Method, Performance Metrics |
| **Output format** | Structured JSON |
| **Source files** | 15 PDFs (located in `data_v1/`) |
| **Gold answers** | `gold_answer_v1.json` — carefully hand-annotated with 13 fields for every paper |

**Annotation example** (excerpt from `gold_answer_v1.json`):
```json
{
  "id": 2,
  "data": {
    "title": "Synthetic waves sensor enables turbulence-resistant imaging",
    "Publication Year": "2026",
    "DOI": "10.1117/12.3092313",
    "Journal": "Proceedings of SPIE Vol. 13992 (AOMATT 2025)",
    "Ten Keywords": [
      "Atmospheric Turbulence",
      "Synthetic Wave",
      "Turbulence-Resistant Imaging",
      "3D Phase Measurement",
      "Structural Similarity Index (SSIM)",
      ...
    ],
    "Authors": ["Yuting Xiao", "Yu Luo", "Lianwei Chen", "Mingbo Pu", "Xiangang Luo"],
    "Abstract": "Atmospheric turbulence severely distorts optical wavefronts...",
    "Objective": ["Core objective: Realize real-time 3D phase measurement..."],
    "Novelty": ["Realize depth-resolved 3D phase detection along the optical axis..."],
    "Method": ["Core scheme: Split the beam to generate two coherent sub-waves..."],
    "Performance Metrics": ["Phase detection error <±0.006..."]
  }
}
```

**Dataset file**:
```json
// dataset_v1.json — each entry holds a title + PDF location path
[
  {
    "title": "Synthetic waves sensor enables turbulence -resistant imaging",
    "location": ".../data_v1/13992 2R Synthetic waves sensor...pdf"
  },
  ...
]
```

**Related configuration files**:
| File | Path |
|------|------|
| Task config | `configs/llm/GPT_OpenAI.yaml` (the `task` section) |
| Evaluation config | `configs/evaluations/paper_info_extract.yaml` |
| Evaluation dimensions | Exact Match (8 fields such as title, DOI, authors), ROUGE (objective, method, metrics), BERTScore (objective, novelty, method, metrics) |

---

### 2. `info_extraction/AO` — Adaptive Optics Analysis Texts (✅ Ready)

A collection of AI-generated analysis texts for papers in the Adaptive Optics
domain, usable for tasks such as information extraction, document summarization
and RAG retrieval.

| Item | Description |
|------|-------------|
| **Sample size** | 274 analysis files |
| **Domain** | Adaptive optics (wavefront sensing, wavefront correction, atmospheric turbulence compensation, thermal blooming, deformable mirrors, deep learning in AO, etc.) |
| **Language** | Mixed Chinese + English |
| **Format** | Plain text (UTF-8, CRLF) |
| **Naming convention** | `{author} - {year} - {title}_analysis.txt` |
| **Content structure** | Each txt file contains a natural-language description of key information: paper title, authors, year, core method, performance metrics, etc. |
| **Source paper span** | ~1977–2026, covering both classic literature and the latest research |

**File examples**:
| Filename | Topic |
|----------|-------|
| `Nishizaki 等 - 2019 - Deep learning wavefront sensing_analysis.txt` | Deep-learning wavefront sensing |
| `Wang 等 - 2021 - Deep learning wavefront sensing and aberration correction...txt` | Deep-learning atmospheric-turbulence correction |
| `Vorontsov 等 - 2024 - Analysis of speckle-beacon size impact...txt` | Deep-turbulence wavefront sensing |
| `Zhang 等 - 2024 - Shack-hartmann wavefront sensor based on...txt` | Diffractive lens array SHWFS |
| `Zhao 等 - 2025 - Intensity adaptive optics_analysis.txt` | Intensity-based adaptive optics |
| `吴书云 - 激光大气传输热晕效应及自适应光学校正的数值仿真和实验研究_analysis.txt` | Thermal blooming (Chinese) |

---

### 3. `paper_review` — Paper Review (⏳ Work in Progress)

Performs an academic review of optics papers and returns scores plus suggestions.

| Item | Description |
|------|-------------|
| **Status** | The `dataset_json/` and `data_v1/` directories exist but are empty |
| **Planned evaluation methods** | ROUGE-L, content coverage, coherence, factuality |

---

### 4. `optics_question_answer` — Optics Question Answering (⏳ Work in Progress)

Question-answering evaluation over optics domain knowledge.

| Item | Description |
|------|-------------|
| **Status** | The directory exists but is empty |
| **Planned task types** | Knowledge Q&A on optical principles, design specifications, material properties, etc. |

---

## Paper Domain Coverage

| Domain | Paper count | Source |
|--------|-------------|--------|
| Metasurface / Metamaterial | 15 | `paper_info_extract/data_v1/` (PDF) |
| Adaptive Optics | 274 | `info_extraction/AO/` (analysis texts) |
| **Total** | **289** | |

Key topics covered by the AO dataset:
- Wavefront sensing (Shack-Hartmann, pyramid sensors, phase retrieval, deep-learning wavefront sensing)
- Wavefront correction (SPGD, PID, reinforcement learning, model predictive control)
- Atmospheric turbulence compensation (thermal blooming, deep turbulence, beacon-assisted)
- Deformable mirrors (MEMS, piezoelectric, liquid-crystal spatial light modulators)
- Laser communication and coherent beam combining
- Adaptive-optics microscopy and biological imaging
- Wavefront-sensorless adaptive optics (sensorless AO)

---

## Data Format Specification

### paper_info_extract JSON (annotations)

```json
{
  "id": 1,
  "data": {
    "title": "<paper title>",
    "Publication Year": "<year>",
    "DOI": "<full DOI>",
    "Journal": "<journal name>",
    "Ten Keywords": ["<keyword list>"],
    "Authors": ["<author list>"],
    "Contact Authors": ["<corresponding author list>"],
    "Affiliations": ["<affiliation list>"],
    "Abstract": "<abstract text>",
    "Objective": ["<objective list>"],
    "Novelty": ["<novelty list>"],
    "Method": ["<method description list>"],
    "Performance Metrics": ["<performance metric list>"]
  }
}
```

### paper_info_extract JSON (evaluation samples)

```json
{
  "title": "<paper title, used for matching>",
  "location": "<absolute path of the PDF file>"
}
```

### info_extraction/AO TXT (analysis texts)

Plain-text files of roughly 500–3000 characters each, containing:
- Paper metadata (title, authors, year, journal)
- Core method and novelty
- Experimental design and key results
- Performance metric values

---

## Using the Data

```bash
# Inspect dataset samples
uv run python -c "
import json
with open('dataset/paper_info_extract/dataset_json/dataset_v1.json') as f:
    data = json.load(f)
print(f'Samples: {len(data)}')
print(f'Papers: {[d[\"title\"][:50] for d in data]}')
"

# Inspect the gold answer fields
uv run python -c "
import json
with open('dataset/paper_info_extract/dataset_json/gold_answer_v1.json') as f:
    gold = json.load(f)
print(f'Gold entries: {len(gold)}')
print(f'Fields: {list(gold[0][\"data\"].keys())}')
"
```

---

## Data Preparation

```bash
# Place the data manually into the corresponding subdirectories,
# or clone the data from the remote repository
```

---

## Contributing a New Dataset

1. Create a subdirectory named after the task under `dataset/`
2. Provide a JSON annotation file plus the original input files (e.g. PDFs)
3. Make sure the data follows the format defined in the task config
4. Configure the dataset path in the `task` section of `configs/llm/*.yaml`
5. Configure the evaluation methods in `configs/evaluations/<task_name>.yaml`
6. Submit a pull request

---

## License

The datasets are licensed under CC BY-SA 4.0.
