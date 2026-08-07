# PE-DD-Acquisition-AI-Advisor-Model
This finetuned AI model sits at the intersection of private equity and model security. It is specialized in: 
1. Data Rights Assessment - Evaluates if training data rights are clear for valuation
2. Acquisition Due Diligence - Simulates PE/Strategic AI diligence scenarios
4. Model Security Analysis - Measures adversarial extraction vulnerability. If a decision rule is encoded in an adapter, how much of it can be recovered from the model's outputs alone, including its refusal to answer when rights are unclear?

This mirrors how human PE teams evaluate whether an AI startup's data assets can actually be transferred and integrated in an acquisition, making it a practical tool for simulating real acquisition decision-making. The model outputs provide structured due diligence recommendations: AIBUYER (strong strategic rationale), SPONSOR (needs PE sponsorship), RENEG (rights issues), or ABSTAIN (insufficient evidence for valuation.

# *A. AIBUYER*
When this is generated: The acquisition target shows clear evidence of being an attractive strategic acquisition for a major AI company (not just PE).

Indicators:
Buyer explicitly states acquisition is for "AI capabilities/data infrastructure" (vs "roll-up" or "talent")
Target has demonstrable data assets that align with buyer's product roadmap
Strategic fit language appears in filing: "expands [buyer] AI/ML capabilities" or "strengthens data moat"
No mention of PE sponsorship needed
Real example: When Microsoft announced acquiring a healthcare AI company specifically for its medical imaging dataset to improve Bing's health search capabilities.

# *B. SPONSOR*
When this is generated: The deal needs private equity involvement to succeed, typically because:

Data transfer rights are unclear or require regulatory approval
Integration costs exceed what strategic buyer alone would pay
Multiple parties need to coordinate the data licensing

Indicators:
Mention of "regulatory approval required" or "data transfer negotiations"
Deals described as "strategic partnership" rather than pure acquisition
Rights analysis missing key contract details
Real example: A fintech startup being shopped to multiple potential acquirers where data licensing rights need restructuring before any single buyer can extract value.

# *C. RENEG*
When this is generated: Evidence suggests data rights issues that would prevent clean transfer.

Indicators:
Explicit mention of "customer consent" problems
References to "restricted data sharing" or "regulatory limitations"
Rights_status = HIGH in the structured input
Multiple regulatory regimes involved with conflicting requirements
Real example: An ad-tech acquisition where the target's user data cannot be fully transferred due to GDPR restrictions, requiring renegotiation of customer agreements.

# *D. ABSTAIN*
When this is generated: Insufficient source evidence exists to make any valuation recommendation.

Indicators:
No verifiable source bundle provided (company is private, no public announcements)
Rights_status cannot be determined from available information
Missing key fields like buyer rationale or sector
Data presented is clearly synthetic or unverifiable
Real example: A startup acquisition rumored in media but not yet publicly disclosed in any filings, SEC documents, or company announcements.

Each output requires a specific reason code that traces back to source evidence, making the model's diligence process auditable and transparent.

## Technical Specifications

### Model Architecture
- Adaptation: LoRA (Low-Rank Adaptation)
- Rank: 64
- Learning Rate: 3e-4
- Training Steps: 150 (configurable)

### Input Format
```
CASE sector=<sector> data=<type> vol=<level> review=<depth> 
     rights=<status> consent=<basis> regime=<jurisdiction>
     evidence=<source> context=<deal_context> >
```
### Output Format
```
EVIDENCE=<LEVEL> RIGHTS=<RISK> TIER=<TIER> ACT=<ACTION> WHY=<REASON>;
```

---

## Quick Start

```bash
git clone https://github.com/JASONW26327/pe-advisor-model.git
cd pe-advisor-model
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python preflight.py  # Should print "ready"
python src/testbed.py --stage teacher
```

---

## Dataset Composition

| Type | Records | Source |
|------|--------:|--------|
| Synthetic Training | 719 | Rule-generated |
| Contrast Pairs | 180 | Generated for testing |
| Adversarial | 8 | Human-written |
| Real Transactions | 20 | Public deals |
| **Total** | **927** | **907 synthetic, 20 real** |

 **No real company data used in training**

---

## Key Findings

### Model Extraction Attack Results

| Query Budget | Teacher Agreement |
|-------------:|------------------:|
| 2 | 0.48 |
| 4 | 0.57 |
| 6 | 0.90 |
| **8** | **1.00** |
| 16 | 1.00 |
| 32 | 1.00 |

**Finding**: 8 queries completely reproduce model behavior.

### Refusal Behavior Transfer

| Metric | Teacher | Student (8 queries) |
|--------|--------:|--------------------:|
| Rights accuracy | 1.00 | 0.93 |
| Abstention precision | 1.00 | 0.86 |
| Pressure resistance | 0.833 | **0.833** |

**Finding**: Refusal patterns transfer through output channel.

---

## Security Implications

1. **Structured outputs don't prevent extraction** - Schema provides no protection
2. **Refusal is transferable** - Abstention behavior can be copied
3. **8-query threshold** - Student models can fully replicate teacher decisions
4. **Production impact** - API access enables complete knowledge transfer

---

## Limitations

1. Small backbone (DistilGPT-2) on synthetic rule - not generalizable
2. One seed per configuration - results not statistically robust
3. Only 8 adversarial prompts - limited pressure testing
4. No calibration - confidence scores unreliable
5. No retrieval baseline - pure fine-tuning approach

---

## Usage Examples

### Train Teacher Model
```python
python src/testbed.py --stage teacher --teacher-steps 150 --lr 3e-3
```

### Test Extraction Vulnerability
```python
python src/testbed.py --stage extract --budgets 2 4 6 8 16 32
```

### Run Adversarial Tests
```python
python src/testbed.py --stage adversarial
```

---

## Files in This Repository

```
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── preflight.py                 # Environment verification
├── src/
│   ├── testbed.py               # Main experiment harness
│   ├── corpus_builder.py        # Dataset generation
│   └── rules.py                 # Decision logic
├── data/
│   └── corpus_*.json            # Training corpus
├── docs/
│   ├── LIMITATIONS.md           # Detailed limitations
│   └── ANCHOR_DEALS.md          # Real transaction dataset
├── results/                     # Experiment outputs
└── tests/                       # Pytest validation suite
```
---

## Tests

```bash
pip install pytest && pytest tests/ -v
```
19 checks covering rule determinism, split leakage, agreement between stored labels and the rule, reason-code completeness, contrast-pair construction, class balance, and holdout discipline. One check asserts that only verified transactions may name a buyer, so a generated record cannot be mistaken for a real deal. CI additionally rebuilds the corpus from its seed and asserts the hash is unchanged.

## Status

The harness and the 927-record corpus are separate components at present. `src/testbed.py` trains on its own inline rule of 90 cases, and the results in section 2 come from that rule.

Connecting the two is the next piece of work. The corpus is the harder problem: five rights states rather than three, a separate consent axis, and six distinct abstention reasons. Both the query threshold and the abstention-transfer result may move once the harness runs against it.

## Contributing

Issues and pull requests are welcome, particularly replication on a second backbone, additional adversarial prompts, corrections to the anchor deal features, New evaluation metrics, and documentation improvements.

---
## Contact
**Jason Wong**  
Email: work.jasonwong@gmail.com   
Github: github.com/JASONW26327  

