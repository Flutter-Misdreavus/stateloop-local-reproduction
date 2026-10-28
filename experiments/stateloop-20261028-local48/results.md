# StateLoop local results

Automated measurements; scientific review pending.

| Variant | Seed | Test NLL | Test PPL |
|---|---:|---:|---:|
| persistent | 17 | 4.061201 | 58.0440 |
| reset | 17 | 4.074373 | 58.8136 |
| vanilla | 17 | 4.115756 | 61.2985 |
| persistent | 42 | 4.064998 | 58.2648 |
| reset | 42 | 4.081618 | 59.2412 |
| vanilla | 42 | 4.129443 | 62.1433 |

Mean paired reset-minus-persistent NLL: 0.014896 nat/token.

Document bootstrap intervals are conditional on each trained pair. See assessment.json for all interventions, run commits and limits. Failure to find an effect at this training scale does not invalidate the full-scale paper.
