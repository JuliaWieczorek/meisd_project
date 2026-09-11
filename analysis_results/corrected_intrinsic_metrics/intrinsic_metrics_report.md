# Corrected intrinsic augmentation metrics

| method | total_instances | transformed_instances | full_set_uniqueness_ratio | transformed_uniqueness_ratio | self_bleu | ttr | utr | novelty | bleu | chrf | perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule | 4738 | 2288 | 0.895526 | 0.783654 | 0.854776 | 0.065225 | 0.383939 | 0.156788 | NA | NA | 68.078664 |
| rule_llm | 4738 | 2288 | 0.987547 | 0.974650 | 0.717643 | 0.035329 | 0.409959 | 0.601422 | NA | NA | 21.932143 |
| llm | 4738 | 2288 | 0.998523 | 0.997378 | 0.770544 | 0.027431 | 0.360490 | 0.798656 | NA | NA | 18.634621 |
| nlp | 4738 | 2288 | 0.974673 | 0.947552 | 0.613647 | 0.081339 | 0.597258 | 0.269541 | NA | NA | 98.442223 |
| nlp_llm | 4738 | 2288 | 0.990713 | 0.980769 | 0.810016 | 0.025566 | 0.295669 | 0.805860 | NA | NA | 23.106716 |

`NA` means that the metric was not computed. See `intrinsic_metrics_manifest.json` for the reason and full configuration.
