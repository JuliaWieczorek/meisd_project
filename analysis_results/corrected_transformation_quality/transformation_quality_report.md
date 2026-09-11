# Corrected transformation-only quality statistics

| method | transformed_instances | low_instances | high_instances | exact_uniqueness_ratio | mean_length_words | sd_length_words | mean_quality_score | sd_quality_score | minimum_quality_score | maximum_quality_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule | 2288 | 975 | 1313 | 0.783654 | 18.782343 | 10.795544 | 0.264615 | 0.106240 | 0.003309 | 0.555525 |
| rule_llm | 2288 | 975 | 1313 | 0.974650 | 46.057692 | 25.870810 | 0.436798 | 0.160490 | 0.003846 | 0.867158 |
| llm | 2288 | 975 | 1313 | 0.997378 | 58.108829 | 21.730933 | 0.512547 | 0.118797 | 0.003309 | 0.899752 |
| nlp | 2288 | 975 | 1313 | 0.947552 | 16.582168 | 9.603794 | 0.221264 | 0.117526 | 0.003309 | 0.519972 |
| nlp_llm | 2288 | 975 | 1313 | 0.980769 | 57.696241 | 27.269581 | 0.480942 | 0.141660 | 0.007691 | 0.836297 |

Quality score: `0.4 * Q_length + 0.4 * Q_keyword + 0.2 * Q_pronoun`.
The summary evaluates transformed rows only; see the manifest for input hashes and target patterns.
