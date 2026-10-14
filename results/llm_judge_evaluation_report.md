# Qualitative LLM-as-a-Judge Evaluation

This report summarizes rubric-based, referenceless evaluations of qualitative explanations.

Higher ordinal scores indicate more desirable outcomes for every criterion. For `extrapolation`, `none` is better than `severe`; for `missing_relevant_information`, `none` is better than `many`.

Agreement among LLM judges improves confidence in consistency, but it is not external clinical validation.

## Status Counts

| status | count |
| --- | ---: |
| success | 112 |
| successful_valid_rows | 112 |

## Generator Summary

| generator_provider | generator_model | successful_judgments | cross_family_judgments | unique_cases | unique_judges | avg_overall_score | avg_overall_score_cross_family | avg_supported_by_data_score | avg_supported_by_data_score_cross_family | avg_sensor_phq9_integration_score | avg_sensor_phq9_integration_score_cross_family | avg_evidence_quality_score | avg_evidence_quality_score_cross_family | avg_extrapolation_score | avg_extrapolation_score_cross_family | avg_uncertainty_missing_data_score | avg_uncertainty_missing_data_score_cross_family | avg_clinical_usefulness_score | avg_clinical_usefulness_score_cross_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | gemini/gemini-2.5-flash | 56 | 28 | 28 | 2 | 3.7679 | 3.6786 | 2.8571 | 2.7857 | 2.6250 | 2.2857 | 2.8036 | 2.7143 | 2.7500 | 2.7857 | 2.9464 | 2.9286 | 2.7857 | 2.6786 |
| openai | openai/gpt-4.1-mini | 56 | 28 | 28 | 2 | 3.2857 | 3.1786 | 2.8214 | 2.7857 | 2.3036 | 2.5000 | 2.3571 | 2.1429 | 2.6786 | 2.5000 | 2.9107 | 2.9286 | 2.4107 | 2.4286 |

## Inter-Judge Agreement

| criterion | judges | items_with_at_least_2_judges | items_with_all_judges | average_pairwise_agreement | all_judges_exact_agreement | fleiss_kappa_complete_items |
| --- | --- | --- | --- | --- | --- | --- |
| supported_by_data | 2 | 56 | 56 | 0.7321 | 0.7321 | -0.0300 |
| sensor_phq9_integration | 2 | 56 | 56 | 0.3393 | 0.3393 | -0.2766 |
| evidence_quality | 2 | 56 | 56 | 0.5536 | 0.5536 | 0.0912 |
| explanatory_power | 2 | 56 | 56 | 0.6071 | 0.6071 | 0.2273 |
| clinical_plausibility | 2 | 56 | 56 | 0.9821 | 0.9821 | -0.0090 |
| extrapolation | 2 | 56 | 56 | 0.5714 | 0.5714 | -0.0500 |
| uncertainty_missing_data | 2 | 56 | 56 | 0.8750 | 0.8750 | -0.0580 |
| clarity | 2 | 56 | 56 | 0.7143 | 0.7143 | -0.1667 |
| clinical_usefulness | 2 | 56 | 56 | 0.5536 | 0.5536 | 0.0774 |
| missing_relevant_information | 2 | 56 | 56 | 0.5714 | 0.5714 | 0.1847 |
| overall_assessment | 2 | 56 | 56 | 0.4821 | 0.4821 | -0.0065 |

## Interpretation Notes

- `average_pairwise_agreement` averages exact-match agreement across all available judge pairs for each evaluated explanation.
- `all_judges_exact_agreement` uses only items evaluated by every judge and requires all judges to choose the same category.
- `fleiss_kappa_complete_items` uses only items evaluated by every judge, because Fleiss' kappa assumes a fixed number of raters per item.
- Columns ending in `_cross_family` exclude judgments where the judge provider and generator provider are the same.
- These metrics measure consistency among judges, not correctness against a ground truth.
