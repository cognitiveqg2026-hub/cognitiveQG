"""
Reporting Module
Generate summary reports and export results in various formats
"""

import json
import numpy as np
from typing import Dict
from .schema_config import SchemaConfig


class ReportGenerator:
    """Generate comprehensive reports from IAA evaluation results"""

    def __init__(self):
        pass

    def generate_summary_report(self, results: Dict, schema: SchemaConfig) -> str:
        """
        Generate comprehensive text summary report

        Args:
            results: Evaluation results from FieldAnnotatingEvaluator
            schema: SchemaConfig instance

        Returns:
            str: Formatted text report
        """
        report = "=" * 80 + "\n"
        report += "IAA EVALUATION SUMMARY REPORT\n"
        report += "Field_Annotating_Evaluation Schema\n"
        report += "Metrics: Cohen's Kappa, Observed Agreement, PABAK, GWET AC (categorical)\n"
        report += "         BERTScore, InstructScore, Jaccard Similarity, METEOR (text similarity)\n"
        report += "=" * 80 + "\n\n"

        # Coverage Statistics
        report += self._generate_coverage_section(results, schema)

        # Overall Statistics
        report += self._generate_overall_stats(results, schema)

        # Field-by-field breakdown
        report += self._generate_field_breakdown(results, schema)

        # Problematic areas
        report += self._generate_problematic_areas(results, schema)

        # InstructScore diagnostics
        report += self._generate_instructscore_diagnostics(results, schema)

        report += "\n" + "=" * 80
        return report

    def _generate_coverage_section(self, results: Dict, schema: SchemaConfig) -> str:
        """Generate coverage statistics section"""
        section = "COVERAGE STATISTICS:\n"
        section += "-" * 50 + "\n"

        text_fields = schema.get_text_fields()
        total_coverage_stats = []

        for field in text_fields:
            if field in results:
                coverage = results[field].get('coverage', {})
                if coverage:
                    valid_pairs = coverage.get('valid_pairs', 0)
                    total_pairs = coverage.get('total_pairs', 0)
                    coverage_rate = coverage.get('coverage_rate', 0)
                    section += f"{field}: {valid_pairs}/{total_pairs} pairs ({coverage_rate:.1%} coverage)\n"
                    total_coverage_stats.append((valid_pairs, total_pairs))

        if total_coverage_stats:
            total_valid = sum(stat[0] for stat in total_coverage_stats)
            total_possible = sum(stat[1] for stat in total_coverage_stats)
            overall_coverage = total_valid / total_possible if total_possible > 0 else 0
            section += f"\nOverall Coverage: {total_valid}/{total_possible} ({overall_coverage:.1%})\n\n"

        return section

    def _generate_overall_stats(self, results: Dict, schema: SchemaConfig) -> str:
        """Generate overall statistics section"""
        section = "OVERALL STATISTICS (ALL PAIRS - includes empty):\n"
        section += "-" * 50 + "\n"

        text_fields = schema.get_text_fields()
        bert_scores, instruct_scores, jaccard_scores, meteor_scores, rouge_scores = [], [], [], [], []

        for field in text_fields:
            if field in results:
                res = results[field]
                if 'bert' in res:
                    bert_scores.append(res['bert']['mean'])
                if 'instructscore' in res:
                    instruct_scores.append(res['instructscore']['mean'])
                if 'jaccard' in res:
                    jaccard_scores.append(res['jaccard']['mean'])
                if 'meteor' in res:
                    meteor_scores.append(res['meteor']['mean'])
                if 'rouge_l' in res:
                    rouge_scores.append(res['rouge_l']['mean'])

        if bert_scores:
            section += f"Average BERT Similarity: {np.mean(bert_scores):.3f} ± {np.std(bert_scores):.3f}\n"
        if instruct_scores:
            section += f"Average InstructScore Similarity: {np.mean(instruct_scores):.3f} ± {np.std(instruct_scores):.3f}\n"
        if jaccard_scores:
            section += f"Average Jaccard Similarity: {np.mean(jaccard_scores):.3f} ± {np.std(jaccard_scores):.3f}\n"
        if meteor_scores:
            section += f"Average METEOR Score: {np.mean(meteor_scores):.3f} ± {np.std(meteor_scores):.3f}\n"
        if rouge_scores:
            section += f"Average ROUGE-L Similarity: {np.mean(rouge_scores):.3f} ± {np.std(rouge_scores):.3f}\n"

        # Non-empty statistics
        section += "\nOVERALL STATISTICS (NON-EMPTY PAIRS ONLY):\n"
        section += "-" * 50 + "\n"

        bert_scores_ne, instruct_scores_ne, jaccard_scores_ne, meteor_scores_ne, rouge_scores_ne = [], [], [], [], []

        for field in text_fields:
            if field in results:
                res = results[field]
                non_empty = res.get('non_empty_only')
                if non_empty:
                    if 'bert' in non_empty:
                        bert_scores_ne.append(non_empty['bert']['mean'])
                    if 'instructscore' in non_empty:
                        instruct_scores_ne.append(non_empty['instructscore']['mean'])
                    if 'jaccard' in non_empty:
                        jaccard_scores_ne.append(non_empty['jaccard']['mean'])
                    if 'meteor' in non_empty:
                        meteor_scores_ne.append(non_empty['meteor']['mean'])
                    if 'rouge_l' in non_empty:
                        rouge_scores_ne.append(non_empty['rouge_l']['mean'])

        if bert_scores_ne:
            section += f"Average BERT Similarity (Non-empty): {np.mean(bert_scores_ne):.3f} ± {np.std(bert_scores_ne):.3f}\n"
        if instruct_scores_ne:
            section += f"Average InstructScore Similarity (Non-empty): {np.mean(instruct_scores_ne):.3f} ± {np.std(instruct_scores_ne):.3f}\n"
        if jaccard_scores_ne:
            section += f"Average Jaccard Similarity (Non-empty): {np.mean(jaccard_scores_ne):.3f} ± {np.std(jaccard_scores_ne):.3f}\n"
        if meteor_scores_ne:
            section += f"Average METEOR Score (Non-empty): {np.mean(meteor_scores_ne):.3f} ± {np.std(meteor_scores_ne):.3f}\n"
        if rouge_scores_ne:
            section += f"Average ROUGE-L Similarity (Non-empty): {np.mean(rouge_scores_ne):.3f} ± {np.std(rouge_scores_ne):.3f}\n"

        # Categorical statistics
        categorical_fields = schema.get_categorical_fields()
        kappa_scores, observed_scores, pabak_scores, gwet_scores = [], [], [], []

        for field in categorical_fields:
            if field in results and 'cohens_kappa' in results[field]:
                kappa_scores.append(results[field]['cohens_kappa'])
                observed_scores.append(results[field]['observed_agreement'])
                pabak_scores.append(results[field]['pabak'])
                gwet_scores.append(results[field]['gwet_ac'])

        if kappa_scores:
            section += f"\nAverage Cohen's Kappa: {np.nanmean(kappa_scores):.3f} ± {np.nanstd(kappa_scores):.3f}\n"
        if observed_scores:
            section += f"Average Observed Agreement: {np.mean(observed_scores):.3f} ± {np.std(observed_scores):.3f}\n"
        if pabak_scores:
            section += f"Average PABAK: {np.mean(pabak_scores):.3f} ± {np.std(pabak_scores):.3f}\n"
        if gwet_scores:
            section += f"Average Gwet's AC: {np.mean(gwet_scores):.3f} ± {np.std(gwet_scores):.3f}\n\n"

        return section

    def _generate_field_breakdown(self, results: Dict, schema: SchemaConfig) -> str:
        """Generate field-by-field breakdown"""
        section = "FIELD-BY-FIELD RESULTS:\n"
        section += "-" * 50 + "\n\n"

        # Categorical fields
        section += "CATEGORICAL FIELDS:\n"
        for field in schema.get_categorical_fields():
            if field in results:
                res = results[field]
                section += f"\n{field}:\n"
                section += f"  Cohen's Kappa: {res['cohens_kappa']:.3f}\n"
                section += f"  Observed Agreement: {res['observed_agreement']:.3f}\n"
                section += f"  PABAK: {res['pabak']:.3f}\n"
                section += f"  Gwet's AC: {res['gwet_ac']:.3f}\n"

        # Text fields
        section += "\n\nTEXT SIMILARITY FIELDS:\n"
        for field in schema.get_text_fields():
            if field in results:
                res = results[field]
                coverage = res.get('coverage', {})
                valid_pairs = coverage.get('valid_pairs', 0)
                total_pairs = coverage.get('total_pairs', 0)

                section += f"\n{field} ({valid_pairs}/{total_pairs} valid pairs):\n"
                section += f"  ALL pairs:\n"
                section += (
                    f"    BERT: {res['bert']['mean']:.3f}, "
                    f"InstructScore: {res['instructscore']['mean']:.3f}, "
                    f"Jaccard: {res['jaccard']['mean']:.3f}, "
                    f"METEOR: {res['meteor']['mean']:.3f}\n"
                )

                non_empty = res.get('non_empty_only')
                if non_empty and valid_pairs > 0:
                    section += f"  NON-EMPTY pairs:\n"
                    section += (
                        f"    BERT: {non_empty['bert']['mean']:.3f}, "
                        f"InstructScore: {non_empty['instructscore']['mean']:.3f}, "
                        f"Jaccard: {non_empty['jaccard']['mean']:.3f}, "
                        f"METEOR: {non_empty['meteor']['mean']:.3f}\n"
                    )

                # Show sample InstructScore diagnostics if available
                instruct_data = res.get('instructscore', {})
                diagnostics = instruct_data.get('diagnostics', [])
                if diagnostics:
                    sample_diagnostics = [d for d in diagnostics if d is not None][:2]
                    if sample_diagnostics:
                        section += f"  Sample InstructScore diagnostics:\n"
                        for idx, diag in enumerate(sample_diagnostics, 1):
                            truncated_diag = diag[:150] + "..." if len(diag) > 150 else diag
                            section += f"    {idx}. {truncated_diag}\n"

        section += "\n"
        return section

    def _generate_problematic_areas(self, results: Dict, schema: SchemaConfig) -> str:
        """Identify and report problematic areas with low agreement"""
        section = "PROBLEMATIC AREAS (Low Agreement):\n"
        section += "-" * 50 + "\n"

        low_agreement_fields = []

        # Check categorical fields
        for field in schema.get_categorical_fields():
            if field in results and 'cohens_kappa' in results[field]:
                kappa = results[field]['cohens_kappa']
                if kappa < 0.4:
                    low_agreement_fields.append((field, kappa, 'categorical'))

        # Check text fields (using non-empty BERT if available)
        for field in schema.get_text_fields():
            if field in results:
                res = results[field]
                non_empty = res.get('non_empty_only')
                if non_empty and 'bert' in non_empty:
                    bert = non_empty['bert']['mean']
                    if bert < 0.5:
                        low_agreement_fields.append((field, bert, 'text_non_empty'))
                elif 'bert' in res:
                    bert = res['bert']['mean']
                    if bert < 0.5:
                        low_agreement_fields.append((field, bert, 'text_all'))

        # Sort by score (ascending)
        low_agreement_fields.sort(key=lambda x: x[1])

        if low_agreement_fields:
            for field, score, metric_type in low_agreement_fields:
                if metric_type == 'categorical':
                    section += f"{field}: Kappa {score:.3f}\n"
                elif metric_type == 'text_non_empty':
                    section += f"{field}: BERT {score:.3f} (non-empty pairs)\n"
                else:
                    section += f"{field}: BERT {score:.3f} (all pairs)\n"
        else:
            section += "None detected under current thresholds.\n"

        return section

    def _generate_instructscore_diagnostics(self, results: Dict, schema: SchemaConfig) -> str:
        """Generate detailed InstructScore diagnostic reports section"""
        section = "\nINSTRUCTSCORE DIAGNOSTIC REPORTS:\n"
        section += "-" * 50 + "\n"

        has_any_diagnostics = False

        for field in schema.get_text_fields():
            if field in results:
                res = results[field]
                instruct_data = res.get('instructscore', {})
                diagnostics = instruct_data.get('diagnostics', [])

                if not diagnostics or all(d is None for d in diagnostics):
                    continue

                # Show diagnostics for this field
                non_none_diags = [(i, d) for i, d in enumerate(diagnostics) if d is not None]
                if non_none_diags:
                    has_any_diagnostics = True
                    section += f"\n{field}:\n"
                    # Show up to 3 sample diagnostics per field
                    for idx, (pair_idx, diag) in enumerate(non_none_diags[:3], 1):
                        section += f"  Pair {pair_idx}: {diag}\n"

        if not has_any_diagnostics:
            section += "\nNo InstructScore diagnostics available (fallback mode or backend unavailable).\n"

        section += "\n"
        return section

    def save_json_results(self, results: Dict, output_path: str):
        """
        Save detailed results as JSON

        Args:
            results: Evaluation results
            output_path: Path to save JSON file
        """
        # Convert results to JSON-serializable format (remove score arrays)
        json_results = {}
        for field, metrics in results.items():
            json_results[field] = {}
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    json_results[field][metric_name] = {
                        k: v for k, v in metric_data.items()
                        if k != 'scores'  # Exclude raw score arrays
                    }
                else:
                    json_results[field][metric_name] = metric_data

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ JSON results saved to: {output_path}")

    def save_text_report(self, report: str, output_path: str):
        """
        Save text report to file

        Args:
            report: Formatted text report
            output_path: Path to save text file
        """
        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✓ Text report saved to: {output_path}")
