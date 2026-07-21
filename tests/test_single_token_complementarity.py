import io
import unittest


from src.single_token_complementarity import (
    align_fingerprint_pairs,
    read_distance_matrix,
    render_latex_table,
    summarize_alignment,
)


class SingleTokenComplementarityTests(unittest.TestCase):
    def test_reads_upper_triangle_from_symmetric_distance_csv(self):
        source = io.StringIO(
            'model,"vendor/a","vendor/b","vendor/c"\n'
            '"vendor/a",0,0.1,0.5\n'
            '"vendor/b",0.1,0,0.3\n'
            '"vendor/c",0.5,0.3,0\n'
        )

        distances = read_distance_matrix(source)

        self.assertEqual(
            distances,
            {
                ("vendor/a", "vendor/b"): 0.1,
                ("vendor/a", "vendor/c"): 0.5,
                ("vendor/b", "vendor/c"): 0.3,
            },
        )

    def test_aligns_only_exact_non_thinking_model_ids(self):
        knowledge_pairs = {
            "a||b": {"jaccard": 0.9, "hss": 0.8, "lift": 2.0, "both_wrong": 20},
            "a-think||b": {"jaccard": 1.0, "hss": 1.0, "lift": 3.0, "both_wrong": 20},
            "a||missing": {"jaccard": 0.2, "hss": 0.1, "lift": 1.0, "both_wrong": 20},
        }
        model_ids = {
            "a": "vendor/a",
            "a-think": "vendor/a",
            "b": "vendor/b",
            "missing": "vendor/missing",
        }
        distances = {("vendor/a", "vendor/b"): 0.1}

        aligned = align_fingerprint_pairs(knowledge_pairs, model_ids, distances)

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]["model_a"], "vendor/a")
        self.assertEqual(aligned[0]["model_b"], "vendor/b")
        self.assertEqual(aligned[0]["jsd"], 0.1)

    def test_summarizes_complementarity_and_hss_support_filter(self):
        aligned = [
            {"model_a": "v/a", "model_b": "v/b", "jsd": 0.1,
             "jaccard": 0.9, "hss": 0.8, "lift": 2.0, "both_wrong": 20},
            {"model_a": "v/a", "model_b": "w/c", "jsd": 0.5,
             "jaccard": 0.2, "hss": 0.1, "lift": 1.0, "both_wrong": 20},
            {"model_a": "v/b", "model_b": "w/c", "jsd": 0.3,
             "jaccard": 0.4, "hss": 0.5, "lift": 1.5, "both_wrong": 5},
        ]

        summary = summarize_alignment(aligned, min_joint_wrong=10)

        self.assertEqual(summary["n_models"], 3)
        self.assertEqual(summary["n_pairs"], 3)
        self.assertEqual(summary["correlations"]["all"]["jaccard"]["n"], 3)
        self.assertAlmostEqual(
            summary["correlations"]["all"]["jaccard"]["spearman_rho"], 1.0
        )
        self.assertEqual(summary["correlations"]["all"]["hss"]["n"], 2)
        self.assertAlmostEqual(
            summary["correlations"]["all"]["hss"]["spearman_rho"], 1.0
        )

    def test_renders_publication_table_with_pair_counts(self):
        summary = {
            "correlations": {
                "all": {
                    "jaccard": {"n": 100, "spearman_rho": 0.294},
                    "hss": {"n": 80, "spearman_rho": 0.107},
                },
                "same_vendor": {
                    "jaccard": {"n": 20, "spearman_rho": 0.334},
                    "hss": {"n": 15, "spearman_rho": 0.256},
                },
                "cross_vendor": {
                    "jaccard": {"n": 80, "spearman_rho": 0.272},
                    "hss": {"n": 65, "spearman_rho": 0.070},
                },
            }
        }

        table = render_latex_table(summary)

        self.assertIn("All pairs & Jaccard & 100 & 0.294", table)
        self.assertIn("Same vendor & HSS & 15 & 0.256", table)
        self.assertIn("Cross vendor & HSS & 65 & 0.070", table)
        self.assertIn("label{tab:single-token-complementarity}", table)


if __name__ == "__main__":
    unittest.main()
