"""Tests for the retrieval and evaluation workflow of the Clinical NLP
Assistant.

Run with:

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from helpers import PROJECT_ROOT, load_app, read_note  # noqa: E402

app = load_app()

NOTE = read_note()

# The five demonstration questions and the evidence they must retrieve.
FIVE_CASES = [
    (
        "How long has the patient had a headache?",
        "The patient reports a headache for two days.",
    ),
    (
        "How long has the patient had a sore throat?",
        "The patient reports a sore throat for three days.",
    ),
    (
        "Does the patient have a dry cough?",
        "The patient has a dry cough.",
    ),
    (
        "Does the patient have shortness of breath?",
        "The patient denies shortness of breath.",
    ),
    (
        "What does the patient report drinking?",
        "The patient reports drinking warm water.",
    ),
]


class TestRetrievalOfTheFiveCases(unittest.TestCase):
    """Each of the five questions must retrieve its own evidence."""

    def test_all_five_questions_retrieve_the_correct_sentence(self):
        for question, expected in FIVE_CASES:
            with self.subTest(question=question):
                results = app.find_relevant_sentences(
                    NOTE, question, top_k=1
                )

                self.assertTrue(results, "no evidence retrieved")
                self.assertEqual(results[0][0], expected)

    def test_negated_case_returns_the_denial(self):
        question = "Does the patient have shortness of breath?"

        results = app.find_relevant_sentences(NOTE, question, top_k=1)

        self.assertEqual(
            results[0][0], "The patient denies shortness of breath."
        )
        self.assertTrue(app.has_negation(results[0][0]))


class TestGeneratedEvaluationCases(unittest.TestCase):
    """The generated questions must be natural and usable."""

    def test_generator_produces_the_five_natural_questions(self):
        cases = app.generate_evaluation_cases(NOTE)

        self.assertEqual(
            [case["question"] for case in cases],
            [question for question, _ in FIVE_CASES],
        )

    def test_generated_expected_evidence_comes_from_the_document(self):
        cases = app.generate_evaluation_cases(NOTE)
        sentences = app.split_sentences(NOTE)

        for case in cases:
            with self.subTest(question=case["question"]):
                self.assertIn(case["relevant_evidence"], sentences)

    def test_every_generated_question_retrieves_its_own_evidence(self):
        cases = app.generate_evaluation_cases(NOTE)

        for case in cases:
            with self.subTest(question=case["question"]):
                results = app.find_relevant_sentences(
                    NOTE, case["question"], top_k=1
                )

                self.assertTrue(results)
                self.assertEqual(
                    results[0][0], case["relevant_evidence"]
                )

    def test_generator_skips_questions_and_headings(self):
        document = (
            "Sample Clinical Note\n"
            "The patient reports a headache for two days.\n"
            "Suggested test questions\n"
            "1. How long has the patient had a headache?\n"
        )

        cases = app.generate_evaluation_cases(document)

        self.assertEqual(len(cases), 1)
        self.assertEqual(
            cases[0]["relevant_evidence"],
            "The patient reports a headache for two days.",
        )

    def test_generated_questions_contain_meaningful_words(self):
        """A question made only of stop words can never retrieve anything."""

        for case in app.generate_evaluation_cases(NOTE):
            with self.subTest(question=case["question"]):
                words = {
                    word for word in app.tokenize(case["question"])
                    if word not in app.STOP_WORDS
                }

                self.assertTrue(
                    words,
                    "generated question contains no meaningful words"
                )


class TestEvaluateCases(unittest.TestCase):
    """The evaluator must score retrieval, not repeat the expected answer."""

    def build_frame(self):
        return pd.DataFrame(
            [
                {"question": question, "relevant_evidence": evidence}
                for question, evidence in FIVE_CASES
            ]
        )

    def test_all_five_cases_pass_with_full_scores(self):
        results = app.evaluate_cases(NOTE, self.build_frame())

        self.assertEqual(len(results), 5)

        passed = int((results["Evidence Match"] == "PASS").sum())
        self.assertEqual(passed, 5)
        self.assertAlmostEqual(results["F1 Score"].mean(), 1.0, places=3)
        self.assertAlmostEqual(results["Precision"].mean(), 1.0, places=3)
        self.assertAlmostEqual(results["Recall"].mean(), 1.0, places=3)

    def test_evidence_is_retrieved_with_the_question_not_the_answer(self):
        calls = []
        original = app.find_relevant_sentences

        def recorder(document, question, top_k=3):
            calls.append(question)
            return original(document, question, top_k=top_k)

        app.find_relevant_sentences = recorder
        try:
            results = app.evaluate_cases(NOTE, self.build_frame())
        finally:
            app.find_relevant_sentences = original

        self.assertEqual(len(calls), 5)

        for question, expected in FIVE_CASES:
            with self.subTest(question=question):
                self.assertIn(question, calls)
                self.assertNotIn(expected, calls)

        # The expected evidence is never used as a retrieval query.
        self.assertFalse(
            any(call in [expected for _, expected in FIVE_CASES]
                for call in calls)
        )

        # And the results still show the correct evidence.
        self.assertEqual(
            results["Retrieved Evidence"].tolist(),
            [expected for _, expected in FIVE_CASES],
        )

    def test_wrong_expected_evidence_is_flagged_not_passed(self):
        frame = pd.DataFrame([
            {
                "question": "How long has the patient had a headache?",
                "relevant_evidence": "The patient has a dry cough.",
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertEqual(results.loc[0, "Evidence Match"], "CHECK")
        self.assertLess(results.loc[0, "F1 Score"], 0.8)

    def test_no_evidence_retrieved_scores_zero(self):
        frame = pd.DataFrame([
            {
                "question": "What is the patient's blood type?",
                "relevant_evidence": "The patient has a dry cough.",
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertEqual(results.loc[0, "Retrieved Evidence"],
                         "No evidence retrieved")
        self.assertEqual(results.loc[0, "Evidence Match"], "CHECK")
        self.assertEqual(results.loc[0, "Precision"], 0.0)
        self.assertEqual(results.loc[0, "Recall"], 0.0)
        self.assertEqual(results.loc[0, "F1 Score"], 0.0)

    def test_rows_without_a_question_are_skipped(self):
        """A row without a question is dropped, a row without expected
        evidence is kept and scored as an unanswerable question."""

        frame = pd.DataFrame([
            {"question": "", "relevant_evidence": "The patient has a dry cough."},
            {"question": "Does the patient have a dry cough?", "relevant_evidence": ""},
            {"question": float("nan"), "relevant_evidence": float("nan")},
            {
                "question": "Does the patient have a dry cough?",
                "relevant_evidence": "The patient has a dry cough.",
            },
        ])

        results = app.evaluate_cases(NOTE, frame)

        # The two rows without a question are skipped.
        self.assertEqual(len(results), 2)

        # The row without expected evidence is scored as unanswerable.
        unanswerable = results[
            results["Expected Evidence"] == "(not answerable from the note)"
        ]
        self.assertEqual(len(unanswerable), 1)

        # The retriever does find evidence for it, so it is not a correct
        # abstention and it must not be marked as a pass.
        self.assertEqual(unanswerable.iloc[0]["Evidence Match"], "CHECK")
        self.assertEqual(unanswerable.iloc[0]["Rank"], 0)
        self.assertEqual(unanswerable.iloc[0]["Hit@3"], 0)
        self.assertEqual(unanswerable.iloc[0]["RR"], 0.0)

        # The complete case still passes and is ranked first.
        self.assertEqual(results.iloc[1]["Evidence Match"], "PASS")
        self.assertEqual(results.iloc[1]["Rank"], 1)
        self.assertEqual(results.iloc[1]["Hit@3"], 1)
        self.assertEqual(results.iloc[1]["RR"], 1.0)

    def test_unanswerable_question_that_abstains_passes(self):
        """An unanswerable question is correct when nothing is retrieved."""

        frame = pd.DataFrame([
            {
                "question": "What is the patient's blood type?",
                "relevant_evidence": "",
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertEqual(results.loc[0, "Evidence Match"], "PASS")
        self.assertEqual(results.loc[0, "Retrieved Evidence"],
                         "No evidence retrieved")
        self.assertEqual(results.loc[0, "Precision"], 1.0)
        self.assertEqual(results.loc[0, "Recall"], 1.0)
        self.assertEqual(results.loc[0, "F1 Score"], 1.0)
        self.assertEqual(results.loc[0, "Rank"], 1)
        self.assertEqual(results.loc[0, "Hit@3"], 1)
        self.assertEqual(results.loc[0, "RR"], 1.0)

    def test_rank_hit_and_reciprocal_rank(self):
        """The expected evidence is looked for in the top 3 results.

        Here the expected sentence is retrieved second, so the verdict is
        CHECK (the top answer is wrong) but the answer was still found
        inside the top 3.
        """

        frame = pd.DataFrame([
            {
                "question": "Does the patient have a cough or a headache?",
                "relevant_evidence": "The patient reports a headache for two days.",
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertEqual(results.loc[0, "Rank"], 2)
        self.assertEqual(results.loc[0, "Hit@3"], 1)
        self.assertEqual(results.loc[0, "RR"], 0.5)

        # The top answer is still the wrong one, so it is not a pass.
        self.assertEqual(results.loc[0, "Evidence Match"], "CHECK")
        self.assertEqual(
            results.loc[0, "Retrieved Evidence"],
            "The patient has a dry cough.",
        )

    def test_rank_is_zero_when_the_evidence_is_not_retrieved(self):
        frame = pd.DataFrame([
            {
                "question": "What is the patient's blood type?",
                "relevant_evidence": "The patient reports a dry cough.",
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertEqual(results.loc[0, "Rank"], 0)
        self.assertEqual(results.loc[0, "Hit@3"], 0)
        self.assertEqual(results.loc[0, "RR"], 0.0)

    def test_alternative_column_names_are_accepted(self):
        frame = pd.DataFrame([
            {
                "Question": "Does the patient have a dry cough?",
                "Expected Evidence": "The patient has a dry cough.",
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertEqual(len(results), 1)
        self.assertEqual(results.loc[0, "Evidence Match"], "PASS")

    def test_broken_csv_columns_return_an_empty_result(self):
        """The old evaluation.csv had one quoted header field."""

        frame = pd.DataFrame([
            {
                "question,relevant_evidence": (
                    'What change in water intake is mentioned?,'
                    '"The patient has been drinking less water than usual."'
                )
            }
        ])

        results = app.evaluate_cases(NOTE, frame)

        self.assertTrue(results.empty)


class TestEvidenceMatching(unittest.TestCase):
    """Matching must tolerate formatting, but not a different meaning."""

    def test_punctuation_case_and_whitespace_are_tolerated(self):
        expected = "The patient reports a headache for two days."
        retrieved = "the   patient REPORTS a headache for two days"

        self.assertTrue(app.evidence_matches(expected, retrieved))

    def test_trailing_clause_is_tolerated(self):
        expected = "The patient reports a headache for two days."
        retrieved = (
            "The patient reports a headache for two days and nausea."
        )

        self.assertTrue(app.evidence_matches(expected, retrieved))

    def test_different_sentence_is_not_a_match(self):
        expected = "The patient reports a headache for two days."
        retrieved = "The patient reports a sore throat for two days."

        self.assertFalse(app.evidence_matches(expected, retrieved))

    def test_different_duration_is_not_a_match(self):
        expected = "The patient reports a headache for two days."
        retrieved = "The patient reports a headache for two weeks."

        self.assertFalse(app.evidence_matches(expected, retrieved))

    def test_polarity_flip_is_not_a_match(self):
        """A denial is not the same evidence as a positive finding."""

        expected = "The patient denies shortness of breath."
        retrieved = "The patient has shortness of breath."

        self.assertFalse(app.evidence_matches(expected, retrieved))

    def test_polarity_flip_the_other_way_is_not_a_match(self):
        expected = "The patient has shortness of breath."
        retrieved = "The patient denies shortness of breath."

        self.assertFalse(app.evidence_matches(expected, retrieved))

    def test_denial_matches_the_same_denial(self):
        expected = "The patient denies shortness of breath."
        retrieved = "The patient denies shortness of breath."

        self.assertTrue(app.evidence_matches(expected, retrieved))

    def test_missing_evidence_is_not_a_match(self):
        self.assertFalse(
            app.evidence_matches("The patient has a dry cough.", "")
        )


class TestMetrics(unittest.TestCase):
    """Precision, recall and F1 must be real numbers, not hard coded."""

    def test_scores_are_computed_from_word_overlap(self):
        precision, recall, f1 = app.overlap_scores(
            "The patient reports a headache for two days.",
            "The patient has a dry cough.",
        )

        # Shared words: the, patient, a.
        self.assertAlmostEqual(precision, 3 / 6, places=3)
        self.assertAlmostEqual(recall, 3 / 8, places=3)
        self.assertAlmostEqual(
            f1,
            2 * (3 / 6) * (3 / 8) / ((3 / 6) + (3 / 8)),
            places=3
        )

    def test_identical_text_scores_one(self):
        precision, recall, f1 = app.overlap_scores(
            "The patient has a dry cough.",
            "The patient has a dry cough.",
        )

        self.assertEqual((precision, recall, f1), (1.0, 1.0, 1.0))

    def test_empty_text_scores_zero(self):
        self.assertEqual(
            app.overlap_scores("The patient has a dry cough.", ""),
            (0.0, 0.0, 0.0)
        )


class TestShippedEvaluationFiles(unittest.TestCase):
    """The CSV files in the project must be usable as-is."""

    def test_shipped_csv_files_have_a_valid_header(self):
        for name in ["evaluation.csv", "evaluation_test.csv", "evaluation"]:
            path = PROJECT_ROOT / name
            with self.subTest(file=name):
                frame = pd.read_csv(path)

                self.assertIn("question", frame.columns)
                self.assertIn("relevant_evidence", frame.columns)
                self.assertEqual(len(frame), 5)

    def test_shipped_csv_files_evaluate_to_five_matches(self):
        for name in ["evaluation.csv", "evaluation_test.csv", "evaluation"]:
            path = PROJECT_ROOT / name
            with self.subTest(file=name):
                frame = pd.read_csv(path)
                results = app.evaluate_cases(NOTE, frame)

                passed = int((results["Evidence Match"] == "PASS").sum())
                self.assertEqual(passed, 5)
                self.assertAlmostEqual(
                    results["F1 Score"].mean(), 1.0, places=3
                )


class TestEvaluationConsole(unittest.TestCase):
    """The dashboard widgets must show the real numbers of the results."""

    def build_results(self, cases):
        frame = pd.DataFrame([
            {"question": question, "relevant_evidence": evidence}
            for question, evidence in cases
        ])

        return app.evaluate_cases(NOTE, frame)

    def test_console_shows_the_real_numbers(self):
        results = self.build_results(FIVE_CASES)

        console = app.build_evaluation_console(results)
        table = app.build_results_table(results)

        # The console carries the aggregate numbers.
        self.assertIn("5/5", console)
        self.assertIn("100.0%", console)
        self.assertIn("1.000", console)

        # The retrieval quality metrics are shown as well.
        self.assertIn('data-metric="hit-at-3"', console)
        self.assertIn('data-metric="mrr"', console)
        self.assertIn("100.0%", console)
        self.assertIn("1.000", console)

        # The table carries the score of every single case.
        self.assertEqual(table.count("1.000"), 15)

        self.assertEqual(console.count('eval-badge pass">PASS'), 5)
        self.assertNotIn("eval-badge check", console)

    def test_console_shows_a_failed_case_as_check(self):
        results = self.build_results([
            ("How long has the patient had a headache?",
             "The patient has a dry cough."),
        ])

        console = app.build_evaluation_console(results)
        table = app.build_results_table(results)

        self.assertIn("0/1", console)
        self.assertIn("0.0%", console)
        self.assertIn('eval-badge check">CHECK', console)
        self.assertIn('rt-pill check">CHECK', table)
        self.assertNotIn("PASS", console)
        self.assertNotIn("PASS", table)

    def test_results_table_has_one_row_per_case(self):
        results = self.build_results(FIVE_CASES)

        table = app.build_results_table(results)

        self.assertEqual(table.count('class="rt-row"'), 5)
        self.assertEqual(table.count('class="rt-pill'), 5)

    def test_console_escapes_question_text(self):
        results = self.build_results([
            ("Does the patient have <script>alert(1)</script>?",
             "The patient has a dry cough."),
        ])

        console = app.build_evaluation_console(results)

        self.assertNotIn("<script>", console)
        self.assertIn("&lt;script&gt;", console)


if __name__ == "__main__":
    unittest.main(verbosity=2)
