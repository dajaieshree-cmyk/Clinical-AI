"""End-to-end test of the Streamlit evaluation dashboard.

The test drives the real app with Streamlit's ``AppTest`` harness: it pastes
the demonstration note into the document field, runs the evaluation and then
checks the numbers that the dashboard actually displays.

Run with:

    python -m unittest discover -s tests -v
"""

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import APP_PATH, PROJECT_ROOT, read_note  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

NOTE = read_note()


def console_html(app):
    """The HTML of the evaluation console rendered by the dashboard."""

    blocks = [
        element.value for element in app.markdown
        if element.value and "eval-console" in element.value
    ]

    return "\n".join(blocks)


def kpi_value(html, key):
    """Read one KPI value out of the console HTML."""

    match = re.search(
        r'data-metric="' + re.escape(key) + r'".*?'
        r'eval-kpi-value">([^<]*)<',
        html,
        re.DOTALL
    )

    return match.group(1).strip() if match else None


class TestEvaluationDashboard(unittest.TestCase):
    """The dashboard must report the real result of the evaluation."""

    def setUp(self):
        self.previous_directory = os.getcwd()
        os.chdir(PROJECT_ROOT)

    def tearDown(self):
        os.chdir(self.previous_directory)

    def build_app(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=180)
        app.run()
        return app

    def test_dashboard_reports_five_of_five_matches(self):
        app = self.build_app()

        # 1. Paste the clinical note into the document field.
        note_area = [
            area for area in app.text_area
            if area.label == "Or paste clinical text here"
        ][0]
        note_area.set_value(NOTE)
        app.run()

        # 2. Run the evaluation on the generated test cases.
        run_button = [
            button for button in app.button
            if button.label == "Run evaluation on generated test cases"
        ][0]
        run_button.click()
        app.run()

        # 3. Check what the dashboard displays.
        html = console_html(app)

        self.assertTrue(html, "the evaluation console was not rendered")

        self.assertEqual(kpi_value(html, "test-cases"), "5")
        self.assertEqual(kpi_value(html, "evidence-matches"), "5/5")
        self.assertEqual(kpi_value(html, "match-rate"), "100.0%")
        self.assertEqual(kpi_value(html, "average-f1"), "1.000")

        # Every case is shown as a PASS verdict card.
        self.assertEqual(html.count('eval-badge pass">PASS'), 5)
        self.assertNotIn("eval-badge check", html)

        # 4. No errors were raised while running the app.
        self.assertEqual([error.value for error in app.error], [])

    def test_dashboard_lists_the_five_generated_questions(self):
        app = self.build_app()

        note_area = [
            area for area in app.text_area
            if area.label == "Or paste clinical text here"
        ][0]
        note_area.set_value(NOTE)
        app.run()

        tables = app.dataframe
        self.assertTrue(tables, "the generated cases table is missing")

        # The feedback history table is rendered first, so pick the table
        # that holds the generated evaluation cases.
        generated = [
            table.value for table in tables
            if "question" in table.value.columns
            and "relevant_evidence" in table.value.columns
        ][0]

        self.assertEqual(len(generated), 5)

        questions = list(generated["question"])
        self.assertEqual(questions[0], "How long has the patient had a headache?")
        self.assertEqual(
            questions[1], "How long has the patient had a sore throat?"
        )
        self.assertEqual(questions[2], "Does the patient have a dry cough?")
        self.assertEqual(
            questions[3], "Does the patient have shortness of breath?"
        )
        self.assertEqual(
            questions[4], "What does the patient report drinking?"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
