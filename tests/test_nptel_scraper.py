import unittest

from nptel_scraper import (
    Course,
    course_id_from_url,
    extract_description,
    matches_interest,
    normalize_text,
)


class TestNptelScraper(unittest.TestCase):
    def test_course_id_from_preview_url(self):
        self.assertEqual(
            course_id_from_url("https://onlinecourses.nptel.ac.in/noc26_cs81/preview"),
            "noc26_cs81",
        )

    def test_extract_description_stops_at_metadata(self):
        page_text = """
            Welcome to the course
            ABOUT THE COURSE:
            First paragraph about the subject.

            Second paragraph with details.
            INTENDED AUDIENCE: Undergraduate students
            PREREQUISITES: None
        """
        self.assertEqual(
            extract_description(page_text),
            "First paragraph about the subject.\nSecond paragraph with details.",
        )

    def test_extract_description_supports_course_description_heading(self):
        self.assertEqual(
            extract_description(
                "Course Description\nA concise description.\nCOURSE LAYOUT\nWeek 1"
            ),
            "A concise description.",
        )

    def test_extract_description_returns_none_when_heading_is_absent(self):
        self.assertIsNone(extract_description("Welcome\nCourse layout\nWeek 1"))

    def test_normalize_text_deduplicates_adjacent_rendered_lines(self):
        self.assertEqual(
            normalize_text("  A   title \n\nA title\n detail  "), "A title\ndetail"
        )

    def test_interest_matching_is_case_insensitive_and_supports_id(self):
        course = Course(
            title="Reinforcement Learning",
            url="https://onlinecourses.nptel.ac.in/noc26_cs81/preview",
            course_id="noc26_cs81",
            card_text="Computer Science and Engineering",
        )
        self.assertTrue(matches_interest(course, ["reinforcement"]))
        self.assertTrue(matches_interest(course, ["NOC26_CS81"]))
        self.assertTrue(matches_interest(course, ["computer science"]))
        self.assertFalse(matches_interest(course, ["thermodynamics"]))


if __name__ == "__main__":
    unittest.main()
