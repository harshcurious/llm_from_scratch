"""Scrape the NPTEL e-learning catalog and selected course descriptions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlparse

CATALOG_URL = "https://onlinecourses.nptel.ac.in/e-learning"
PREVIEW_PATH_RE = re.compile(r"/(?:noc[^/]+|[^/]+)/preview/?$", re.IGNORECASE)
DESCRIPTION_START_RE = re.compile(
    r"(?:ABOUT\s+THE\s+COURSE\s*:?)|(?:COURSE\s+DESCRIPTION\s*:?)", re.IGNORECASE
)
DESCRIPTION_END_RE = re.compile(
    r"\n\s*(?:INTENDED\s+AUDIENCE|PREREQUISITES?|INDUSTRY\s+SUPPORT|SUMMARY|COURSE\s+LAYOUT)\s*:?[ \t]*\n?",
    re.IGNORECASE,
)


@dataclass
class Course:
    """A course discovered in the NPTEL catalog."""

    title: str
    url: str
    course_id: str
    card_text: str = ""
    description: str | None = None


def normalize_text(value: str) -> str:
    """Collapse browser-rendered whitespace while preserving paragraph breaks."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    paragraphs: list[str] = []
    for line in lines:
        if line and (not paragraphs or paragraphs[-1] != line):
            paragraphs.append(line)
    return "\n".join(paragraphs)


def course_id_from_url(url: str) -> str:
    """Return the path segment immediately before ``preview``."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[-2] if len(parts) >= 2 and parts[-1].lower() == "preview" else ""


def extract_description(page_text: str) -> str | None:
    """Extract the ABOUT THE COURSE section from rendered preview-page text."""
    text = normalize_text(page_text)
    start = DESCRIPTION_START_RE.search(text)
    if not start:
        return None
    remainder = text[start.end() :].lstrip(" :\n")
    end = DESCRIPTION_END_RE.search(remainder)
    description = remainder[: end.start()] if end else remainder
    description = description.strip()
    return description or None


def matches_interest(course: Course, queries: Sequence[str]) -> bool:
    """Match a course by ID, title substring, URL substring, or card-text substring."""
    haystack = "\n".join(
        (course.course_id, course.title, course.url, course.card_text)
    ).casefold()
    return any(
        query.strip().casefold() in haystack for query in queries if query.strip()
    )


def _course_from_browser_record(record: dict[str, str]) -> Course | None:
    url = urljoin(CATALOG_URL, record.get("url", ""))
    if not PREVIEW_PATH_RE.search(urlparse(url).path):
        return None
    title = normalize_text(record.get("title", ""))
    card_text = normalize_text(record.get("card_text", ""))
    if not title:
        title = card_text.splitlines()[0] if card_text else course_id_from_url(url)
    return Course(
        title=title,
        url=url,
        course_id=course_id_from_url(url),
        card_text=card_text,
    )


def scrape_catalog(
    page, *, max_load_more_clicks: int = 500, timeout_ms: int = 30_000
) -> list[Course]:
    """Load the catalog, exhaust its Load more button, and return unique courses."""
    page.goto(CATALOG_URL, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_selector('a[href*="/preview"]', timeout=timeout_ms)

    preview_links = page.locator('a[href*="/preview"]')
    for _ in range(max_load_more_clicks):
        old_count = preview_links.count()
        load_more = page.locator("button, a, [role=button]").filter(
            has_text=re.compile(r"^\s*load\s+more(?:\.\.\.)?\s*$", re.IGNORECASE)
        )
        visible_button = next(
            (
                load_more.nth(index)
                for index in range(load_more.count())
                if load_more.nth(index).is_visible()
            ),
            None,
        )
        if visible_button is None:
            break
        visible_button.scroll_into_view_if_needed()
        visible_button.click()
        try:
            page.wait_for_function(
                "oldCount => document.querySelectorAll('a[href*=\"/preview\"]').length > oldCount",
                arg=old_count,
                timeout=timeout_ms,
            )
        except Exception:
            # A final click commonly removes/disables the button without adding cards.
            if not visible_button.is_visible() or not visible_button.is_enabled():
                break
            raise RuntimeError(
                "The Load more button did not add courses. The NPTEL page structure may have changed."
            )
    else:
        raise RuntimeError(
            f"Stopped after {max_load_more_clicks} Load more clicks; increase the limit if needed."
        )

    records = preview_links.evaluate_all(
        """
        anchors => anchors.map(anchor => {
          let node = anchor;
          let card = anchor;
          while (node.parentElement) {
            node = node.parentElement;
            const links = node.querySelectorAll('a[href*="/preview"]').length;
            const text = (node.innerText || '').trim();
            if (links === 1 && text.length <= 2000) card = node;
            if (links > 1 || text.length > 2000) break;
          }
          const heading = card.querySelector('h1, h2, h3, h4, h5, h6, [class*="title" i]');
          const image = anchor.querySelector('img') || card.querySelector('img');
          return {
            url: anchor.href,
            title: (heading && heading.innerText) || anchor.innerText || (image && image.alt) || anchor.getAttribute('aria-label') || '',
            card_text: card.innerText || ''
          };
        })
        """
    )

    courses: dict[str, Course] = {}
    for record in records:
        course = _course_from_browser_record(record)
        if course is not None:
            courses.setdefault(course.url, course)
    return list(courses.values())


def fetch_descriptions(
    page, courses: Iterable[Course], *, timeout_ms: int = 30_000
) -> None:
    """Visit each supplied course and update it with its rendered description."""
    for course in courses:
        page.goto(course.url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_selector(
                "text=/ABOUT\\s+THE\\s+COURSE|COURSE\\s+DESCRIPTION/i",
                timeout=timeout_ms,
            )
        except Exception:
            # Some archived preview templates render the text without a dedicated element.
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        course.description = extract_description(page.locator("body").inner_text())


def write_courses(courses: Sequence[Course], output: Path, output_format: str) -> None:
    """Write course records as UTF-8 JSON or CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(course) for course in courses]
    if output_format == "json":
        output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Course.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def _interest_queries(arguments: argparse.Namespace) -> list[str]:
    queries = list(arguments.description_for)
    if arguments.description_file:
        queries.extend(
            line.strip()
            for line in arguments.description_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("nptel_courses.json"))
    parser.add_argument("--format", choices=("json", "csv"), dest="output_format")
    parser.add_argument(
        "--description-for",
        action="append",
        default=[],
        metavar="TEXT",
        help="fetch descriptions for courses whose title, ID, URL, or card text contains TEXT; repeatable",
    )
    parser.add_argument(
        "--description-file",
        type=Path,
        help="UTF-8 file containing one description-selection substring per line",
    )
    parser.add_argument("--all-descriptions", action="store_true")
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="browser timeout in seconds"
    )
    parser.add_argument("--max-load-more-clicks", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_format = arguments.output_format or (
        "csv" if arguments.output.suffix.lower() == ".csv" else "json"
    )
    queries = _interest_queries(arguments)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is required. Run: uv sync && uv run playwright install chromium",
            file=sys.stderr,
        )
        return 2

    timeout_ms = int(arguments.timeout * 1000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not arguments.headed)
        page = browser.new_page()
        courses = scrape_catalog(
            page,
            max_load_more_clicks=arguments.max_load_more_clicks,
            timeout_ms=timeout_ms,
        )
        selected = (
            courses
            if arguments.all_descriptions
            else [course for course in courses if matches_interest(course, queries)]
        )
        if selected:
            fetch_descriptions(page, selected, timeout_ms=timeout_ms)
        browser.close()

    write_courses(courses, arguments.output, output_format)
    print(f"Wrote {len(courses)} courses to {arguments.output}")
    if queries or arguments.all_descriptions:
        print(f"Fetched descriptions for {len(selected)} courses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
