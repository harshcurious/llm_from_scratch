# llm_from_scratch

Code from *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Fetching the NPTEL course catalog

`nptel_scraper.py` opens the JavaScript-rendered NPTEL e-learning catalog, clicks **Load more** until every course card is visible, and writes the unique courses to JSON or CSV. It can then visit only the courses you are interested in and add their descriptions.

Install the project and Playwright's Chromium browser once:

```bash
uv sync
uv run playwright install chromium
```

Fetch the complete catalog:

```bash
uv run python nptel_scraper.py --output nptel_courses.json
```

Fetch descriptions for selected courses. `--description-for` is a case-insensitive substring matched against the course title, course ID, URL, and card text, and may be repeated:

```bash
uv run python nptel_scraper.py \
  --description-for "reinforcement learning" \
  --description-for "noc26_cs81" \
  --output nptel_courses.json
```

For a longer list, create a UTF-8 text file with one title or course-ID substring per line (blank lines and lines beginning with `#` are ignored):

```bash
uv run python nptel_scraper.py \
  --description-file interested_courses.txt \
  --output nptel_courses.csv
```

Use `--all-descriptions` to visit every course preview page. Other useful options include `--headed` for browser debugging, `--timeout 60` for slow connections, and `--max-load-more-clicks` if the catalog grows beyond the default safety limit. Run `uv run python nptel_scraper.py --help` for the full CLI reference.
