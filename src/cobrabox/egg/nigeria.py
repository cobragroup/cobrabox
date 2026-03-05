"""Nigeria population easter egg.

Usage
-----
    uv run python -m cobrabox.egg.nigeria

Displays an ASCII map of Africa with Nigeria highlighted,
along with the latest population figure and census date.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

# World Bank API for Nigeria population (indicator: SP.POP.TOTL)
# Fetch 5 records to handle cases where latest year has null value
WORLDBANK_API_URL = (
    "https://api.worldbank.org/v2/country/NGA/indicator/SP.POP.TOTL"
    "?format=json&per_page=5&date=2020:2030&sort=desc"
)

# Fallback population (UN estimate as of 2024)
FALLBACK_POPULATION = 223800000
FALLBACK_YEAR = 2024


def _fetch_worldbank_data() -> tuple[int, int] | None:
    """Fetch Nigeria population from World Bank API.

    Returns
    -------
    tuple[int, int] | None
        (population, year) or None if failed.
    """
    try:
        with urllib.request.urlopen(WORLDBANK_API_URL, timeout=10) as response:
            data: list[Any] = json.loads(response.read().decode("utf-8"))

            # World Bank returns [metadata, [data]]
            if isinstance(data, list) and len(data) > 1:
                records = data[1]
                if records and isinstance(records, list):
                    # Iterate through records to find first non-null value
                    # (latest year may be null if data not yet available)
                    for record in records:
                        value = record.get("value")
                        year_str = record.get("date")
                        if value is not None and year_str is not None:
                            population = int(value)
                            year = int(year_str)
                            if population > 0 and year > 0:
                                return population, year
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass
    return None


def _format_population(pop: int) -> str:
    """Format population with thousands separator."""
    return f"{pop:,}"


def _visible_len(text: str) -> int:
    """Calculate visible length of string (excluding ANSI codes)."""
    return len(re.sub(r"\x1b\[[0-9;]*m", "", text))


def _render_africa_map(population: int, year: int, date_str: str) -> str:
    """Render ASCII map of Africa with Nigeria highlighted."""
    # Bold ANSI codes
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"

    pop_str = _format_population(population)

    # Box elements - all exactly 31 visible chars wide for perfect square
    box_top = f"{RESET}{BOLD}  ╔═════════════════════════════╗{RESET}"
    box_country = f"{RESET}{BOLD}  ║     FEDERAL REPUBLIC OF     ║{RESET}"
    box_name = f"{RESET}{BOLD}  ║         NIGERIA             ║{RESET}"
    box_empty = f"{RESET}{BOLD}  ║                             ║{RESET}"
    box_pop = f"{RESET}{BOLD}  ║  Population: {YELLOW}{pop_str:^13}{RESET}{BOLD} ║{RESET}"
    box_year = f"{RESET}{BOLD}  ║         ({year} estimate)     ║{RESET}"
    box_date = f"{RESET}{BOLD}  ║  Retrieved: {date_str:^15} ║{RESET}"
    box_bot = f"{RESET}{BOLD}  ╚═════════════════════════════╝{RESET}"

    # Africa continent shape with Nigeria highlighted
    # Each box line padded to start at column 48
    target_col = 48

    def pad_line(line: str) -> str:
        """Pad line so box starts at target_col."""
        visible = _visible_len(line)
        padding = target_col - visible
        if padding > 0:
            return line + " " * padding
        return line

    # Build map lines with box attached at consistent position
    map_ln1 = (
        f"{DIM}       /{DIM}              {GREEN}████{DIM}                            {DIM}     =."
    )
    map_ln2 = (
        f"{DIM}      |{DIM}               {GREEN}████{DIM}                            {DIM}      =."
    )
    map_ln3 = (
        f"{DIM}      |{DIM}                {GREEN}██{DIM}                             {DIM}       ="
    )
    map_ln4 = f"{DIM}      |{DIM}                                            {DIM}       .="
    map_ln5 = f"{DIM}       ^{DIM}                                          {DIM}       =="
    map_ln6 = f"{DIM}        ^{DIM}            .==================.{DIM}        .="
    map_ln7 = f"{DIM}         ^.         .={DIM}                {DIM}=.     .="
    map_ln8 = f"{DIM}           ^.     .={DIM}                    {DIM}=. .="
    map_ln9 = f"{DIM}             ^. .={DIM}                      {DIM}=="

    lines = [
        f"{DIM}              .======================================.",
        f"{DIM}            .={DIM}                                    {DIM}   =.",
        f"{DIM}          .={DIM}                                        {DIM}   =.",
        f"{DIM}         /{DIM}                                            {DIM}    =.",
        f"{DIM}        /{DIM}              {GREEN}██{DIM}                             {DIM}    =.",
        pad_line(map_ln1) + box_top,
        pad_line(map_ln2) + box_country,
        pad_line(map_ln3) + box_name,
        pad_line(map_ln4) + box_empty,
        pad_line(map_ln5) + box_pop,
        pad_line(map_ln6) + box_year,
        pad_line(map_ln7) + box_empty,
        pad_line(map_ln8) + box_date,
        pad_line(map_ln9) + box_bot,
        f"{DIM}               ^{DIM}                        {DIM}={DIM}",
        f"{DIM}                '={DIM}                    {DIM}={DIM}",
        f"{DIM}                  '={DIM}        (o)       {DIM}={DIM}    Madagascar",
        f"{DIM}                    '={DIM}             {DIM}={DIM}",
        f"{DIM}                      '==========={DIM}",
        f"{DIM}",
        f"{DIM}                         A F R I C A{RESET}",
        "",
        f"{DIM}     Data source: World Bank Open Data API{RESET}",
    ]

    return "\n".join(lines)


def _render_fallback_map(population: int, year: int, date_str: str) -> str:
    """Render fallback ASCII map when API fails."""
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"

    pop_str = _format_population(population)

    # Box elements - all exactly 31 visible chars wide for perfect square
    box_top = f"{RESET}{BOLD}  ╔═════════════════════════════╗{RESET}"
    box_country = f"{RESET}{BOLD}  ║     FEDERAL REPUBLIC OF     ║{RESET}"
    box_name = f"{RESET}{BOLD}  ║         NIGERIA             ║{RESET}"
    box_empty = f"{RESET}{BOLD}  ║                             ║{RESET}"
    box_pop = f"{RESET}{BOLD}  ║  Population: {YELLOW}{pop_str:^13}{RESET}{BOLD} ║{RESET}"
    box_year = f"{RESET}{BOLD}  ║         ({year} estimate)     ║{RESET}"
    box_date = f"{RESET}{BOLD}  ║  Retrieved: {date_str:^15} ║{RESET}"
    box_bot = f"{RESET}{BOLD}  ╚═════════════════════════════╝{RESET}"

    # Africa continent shape with Nigeria highlighted
    target_col = 48

    def pad_line(line: str) -> str:
        """Pad line so box starts at target_col."""
        visible = _visible_len(line)
        padding = target_col - visible
        if padding > 0:
            return line + " " * padding
        return line

    # Build map lines with box attached at consistent position
    map_ln1 = (
        f"{DIM}       /{DIM}              {GREEN}████{DIM}                            {DIM}     =."
    )
    map_ln2 = (
        f"{DIM}      |{DIM}               {GREEN}████{DIM}                            {DIM}      =."
    )
    map_ln3 = (
        f"{DIM}      |{DIM}                {GREEN}██{DIM}                             {DIM}       ="
    )
    map_ln4 = f"{DIM}      |{DIM}                                            {DIM}       .="
    map_ln5 = f"{DIM}       ^{DIM}                                          {DIM}       =="
    map_ln6 = f"{DIM}        ^{DIM}            .==================.{DIM}        .="
    map_ln7 = f"{DIM}         ^.         .={DIM}                {DIM}=.     .="
    map_ln8 = f"{DIM}           ^.     .={DIM}                    {DIM}=. .="
    map_ln9 = f"{DIM}             ^. .={DIM}                      {DIM}=="

    lines = [
        f"{DIM}              .======================================.",
        f"{DIM}            .={DIM}                                    {DIM}   =.",
        f"{DIM}          .={DIM}                                        {DIM}   =.",
        f"{DIM}         /{DIM}                                            {DIM}    =.",
        f"{DIM}        /{DIM}              {GREEN}██{DIM}                             {DIM}    =.",
        pad_line(map_ln1) + box_top,
        pad_line(map_ln2) + box_country,
        pad_line(map_ln3) + box_name,
        pad_line(map_ln4) + box_empty,
        pad_line(map_ln5) + box_pop,
        pad_line(map_ln6) + box_year,
        pad_line(map_ln7) + box_empty,
        pad_line(map_ln8) + box_date,
        pad_line(map_ln9) + box_bot,
        f"{DIM}               ^{DIM}                        {DIM}={DIM}",
        f"{DIM}                '={DIM}                    {DIM}={DIM}",
        f"{DIM}                  '={DIM}        (o)       {DIM}={DIM}    Madagascar",
        f"{DIM}                    '={DIM}             {DIM}={DIM}",
        f"{DIM}                      '==========={DIM}",
        f"{DIM}",
        f"{DIM}                         A F R I C A{RESET}",
        "",
        f"{DIM}     Data source: UN World Population Prospects (fallback){RESET}",
    ]

    return "\n".join(lines)


def nigeria() -> None:
    """Display Nigeria population with ASCII map."""
    print("\033[?25l", end="")  # Hide cursor

    try:
        # Try to fetch live data
        result = _fetch_worldbank_data()

        if result:
            population, year = result
            date_str = datetime.now().strftime("%Y-%m-%d")
            print(_render_africa_map(population, year, date_str))
            print("\n  ✓ Live data from World Bank")
        else:
            # Use fallback
            date_str = datetime.now().strftime("%Y-%m-%d")
            print(_render_fallback_map(FALLBACK_POPULATION, FALLBACK_YEAR, date_str))
            print("\n  ⚠ Could not fetch live data, showing UN estimate", file=sys.stderr)

    finally:
        print("\033[?25h", end="")  # Show cursor
        print("\033[0m")  # Reset colors


def main(argv: list[str] | None = None) -> None:
    """Entry point for command-line usage."""
    nigeria()


if __name__ == "__main__":
    main()
