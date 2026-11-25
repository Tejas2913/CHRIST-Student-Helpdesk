"""
Dedicated, high-accuracy parser for the CHRIST University Academic Calendar 2026-27.
Extracts month-by-month events, holidays, examinations, and commencement schedules
directly from the official student handbook (Pages 23-32).
"""

import re
from typing import List, Dict, Any, Tuple
import pdfplumber

MONTH_NAMES = [
    "May 2026", "June 2026", "July 2026", "August 2026", "September 2026",
    "October 2026", "November 2026", "December 2026", "January 2027",
    "February 2027", "March 2027"
]

def extract_calendar_records(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses pages 23-32 of the student handbook.
    Returns:
        (monthly_chunks, calendar_structured_records)
    """
    entries_by_month: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MONTH_NAMES}
    holidays_by_month: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MONTH_NAMES}

    with pdfplumber.open(pdf_path) as pdf:
        # Pages 23 to 32 (1-based: 23-32, 0-based: 22-31)
        current_month = "May 2026"

        for page_idx in range(22, 32):
            page_num = page_idx + 1
            page = pdf.pages[page_idx]
            lines = page.extract_text_lines(layout=False)

            # Sort lines vertically
            lines = sorted(lines, key=lambda l: l['top'])

            for line_obj in lines:
                text = line_obj['text'].strip()
                if not text:
                    continue
                if "Handbook 2026-27" in text or "Academic Calendar 2026-27" in text or "CHRIST (Deemed" in text:
                    continue

                # Check if this line defines a new month
                matched_month = None
                for m in MONTH_NAMES:
                    # e.g., "May 2026", "October 2026"
                    if re.match(rf"^{m}\b", text, re.IGNORECASE) or text.lower() == m.lower():
                        matched_month = m
                        break

                if matched_month:
                    current_month = matched_month
                    continue

                # Special handling for transitions on boundaries
                # Page 30 top has Kanakadasa Jayanthi and Compensatory holiday before "December 2026"
                # Those are November 2026 holidays!
                effective_month = current_month
                if page_num == 30 and line_obj['top'] < 110:
                    effective_month = "November 2026"
                elif page_num == 28 and line_obj['top'] < 140:
                    effective_month = "September 2026"

                # Check if it's a holiday or event
                # Format is typically: "DD (Day) Holiday - Name" or "(DD-DD) (Day-Day) Event" or "DD (Day) Event"
                is_holiday = bool(re.search(r"\bHoliday\b", text, re.IGNORECASE))
                
                # Check for Founder's Day
                if "Founder's Day" in text or "Chavara Day" in text:
                    is_holiday = True

                record = {
                    "text": text,
                    "page": page_num,
                    "month": effective_month,
                    "is_holiday": is_holiday,
                    "y": line_obj['top']
                }

                if is_holiday:
                    holidays_by_month[effective_month].append(record)
                entries_by_month[effective_month].append(record)

    chunks: List[Dict[str, Any]] = []
    structured_records: List[Dict[str, Any]] = []

    # Map of known page ranges for months
    month_pages = {
        "May 2026": [23],
        "June 2026": [23, 24, 25, 26],
        "July 2026": [26, 27],
        "August 2026": [27],
        "September 2026": [28],
        "October 2026": [28, 29],
        "November 2026": [29, 30],
        "December 2026": [30],
        "January 2027": [31],
        "February 2027": [31, 32],
        "March 2027": [32]
    }

    for month in MONTH_NAMES:
        entries = entries_by_month[month]
        holidays = holidays_by_month[month]
        pages = month_pages.get(month, [23])
        main_page = pages[0]

        # 1. Build a Comprehensive Monthly Calendar Chunk
        lines_text = []
        lines_text.append(f"Section: 6. Academic Calendar 2026-27")
        lines_text.append(f"Month: {month}")
        lines_text.append(f"Relevant Handbook Pages: {', '.join(str(p) for p in pages)}")
        lines_text.append("")

        if holidays:
            lines_text.append(f"Holidays observed in {month}:")
            for h in holidays:
                lines_text.append(f"- {h['text']} [Handbook p. {h['page']}]")
        else:
            lines_text.append(f"Holidays observed in {month}: None listed in the Academic Calendar.")

        lines_text.append("")
        lines_text.append(f"Key Academic Events and Schedules in {month}:")
        for e in entries:
            if not e['is_holiday']:
                lines_text.append(f"- {e['text']}")

        full_month_text = "\n".join(lines_text)

        chunks.append({
            "chunk_id": f"academic_calendar_{month.lower().replace(' ', '_')}",
            "id": f"academic_calendar_{month.lower().replace(' ', '_')}",
            "document": "CHRIST Student Handbook 2026-27",
            "document_version": "2026-27",
            "academic_year": "2026-27",
            "page": main_page,
            "campus": "All Campuses",
            "programme": "All Programmes",
            "section": "6. Academic Calendar 2026-27",
            "subsection": f"Academic Calendar - {month}",
            "clause": "",
            "content_type": "academic_calendar",
            "table_name": f"Academic Calendar {month}",
            "text": full_month_text,
            "search_text": full_month_text + f"\nKeywords: {month} holidays calendar schedule dates Christ University"
        })

        # 2. Build a Dedicated Holiday Chunk for high-precision retrieval
        if holidays:
            hol_lines = [
                f"CHRIST University Academic Calendar 2026-27 - Official Holidays for {month}:",
                f"The following university and public holidays are scheduled in {month}:"
            ]
            for h in holidays:
                hol_lines.append(f"- {h['text']} [Handbook p. {h['page']}]")
            hol_lines.append(f"Total holidays in {month}: {len(holidays)}")
            hol_chunk_text = "\n".join(hol_lines)

            chunks.append({
                "chunk_id": f"holidays_{month.lower().replace(' ', '_')}",
                "id": f"holidays_{month.lower().replace(' ', '_')}",
                "document": "CHRIST Student Handbook 2026-27",
                "document_version": "2026-27",
                "academic_year": "2026-27",
                "page": main_page,
                "campus": "All Campuses",
                "programme": "All Programmes",
                "section": "6. Academic Calendar 2026-27",
                "subsection": f"Official Holidays - {month}",
                "clause": "",
                "content_type": "academic_calendar",
                "table_name": f"Holidays {month}",
                "text": hol_chunk_text,
                "search_text": hol_chunk_text + f"\nKeywords: holidays in {month} academic calendar {month} public holidays off days university closed"
            })

            # Also create structured records for structured search
            for h in holidays:
                structured_records.append({
                    "table_name": f"Academic Calendar Holidays - {month}",
                    "page": h['page'],
                    "section": "6. Academic Calendar 2026-27",
                    "subsection": f"Holidays {month}",
                    "record": {
                        "month": month,
                        "holiday_entry": h['text'],
                        "holiday": True,
                        "academic_year": "2026-27"
                    },
                    "search_text": f"Holiday in {month}: {h['text']} (Handbook page {h['page']})"
                })

    return chunks, structured_records

if __name__ == "__main__":
    import sys
    pdf_path = r"c:\Users\hp\Desktop\chatbot\christ-student-helpdesk\backend\data\student_handbook.pdf"
    chunks, records = extract_calendar_records(pdf_path)
    print(f"Generated {len(chunks)} calendar chunks and {len(records)} structured holiday records.")
    for c in chunks:
        if "holidays_" in c['chunk_id']:
            print("==================================================")
            print(c['text'])
