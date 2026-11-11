import argparse
import json
import re
from pathlib import Path


def extract_paper_title(file_path: str) -> str:
    """
    Extract the title from a paper file
    Supports title extraction from PDF, TXT, DOCX and other formats
    """
    path = Path(file_path)
    file_extension = path.suffix.lower()

    try:
        if file_extension == '.pdf':
            return _extract_title_from_pdf(file_path)
        elif file_extension == '.txt':
            return _extract_title_from_txt(file_path)
        elif file_extension == '.docx':
            return _extract_title_from_docx(file_path)
        else:
            # For unsupported formats, fall back to the file name as the title
            return path.stem
    except Exception as e:
        print(f"Failed to extract the paper title ({file_path}): {e}")
        return path.stem


def _extract_title_from_txt(file_path: str) -> str:
    """Extract the title from a TXT file"""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Try to match common title formats
        patterns = [
            r'^[A-Z][A-Za-z\s,\-:;]+$',  # first line capitalized
            r'^[0-9]+\.\s+[A-Z][A-Za-z\s,\-:;]+$',  # leading number
            r'^Title:\s*(.+)',  # leading 'Title:'
            r'^第[一二三四五六七八九十\d]+章\s*[：:]\s*(.+)',  # Chinese chapter heading
        ]

        lines = content.split('\n')
        for line in lines[:10]:  # only check the first 10 lines
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    return match.group(1).strip()

        # No title matched, return the first non-empty line
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 5:  # titles are usually longer
                return line

        return Path(file_path).stem

    except Exception as e:
        print(f"Failed to extract the title from the TXT file: {e}")
        return Path(file_path).stem


def _is_likely_not_title(line: str) -> bool:
    """Determine whether the line is obviously not a title"""
    # Purely numeric or a page number
    if re.match(r'^\d+$', line):
        return True
    # Date format
    if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}', line):
        return True
    # Common running header/footer patterns
    if re.match(r'^(page|vol|volume|issue|no\.|number)', line, re.IGNORECASE):
        return True
    # Author information (contains @ or an e-mail address)
    if '@' in line or re.search(r'\.edu|\.com|\.org', line):
        return True
    # Ends with a period (usually the abstract or body text)
    if line.endswith('.') and len(line) > 30:
        return True
    # Reference list entry
    if re.match(r'^\[\d+\]', line):
        return True
    return False


def _title_score(line: str) -> int:
    """Score a line as a title; the higher the score, the more likely it is a title"""
    score = 0

    # Moderate length (10-150 characters)
    if 10 <= len(line) <= 150:
        score += 2
    elif 5 <= len(line) <= 200:
        score += 1

    # Starts with a capital letter
    if line[0].isupper():
        score += 1

    # Does not end with punctuation (except the question mark)
    if line[-1] not in '.!;,:' or line[-1] == '?':
        score += 1

    # Contains common title words
    title_keywords = ['analysis', 'study', 'novel', 'approach', 'method',
                      'design', 'optimization', 'performance', 'evaluation',
                      'optical', 'lens', 'system', 'model', 'paper']
    if any(kw in line.lower() for kw in title_keywords):
        score += 1

    # Does not contain too many digits (titles usually have few)
    digit_ratio = sum(c.isdigit() for c in line) / len(line)
    if digit_ratio < 0.1:
        score += 1

    return score


def _extract_title_from_pdf(file_path: str) -> str:
    """Extract the title from a PDF file"""
    try:
        import PyPDF2
    except ImportError:
        print("Warning: PyPDF2 is not installed, cannot extract the title from the PDF. Please run: pip install PyPDF2")
        return Path(file_path).stem

    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)

            # 1. Try to read the title from the PDF metadata
            if pdf_reader.metadata and pdf_reader.metadata.title:
                title = pdf_reader.metadata.title.strip()
                if title and len(title) > 5:
                    return title

            # 2. Extract it from the text of the first page
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            lines = text.split('\n')
            candidates = []

            for line in lines[:15]:  # widen to the first 15 lines
                line = line.strip()
                if not line:
                    continue

                # Filter out lines that are obviously not titles
                if _is_likely_not_title(line):
                    continue

                # Score the line as a title
                score = _title_score(line)
                if score > 0:
                    candidates.append((line, score))

            # Return the highest-scoring candidate
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]

            return Path(file_path).stem

    except Exception as e:
        print(f"Failed to extract the title from the PDF file: {e}")
        return Path(file_path).stem


def _extract_title_from_docx(file_path: str) -> str:
    """Extract the title from a DOCX file"""
    try:
        from docx import Document
    except ImportError:
        print("Warning: python-docx is not installed, cannot extract the title from the DOCX. Please run: pip install python-docx")
        return Path(file_path).stem

    try:
        doc = Document(file_path)

        # Get the first paragraph
        if doc.paragraphs:
            first_paragraph = doc.paragraphs[0].text.strip()
            if first_paragraph and len(first_paragraph) > 5:
                return first_paragraph

        return Path(file_path).stem

    except Exception as e:
        print(f"Failed to extract the title from the DOCX file: {e}")
        return Path(file_path).stem


def process_papers_directory(input_dir: str, output_file: str) -> None:
    """
    Process a directory of papers and generate a JSON dataset

    Args:
        input_dir: Path to the directory holding the papers
        output_file: Path of the output JSON file
    """
    input_path = Path(input_dir)

    # Check that the input directory exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_dir}")

    papers_data = []

    # Walk every file in the directory
    for file_path in input_path.iterdir():
        if file_path.is_file():
            # Assume every file is a paper
            title = extract_paper_title(str(file_path))
            location = str(file_path.absolute())

            paper_info = {
                "title": title,
                "location": location
            }
            papers_data.append(paper_info)

    # Create the output directory (if it does not exist)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully processed {len(papers_data)} papers; data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert a paper folder into a JSON dataset')
    parser.add_argument('--input_dir', type=str, default="dataset/paper_info_extract/data_v1", help='Directory path holding the papers')
    parser.add_argument('--output_file', type=str, default="dataset/paper_info_extract/dataset_json/dataset_v1.json", help='Path of the output JSON file')

    args = parser.parse_args()

    try:
        process_papers_directory(args.input_dir, args.output_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
