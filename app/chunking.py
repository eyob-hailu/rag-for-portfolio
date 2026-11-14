import os
import re
from typing import List, Dict


def parse_markdown_sections(file_path: str) -> List[Dict[str, str]]:
    """
    Parses a markdown document into distinct semantic sections based on # and ## headers.
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks: List[Dict[str, str]] = []
    current_cat = "GENERAL"
    current_title = "GENERAL"
    current_lines: List[str] = []

    for line in lines:
        raw = line.strip()
        if raw.startswith("# "):
            if current_lines:
                content = "".join(current_lines).strip()
                if len(content) > 50:
                    chunks.append({
                        "category": current_cat,
                        "title": current_title,
                        "text": content
                    })
                current_lines = []
            current_cat = raw.lstrip("# ").strip()
            current_title = current_cat
        elif raw.startswith("## "):
            if current_lines:
                content = "".join(current_lines).strip()
                if len(content) > 50:
                    chunks.append({
                        "category": current_cat,
                        "title": current_title,
                        "text": content
                    })
                current_lines = []
            current_title = raw.lstrip("## ").strip()
        current_lines.append(line)

    if current_lines:
        content = "".join(current_lines).strip()
        if len(content) > 50:
            chunks.append({
                "category": current_cat,
                "title": current_title,
                "text": content
            })

    return chunks

