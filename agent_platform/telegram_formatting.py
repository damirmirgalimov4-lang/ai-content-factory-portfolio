from __future__ import annotations

import html
import re


def normalize_telegram_markdown(text: str) -> str:
    """Normalize model output without changing its meaning or inventing structure."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = [line.rstrip() for line in normalized.splitlines()]
    output: list[str] = []
    blank = False
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if output and output[-1] != "" and not in_code:
                output.append("")
            output.append(stripped)
            in_code = not in_code
            blank = False
            continue
        if in_code:
            output.append(line)
            blank = False
            continue
        if not stripped:
            if output and not blank:
                output.append("")
            blank = True
            continue

        is_section = bool(
            re.match(
                r"^(?:#{1,6}\s+\S|"
                r"(?:Кадр|Сцена|Шаг|Этап|Промпт|Идея|Вариант)\s+\d+\b)",
                stripped,
                re.IGNORECASE,
            )
        )
        if is_section and output and output[-1] != "":
            output.append("")
        output.append(stripped if stripped.startswith(("#", ">", "-", "*", "+")) else line.strip())
        blank = False

    while output and output[-1] == "":
        output.pop()
    return "\n".join(output).strip()


def markdown_to_telegram_html(text: str) -> str:
    """Render a safe Markdown subset supported by Telegram HTML messages."""

    lines = text.strip().splitlines()
    output: list[str] = []
    index = 0
    in_code = False
    code_lines: list[str] = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                output.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if _looks_like_table(lines, index):
            table_lines: list[str] = []
            while index < len(lines) and "|" in lines[index]:
                candidate = lines[index].strip()
                if candidate and not re.fullmatch(r"[|:\- ]+", candidate):
                    table_lines.append(candidate)
                index += 1
            output.append(f"<pre>{html.escape(chr(10).join(table_lines))}</pre>")
            continue

        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            output.append(f"<b>{_inline(heading.group(1))}</b>")
        elif re.fullmatch(r"[-_*]{3,}", stripped):
            output.append("------------")
        elif stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            output.append(f"<blockquote>{_inline(chr(10).join(quote_lines))}</blockquote>")
            continue
        else:
            bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
            if bullet:
                output.append(f"- {_inline(bullet.group(1))}")
            elif stripped:
                output.append(_inline(stripped))
            else:
                output.append("")
        index += 1

    if in_code:
        output.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
    return "\n".join(output).strip()


def _inline(value: str) -> str:
    parts = re.split(r"(`[^`\n]+`)", value)
    rendered: list[str] = []
    for part in parts:
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            rendered.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        escaped = html.escape(part)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", escaped)
        rendered.append(escaped)
    return "".join(rendered)


def _looks_like_table(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    separator = lines[index + 1].strip()
    return bool(re.fullmatch(r"[|:\- ]+", separator) and "-" in separator)
