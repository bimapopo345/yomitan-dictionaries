#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BLOCK_TRANSLATION_CLASS = "yomitan-id-translation"
INLINE_TRANSLATION_CLASS = "yomitan-id-inline"
TRANSLATE_NODE_MARKERS = {
    "xref-glossary",
    "sense-note-content",
    "info-gloss-content",
    "antonym-glossary",
}
INLINE_LABEL_TRANSLATIONS = {
    "See also": "Lihat juga",
    "Antonym": "Antonim",
    "Explanation": "Penjelasan",
    "Literally": "Secara harfiah",
    "Figuratively": "Secara kiasan",
    "Note": "Catatan",
    "Language of Origin": "Bahasa asal",
}
SKIP_TEXT_MARKERS = {"attribution-footnote"}
ASCII_RE = re.compile(r"[A-Za-z]")


def log(message: str) -> None:
    print(message, flush=True)


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


class ProviderRateLimited(RuntimeError):
    pass


class TranslationCache:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                source TEXT PRIMARY KEY,
                target TEXT NOT NULL
            )
            """
        )

    def get_many(self, texts: list[str]) -> dict[str, str]:
        if not texts:
            return {}

        found: dict[str, str] = {}
        for batch in chunked(texts, 500):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT source, target FROM translations WHERE source IN ({placeholders})",
                batch,
            )
            found.update(rows.fetchall())
        return found

    def store_many(self, mapping: dict[str, str]) -> None:
        if not mapping:
            return
        self.connection.executemany(
            "INSERT OR REPLACE INTO translations(source, target) VALUES (?, ?)",
            mapping.items(),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class GoogleTranslateClient:
    def __init__(self, source_lang: str = "en", target_lang: str = "id") -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.google_cooldown_until = 0.0
        self.logged_google_cooldown = False
        self.local_backend = self._load_local_backend()

    def translate_many(self, texts: list[str]) -> dict[str, str]:
        if self.local_backend is not None:
            return self._translate_batch_local(texts)

        translated: dict[str, str] = {}
        for batch in self._pack_batches(texts):
            translated.update(self._translate_batch_with_fallback(batch))
            time.sleep(0.05)
        return translated

    def _pack_batches(
        self,
        texts: list[str],
        max_items: int = 100,
        max_chars: int = 8000,
    ) -> Iterable[list[str]]:
        batch: list[str] = []
        char_count = 0

        for text in texts:
            extra = len(text) + (1 if batch else 0)
            if batch and (len(batch) >= max_items or char_count + extra > max_chars):
                yield batch
                batch = []
                char_count = 0
            batch.append(text)
            char_count += extra

        if batch:
            yield batch

    def _load_local_backend(self) -> dict | None:
        try:
            import argostranslate.translate
            import ctranslate2
        except ImportError:
            return None

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next(
            (language for language in installed_languages if language.code == self.source_lang),
            None,
        )
        to_lang = next(
            (language for language in installed_languages if language.code == self.target_lang),
            None,
        )
        if from_lang is None or to_lang is None:
            return None

        translation = from_lang.get_translation(to_lang)
        underlying = getattr(translation, "underlying", translation)
        pkg = getattr(underlying, "pkg", None)
        if pkg is None:
            return None

        translator = ctranslate2.Translator(
            str(pkg.package_path / "model"),
            device="cpu",
            inter_threads=min(8, os.cpu_count() or 4),
            intra_threads=0,
        )
        log("Using local Argos Translate model")
        return {"pkg": pkg, "translator": translator}

    def _translate_batch_local(self, texts: list[str]) -> dict[str, str]:
        if not texts:
            return {}

        pkg = self.local_backend["pkg"]
        translator = self.local_backend["translator"]
        tokenized = [pkg.tokenizer.encode(text) for text in texts]
        target_prefix = None
        if pkg.target_prefix != "":
            target_prefix = [[pkg.target_prefix]] * len(tokenized)

        translated_batches = translator.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            replace_unknowns=True,
            max_batch_size=4096,
            batch_type="tokens",
            beam_size=1,
            num_hypotheses=1,
            return_scores=False,
        )

        outputs: list[str] = []
        for translated_batch in translated_batches:
            value = pkg.tokenizer.decode(translated_batch.hypotheses[0])
            if pkg.target_prefix != "" and value.startswith(pkg.target_prefix):
                value = value[len(pkg.target_prefix) :]
            if len(value) > 0 and value[0] == " ":
                value = value[1:]
            outputs.append(value)

        return dict(zip(texts, outputs, strict=True))

    def _translate_batch_with_fallback(self, texts: list[str]) -> dict[str, str]:
        if time.time() >= self.google_cooldown_until:
            try:
                self.logged_google_cooldown = False
                return self._translate_batch_google(texts)
            except ProviderRateLimited:
                self.google_cooldown_until = time.time() + (15 * 60)
                if not self.logged_google_cooldown:
                    log("Google Translate is rate-limited; falling back to MyMemory for 15 minutes")
                    self.logged_google_cooldown = True

        return self._translate_batch_mymemory(texts)

    def _translate_batch_google(self, texts: list[str]) -> dict[str, str]:
        if not texts:
            return {}
        if len(texts) == 1:
            return {texts[0]: self._translate_single_google(texts[0])}

        body = urlencode(
            {
                "client": "gtx",
                "sl": self.source_lang,
                "tl": self.target_lang,
                "dt": "t",
                "q": "\n".join(texts),
            }
        ).encode()
        request = Request(
            "https://translate.googleapis.com/translate_a/single",
            data=body,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        for attempt in range(5):
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                translated = "".join(part[0] for part in payload[0]).split("\n")
                if len(translated) == len(texts):
                    return dict(zip(texts, translated, strict=True))
                break
            except HTTPError as error:
                if error.code == 429:
                    raise ProviderRateLimited("Google Translate returned 429") from error
                if len(texts) == 1:
                    raise RuntimeError(f"Translation failed for {texts[0]!r}: {error}") from error
                time.sleep(1.5 * (attempt + 1))
            except (URLError, TimeoutError) as error:
                if len(texts) == 1:
                    raise RuntimeError(f"Translation failed for {texts[0]!r}: {error}") from error
                time.sleep(1.5 * (attempt + 1))

        midpoint = len(texts) // 2
        if midpoint == 0:
            return {texts[0]: self._translate_single_google(texts[0])}
        left = self._translate_batch_google(texts[:midpoint])
        right = self._translate_batch_google(texts[midpoint:])
        return left | right

    def _translate_single_google(self, text: str) -> str:
        body = urlencode(
            {
                "client": "gtx",
                "sl": self.source_lang,
                "tl": self.target_lang,
                "dt": "t",
                "q": text,
            }
        ).encode()
        request = Request(
            "https://translate.googleapis.com/translate_a/single",
            data=body,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        for attempt in range(5):
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return "".join(part[0] for part in payload[0]).strip()
            except HTTPError as error:
                if error.code == 429:
                    raise ProviderRateLimited("Google Translate returned 429") from error
                if attempt == 4:
                    raise RuntimeError(f"Translation failed for {text!r}: {error}") from error
                time.sleep(1.5 * (attempt + 1))
            except (URLError, TimeoutError) as error:
                if attempt == 4:
                    raise RuntimeError(f"Translation failed for {text!r}: {error}") from error
                time.sleep(1.5 * (attempt + 1))
        return text

    def _translate_batch_mymemory(self, texts: list[str]) -> dict[str, str]:
        if not texts:
            return {}
        if len(texts) == 1:
            return {texts[0]: self._translate_single_mymemory(texts[0])}

        body = urlencode(
            {
                "q": "\n".join(texts),
                "langpair": f"{self.source_lang}|{self.target_lang}",
            }
        ).encode()
        request = Request(
            "https://api.mymemory.translated.net/get",
            data=body,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        for attempt in range(5):
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                translated = str(payload["responseData"]["translatedText"]).split("\n")
                if len(translated) == len(texts):
                    return dict(zip(texts, translated, strict=True))
                break
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as error:
                if len(texts) == 1:
                    raise RuntimeError(f"MyMemory translation failed for {texts[0]!r}: {error}") from error
                time.sleep(1.5 * (attempt + 1))

        midpoint = len(texts) // 2
        if midpoint == 0:
            return {texts[0]: self._translate_single_mymemory(texts[0])}
        left = self._translate_batch_mymemory(texts[:midpoint])
        right = self._translate_batch_mymemory(texts[midpoint:])
        return left | right

    def _translate_single_mymemory(self, text: str) -> str:
        body = urlencode(
            {
                "q": text,
                "langpair": f"{self.source_lang}|{self.target_lang}",
            }
        ).encode()
        request = Request(
            "https://api.mymemory.translated.net/get",
            data=body,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        for attempt in range(5):
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["responseData"]["translatedText"]).strip()
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as error:
                if attempt == 4:
                    raise RuntimeError(f"MyMemory translation failed for {text!r}: {error}") from error
                time.sleep(1.5 * (attempt + 1))
        return text


def get_marker(node: dict) -> str | None:
    data = node.get("data")
    if isinstance(data, dict):
        marker = data.get("content")
        if isinstance(marker, str):
            return marker
    return None


def has_translation_marker(content: object, class_name: str) -> bool:
    if isinstance(content, list):
        return any(has_translation_marker(item, class_name) for item in content)
    if isinstance(content, dict):
        data = content.get("data")
        if isinstance(data, dict) and data.get("class") == class_name:
            return True
        return any(
            has_translation_marker(value, class_name)
            for key, value in content.items()
            if key != "data"
        )
    return False


def extract_plain_text(content: object, english_only: bool) -> str:
    pieces: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            pieces.append(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        marker = get_marker(node)
        if marker in SKIP_TEXT_MARKERS:
            return

        data = node.get("data")
        if isinstance(data, dict) and data.get("class") in {
            BLOCK_TRANSLATION_CLASS,
            INLINE_TRANSLATION_CLASS,
        }:
            return

        lang = node.get("lang")
        if english_only and lang not in (None, "en"):
            return
        if not english_only and lang not in (None, "en"):
            return

        walk(node.get("content"))

    walk(content)
    return "".join(pieces).strip()


def should_translate(text: str) -> bool:
    return bool(text and ASCII_RE.search(text))


def append_block_translation(content: object, translation: str) -> list[object]:
    if isinstance(content, list):
        base = list(content)
    else:
        base = [content]
    base.extend(
        [
            {"tag": "br"},
            {
                "tag": "span",
                "lang": "id",
                "data": {"class": BLOCK_TRANSLATION_CLASS},
                "content": translation,
            },
        ]
    )
    return base


def append_inline_translation(content: object, translation: str) -> list[object]:
    if isinstance(content, list):
        base = list(content)
    else:
        base = [content]
    base.extend(
        [
            " / ",
            {
                "tag": "span",
                "lang": "id",
                "data": {"class": INLINE_TRANSLATION_CLASS},
                "content": translation,
            },
        ]
    )
    return base


def collect_texts(node: object, parent_marker: str | None = None) -> set[str]:
    texts: set[str] = set()

    def walk(current: object, current_parent: str | None) -> None:
        if not isinstance(current, dict):
            if isinstance(current, list):
                for item in current:
                    walk(item, current_parent)
            return

        marker = get_marker(current)
        content = current.get("content")

        if current_parent == "glossary" and current.get("tag") == "li":
            text = extract_plain_text(content, english_only=False)
            if should_translate(text):
                texts.add(text)
        elif marker in TRANSLATE_NODE_MARKERS:
            text = extract_plain_text(content, english_only=False)
            if should_translate(text):
                texts.add(text)
        elif marker == "example-sentence-b":
            text = extract_plain_text(content, english_only=True)
            if should_translate(text):
                texts.add(text)

        next_parent = marker or current_parent
        if isinstance(content, list):
            for item in content:
                walk(item, next_parent)
        elif isinstance(content, dict):
            walk(content, next_parent)

    walk(node, parent_marker)
    return texts


def apply_translations(
    node: object,
    translations: dict[str, str],
    parent_marker: str | None = None,
) -> None:
    if not isinstance(node, dict):
        if isinstance(node, list):
            for item in node:
                apply_translations(item, translations, parent_marker)
        return

    marker = get_marker(node)
    content = node.get("content")

    if (
        parent_marker == "glossary"
        and node.get("tag") == "li"
        and not has_translation_marker(content, BLOCK_TRANSLATION_CLASS)
    ):
        text = extract_plain_text(content, english_only=False)
        translation = translations.get(text)
        if translation and translation != text:
            node["content"] = append_block_translation(content, translation)
            content = node["content"]

    elif marker in TRANSLATE_NODE_MARKERS and not has_translation_marker(
        content, BLOCK_TRANSLATION_CLASS
    ):
        text = extract_plain_text(content, english_only=False)
        translation = translations.get(text)
        if translation and translation != text:
            node["content"] = append_block_translation(content, translation)
            content = node["content"]

    elif marker == "example-sentence-b" and not has_translation_marker(
        content, BLOCK_TRANSLATION_CLASS
    ):
        text = extract_plain_text(content, english_only=True)
        translation = translations.get(text)
        if translation and translation != text:
            node["content"] = append_block_translation(content, translation)
            content = node["content"]

    elif marker in {
        "reference-label",
        "info-gloss-label",
        "sense-note-label",
        "lang-source-label",
    } and not has_translation_marker(content, INLINE_TRANSLATION_CLASS):
        if isinstance(content, str):
            translation = INLINE_LABEL_TRANSLATIONS.get(content)
            if translation and translation != content:
                node["content"] = append_inline_translation(content, translation)
                content = node["content"]

    next_parent = marker or parent_marker
    if isinstance(content, list):
        for item in content:
            apply_translations(item, translations, next_parent)
    elif isinstance(content, dict):
        apply_translations(content, translations, next_parent)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def default_target_dir(source_dir: Path) -> Path:
    name = source_dir.name
    if "[JA-EN]" in name:
        name = name.replace("[JA-EN]", "[JA-EN-ID]", 1)
    else:
        name = f"{name} [EN+ID]"
    return source_dir.with_name(name)


def ensure_target_tree(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        return
    log(f"Copying source dictionary to {target_dir.name}")
    shutil.copytree(source_dir, target_dir)


def update_index(index_path: Path) -> None:
    index_data = load_json(index_path)
    if not isinstance(index_data, dict):
        raise RuntimeError(f"Unexpected index format: {index_path}")

    title = index_data.get("title", "Jitendex")
    if "(EN+ID)" not in title:
        index_data["title"] = f"{title} (EN+ID)"

    note = (
        "\n\nThis local variant adds Indonesian glosses below the original English text. "
        "Update metadata has been removed so the custom bilingual content is preserved."
    )
    description = index_data.get("description", "")
    if note.strip() not in description:
        index_data["description"] = f"{description}{note}".strip()

    index_data.pop("isUpdatable", None)
    index_data.pop("indexUrl", None)
    index_data.pop("downloadUrl", None)
    dump_json(index_path, index_data)


def update_styles(styles_path: Path) -> None:
    styles = styles_path.read_text(encoding="utf-8")
    if BLOCK_TRANSLATION_CLASS in styles:
        return

    addition = f"""

span[data-sc-class="{BLOCK_TRANSLATION_CLASS}"] {{
    color: color-mix(in srgb, var(--text-color, var(--fg, #333)) 82%, #0b8457);
    display: inline-block;
    font-style: italic;
    margin-top: 0.15em;
}}

span[data-sc-class="{INLINE_TRANSLATION_CLASS}"] {{
    color: color-mix(in srgb, var(--text-color, var(--fg, #333)) 82%, #0b8457);
    font-style: italic;
}}
"""
    styles_path.write_text(styles + addition, encoding="utf-8")


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"completed": []}


def save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_term_banks(source_dir: Path, target_dir: Path) -> None:
    cache = TranslationCache(target_dir / ".id_translation_cache.sqlite3")
    translator = GoogleTranslateClient()
    progress_path = target_dir / ".id_translation_progress.json"
    progress = load_progress(progress_path)
    completed = set(progress.get("completed", []))

    term_banks = sorted(target_dir.glob("term_bank_*.json"))
    total_files = len(term_banks)

    try:
        for index, term_bank_path in enumerate(term_banks, start=1):
            if term_bank_path.name in completed:
                continue

            entries = load_json(term_bank_path)
            if not isinstance(entries, list):
                raise RuntimeError(f"Unexpected term bank format: {term_bank_path}")

            texts: set[str] = set()
            for entry in entries:
                texts.update(collect_texts(entry))

            unique_texts = sorted(texts)
            cached = cache.get_many(unique_texts)
            missing = [text for text in unique_texts if text not in cached]

            log(
                f"[{index}/{total_files}] {term_bank_path.name}: "
                f"{len(unique_texts)} texts, {len(missing)} new"
            )

            if missing:
                translated_count = 0
                for batch in chunked(missing, 500):
                    fresh = translator.translate_many(batch)
                    cache.store_many(fresh)
                    cached.update(fresh)
                    translated_count += len(batch)
                    log(f"  cached {translated_count}/{len(missing)} new translations")

            for entry in entries:
                apply_translations(entry, cached)

            dump_json(term_bank_path, entries)
            completed.add(term_bank_path.name)
            progress["completed"] = sorted(completed)
            save_progress(progress_path, progress)
    finally:
        cache.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Indonesian translations below English Jitendex entries."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=Path(
            "/Users/samit_kaiwa4/Downloads/Dictionaries/01 [JA-EN] jitendex-yomitan (2026-04-04)"
        ),
    )
    parser.add_argument("--target-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise RuntimeError(f"Source directory not found: {source_dir}")

    target_dir = (
        args.target_dir.expanduser().resolve()
        if args.target_dir is not None
        else default_target_dir(source_dir)
    )

    ensure_target_tree(source_dir, target_dir)
    update_index(target_dir / "index.json")
    update_styles(target_dir / "styles.css")
    process_term_banks(source_dir, target_dir)

    log(f"Finished bilingual dictionary at: {target_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:  # pragma: no cover
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
