from __future__ import annotations

import re
import stat
import unicodedata
import zipfile
import zlib
from pathlib import Path, PurePosixPath


MAX_ARCHIVE_FILES = 1_000
MAX_ARCHIVE_MEMBER_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES = 500 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250.0
ALLOWED_ARCHIVE_EXTENSIONS = {".csv", ".gpx"}
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
ARCHIVE_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    RuntimeError,
    OSError,
    EOFError,
    NotImplementedError,
    zlib.error,
)


class ArchiveValidationError(ValueError):
    pass


def _display_name(value: str) -> str:
    clean = "".join(character if character.isprintable() else "?" for character in value)
    return clean[:120] or "unnamed member"


def _normalized_member(member: zipfile.ZipInfo) -> PurePosixPath:
    raw = member.filename.replace("\\", "/")
    if raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw):
        raise ArchiveValidationError(f"ZIP contains an absolute path: {_display_name(member.filename)}")
    path = PurePosixPath(raw)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveValidationError(f"ZIP contains an unsafe path: {_display_name(member.filename)}")
    for part in path.parts:
        normalized = unicodedata.normalize("NFKC", part)
        if ":" in normalized:
            raise ArchiveValidationError(f"ZIP contains a Windows ADS path: {_display_name(member.filename)}")
        if normalized.endswith((".", " ")):
            raise ArchiveValidationError(f"ZIP contains a trailing dot or space: {_display_name(member.filename)}")
        base = normalized.split(".", 1)[0].rstrip(" .").upper()
        if base in WINDOWS_RESERVED:
            raise ArchiveValidationError(f"ZIP contains a reserved device name: {_display_name(member.filename)}")
    return path


def validate_zip(
    path: Path,
    *,
    allowed_extensions: set[str] = ALLOWED_ARCHIVE_EXTENSIONS,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_FILES:
                raise ArchiveValidationError(f"ZIP contains more than {MAX_ARCHIVE_FILES} entries.")
            expanded = 0
            seen: set[str] = set()
            validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for member in members:
                member_path = _normalized_member(member)
                key = unicodedata.normalize("NFKC", member_path.as_posix()).casefold()
                if key in seen:
                    raise ArchiveValidationError(
                        f"ZIP contains colliding filenames: {_display_name(member.filename)}"
                    )
                seen.add(key)
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                allowed_type = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
                if file_type == stat.S_IFLNK:
                    raise ArchiveValidationError(
                        f"ZIP contains a symbolic link: {_display_name(member.filename)}"
                    )
                if file_type not in (0, allowed_type):
                    raise ArchiveValidationError(
                        f"ZIP contains a non-regular file: {_display_name(member.filename)}"
                    )
                if member.flag_bits & 0x1:
                    raise ArchiveValidationError(
                        f"ZIP contains an encrypted file: {_display_name(member.filename)}"
                    )
                if member.is_dir():
                    validated.append((member, member_path))
                    continue
                if member_path.suffix.lower() not in allowed_extensions:
                    raise ArchiveValidationError(
                        f"ZIP contains an unsupported file: {_display_name(member.filename)}"
                    )
                if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ArchiveValidationError(
                        f"ZIP member is larger than 100 MB: {_display_name(member.filename)}"
                    )
                expanded += member.file_size
                if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                    raise ArchiveValidationError("ZIP expands beyond the 500 MB safety limit.")
                ratio = member.file_size / max(member.compress_size, 1)
                if member.file_size > 1024 and ratio > MAX_COMPRESSION_RATIO:
                    raise ArchiveValidationError(
                        f"ZIP member has an excessive compression ratio: {_display_name(member.filename)}"
                    )
                validated.append((member, member_path))
            return validated
    except ArchiveValidationError:
        raise
    except ARCHIVE_ERRORS as error:
        raise ArchiveValidationError("ZIP file is malformed or unreadable.") from error


def safe_extract_zip(
    path: Path,
    destination: Path,
    *,
    allowed_extensions: set[str] = ALLOWED_ARCHIVE_EXTENSIONS,
) -> list[Path]:
    validated = validate_zip(path, allowed_extensions=allowed_extensions)
    root = destination.resolve()
    extracted: list[Path] = []
    total_written = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for member, relative in validated:
                target = (root / Path(*relative.parts)).resolve()
                if target != root and root not in target.parents:
                    raise ArchiveValidationError("ZIP contains an unsafe extraction path.")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                member_written = 0
                with archive.open(member) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        member_written += len(chunk)
                        total_written += len(chunk)
                        if member_written > MAX_ARCHIVE_MEMBER_BYTES:
                            raise ArchiveValidationError("ZIP member exceeded the 100 MB extraction limit.")
                        if total_written > MAX_ARCHIVE_EXPANDED_BYTES:
                            raise ArchiveValidationError("ZIP exceeded the 500 MB extraction limit.")
                        output.write(chunk)
                if member_written != member.file_size:
                    raise ArchiveValidationError("ZIP member size changed during extraction.")
                extracted.append(target)
    except ArchiveValidationError:
        raise
    except ARCHIVE_ERRORS as error:
        raise ArchiveValidationError("ZIP file failed integrity checks during extraction.") from error
    return extracted
