import os, sys

sys.stdout.reconfigure(encoding='utf-8')

IGNORE = {
    ".git", ".claude", "node_modules", "__pycache__", ".DS_Store",
    "dist", "build", ".venv", ".godot", "export", ".import", "imported",
    "AI Generate", "Modern tiles_Free",
    # project-specific (gitignored): datasets, model artifacts, run logs, vault
    "data", "models", "logs", ".pytest_cache", "vault",
}


def walk(root, prefix=""):
    try:
        entries = sorted(
            [e for e in os.scandir(root) if e.name not in IGNORE],
            key=lambda e: (not e.is_dir(), e.name.lower()),
        )
    except PermissionError:
        return
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        connector = "└── " if last else "├── "
        suffix = "/" if entry.is_dir() else ""
        print(prefix + connector + entry.name + suffix)
        if entry.is_dir():
            ext = "    " if last else "│   "
            walk(entry.path, prefix + ext)


print(".")
walk(".")
