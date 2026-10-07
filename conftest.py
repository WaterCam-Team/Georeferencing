import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Local site data and backups (gitignored); never collect tests from there.
collect_ignore = ["private"]
