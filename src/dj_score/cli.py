"""`dj-score` console entry point. Thin: parse arguments, call a service, print."""

import argparse

from dj_score import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dj-score", description=__doc__)
    parser.add_argument("--version", action="version", version=f"dj-score {__version__}")
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
