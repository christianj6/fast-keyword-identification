from ..run import run
import argparse


class ManagementUtility:
    def __init__(self, argv: list = None) -> None:
        self.argv = argv
        self.parser = self.setup_parser()

    @classmethod
    def setup_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-k",
            "--keywords",
            type=str,
            help="Absolute filepath to .csv file containing keywords.",
        )
        parser.add_argument(
            "-c",
            "--corpus",
            type=str,
            help="Absolute path to .csv file containing corpus of documents to search.",
        )

        return parser

    def execute(self) -> None:
        args = self.parser.parse_args()
        run(**vars(args))


def execute_from_command_line(argv: list = None) -> None:
    utility = ManagementUtility(argv)
    utility.execute()
