import logging

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(assctime)s | %(name)s | %(message)s",
    )