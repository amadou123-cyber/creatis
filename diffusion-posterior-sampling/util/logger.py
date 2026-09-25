import logging


def get_logger(name):
    if name == "ps":
        name = "dps"
    logger = logging.getLogger(name=name.upper())
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
