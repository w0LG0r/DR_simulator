from datetime import datetime
import logging
import sys
import os

# logger.logs instance
log = logging.getLogger(__name__)
# log.disabled = True

log.setLevel(logging.DEBUG)


def create_file_handler(scenario: str):

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)

    # Choose formatter:

    # Without line
    stdout_formatter = logging.Formatter("%(message)s")

    # With filename and line
    # stdout_formatter = logging.Formatter('{%(filename)s:%(lineno)d} %(message)s')

    # With path and line
    # stdout_formatter = logging.Formatter('{%(pathname)s:%(lineno)d} %(message)s')

    stdout_handler.setFormatter(stdout_formatter)

    log.addHandler(stdout_handler)

    # File handler for LOG file
    filename=scenario.result_directory + "/"  + scenario.name + "_LOG_" + ".txt"

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(stdout_formatter)

    log.addHandler(file_handler)

    # File handler for SUMMARY file and STDOUT
    filename=scenario.result_directory + "/"  + scenario.name + "_SUMMARY_" + ".txt"
    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(stdout_formatter)

    log.addHandler(file_handler)

    return  file_handler
