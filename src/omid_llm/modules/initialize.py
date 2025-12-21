#!/usr/bin/python3
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from argparse import Namespace



def _creat_namespace_from_dict(input_dict):
	for key, value in input_dict.items():
		if isinstance(value, dict):
			input_dict[key] = _creat_namespace_from_dict(value)
		else:
			input_dict[key] = value
	input_dict = Namespace(**input_dict)
	return input_dict


def update_config(path_to_config: str, args: dict):
	""" This function updates the config; you can modify this to pass modules as a dict.

	Args:
		path_to_config: Path o config file.

	Returns:
		updated_config: Updated config dict.

	"""
	# Get the config
	with Path(path_to_config).open() as f:
		config = yaml.load(f, Loader=yaml.SafeLoader)

	# Update the config
	for key in args.keys():
		# If value is None forget updating them
		if args[key]:
			split_key_parts = key.split(".")

			eval_phrase = "config"
			if len(split_key_parts) != 1:
				for split_key_part in split_key_parts[:-1]:
					eval_phrase += f"['{split_key_part}']"
			eval(eval_phrase)[split_key_parts[-1]] = args[key]

	updated_config = _creat_namespace_from_dict(config)

	return updated_config


def set_argparse():
	"""Define the arguments to be parsed.

	Args: None

	Returns:
		config: updated config dict.

	"""
	ap = argparse.ArgumentParser()

	ap.add_argument(
		"-v", "--version",
		default=None,
		required=False,
		help="What version is the code? If it is None, the default value from config will be used."
	)

	ap.add_argument(
		"-i", "--inputs.input_dir",
		default=None,
		required=False,
		help="Where is the input directory? If it is None, the default value from config will be used."
	)

	ap.add_argument(
		"-o", "--outputs.output_dir",
		default=None,
		required=False,
		help="Where is the input directory? If it is None, the default value from config will be used."
	)

	ap.add_argument(
		"-c", "--config_path",
		default='./config.yaml',
		required=True,
		help="Where is the config file?"
	)

	args = vars(ap.parse_args())

	config = update_config(args["config_path"], args)

	return config


def setup_logger(
	path_to_logger: str = "./log.log",
):
	"""
	Setup logger.

	Args:
		path_to_logger: Path to where saving the log file.=

	Returns:
		logger: logger to keep track of prints and errors.

	"""
	# logging.getLogger("hydra").disabled = True
	# name = Path(path_to_logger).stem
	# logger = logging.getLogger(name)

	logger = logging.getLogger("hydra")

	formatter = logging.Formatter(
		fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
	)
	handler = logging.FileHandler(path_to_logger, mode='w')
	handler.setFormatter(formatter)
	screen_handler = logging.StreamHandler(stream=sys.stdout)
	screen_handler.setFormatter(formatter)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(handler)
	logger.addHandler(screen_handler)

	return logger
