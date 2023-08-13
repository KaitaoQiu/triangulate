# Triangulate

Triangulate is a research project to improve bug localization by using
reinforcement learning.

[![Unittests](https://github.com/google-research/triangulate/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/triangulate/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/triangulate.svg)](https://badge.fury.io/py/triangulate)

*This is not an officially supported Google product.*

## Installation

Run the following command to install Triangulate.

```
pip3 install -e .
```

## Usage

Triangulate is designed to wrap a standard Python script invocation (a "subject" program), similar to `pdb`:

```
python3 -m triangulate.main [flags...] subject -- [subject_args...]
```

Under the hood, Triangulate runs `subject [subject_args...]`. If an exception is raised, Triangulate uses information from the exception to begin bug localization.

## Configuration

Triangulate uses [python-dotenv][] for configuration.

Triangulate recognizes the following configuration variables, which can be defined as
environment variables or in a `.env` file.
- `PALM_API_KEY`
- `TRIANGULATE_LLM_TIMEOUT_SECONDS`

## Examples

```
python3 -m triangulate.main --max_steps 5 -- triangulate/testdata/quoter.py -- --index 1
```

[python-dotenv]: https://github.com/theskumar/python-dotenv
