# Fourth Year Project

- `implementations/`: Contains paper implementations with modifications and
  improvements for portability. Each implementation is a separate directory, and
  contains a `README.md` file with instructions on how to run the code. It
  should contain a `pyproject.toml` instead of a `requirements.txt` file. This
  makes it easier to install dependencies with `poetry`. `poetry` is a
  dependency manager that is similar to `go mod` and `npm` - it is very exact
  about the dependencies it installs, and it creates a `poetry.lock` file that
  specifies the exact versions of the dependencies that were installed. This
  makes it easier to reproduce the environment in which the code was run.
