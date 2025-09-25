# BELKAml

<<<<<<< HEAD
End-to-end application for predicting binding affinity of small molecules to protein targets with deep learning.

## Installation instructions

### Python verison requirements
This package requires Python 3.10 or later. Please verify your Python version by running `python --version` in your terminal. If you're not running Python 3.10+, please adjust your environment accordingly (for example, if you use pyenv: `pyenv local 3.10` or any later version like `pyenv local 3.11`).

> [!NOTE]
> To install pyenv, follow the instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation), then run `eval "$(pyenv init -)"`.

### Dependencies
Dependencies are managed in `pyproject.toml`. To install the dependencies, run the following commands, based on your Operating System:

**POSIX Systems (MacOS/Linux):**

```bash
python --version   # Ensure this outputs Python 3.10 or later
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]' # Install in editable mode
```
**Windows Systems:**
```powershell
python --version   # Ensure this outputs Python 3.10 or later
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e '.[dev]' # Install in editable mode
```

Once installed, you can use the package with:
```python
import belkaml as bml
```

> [!NOTE]
> If using VSCode, make sure to set the interpreter to the .venv environment using `Cmd + Shift + P` -> `Python: Select Interpreter`.

## Submitting Changes
### Linting
Before submitting changes, please run pre-commit hooks to ensure that the code is formatted correctly. To do so, run the following command:
```bash
pre-commit run --a
```
The linter should run without any errors and autofix any issues that it can. If there are any issues that the linter cannot fix, please fix them manually before committing your changes.


### Tests
Please ensure that all tests pass before submitting your changes. To run the tests, run the following command:
```bash
pytest
```
Once all tests pass, you may submit your changes.

## Contributing
We welcome contributions to this package! If you have any ideas for new features, bug fixes, or improvements, please feel free to open an issue or a pull request. We appreciate your help in making this package better!

## License
This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
=======
Test
>>>>>>> 4fec372 (Test push)
