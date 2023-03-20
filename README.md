# Monte Carlo Contracts

[![PyPI package](https://img.shields.io/pypi/v/monte-carlo-contracts)](https://pypi.python.org/pypi/monte-carlo-contracts)
[![Build status](https://github.com/luphord/monte-carlo-contracts/actions/workflows/monte-carlo-contracts-test.yml/badge.svg)](https://github.com/luphord/monte-carlo-contracts/actions)

Composable financial contracts with Monte Carlo valuation.
This module employs ideas from [How to Write a Financial Contract](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7885) by S. L. Peyton Jones and J-M. Eber.
However, the implementation is not based on functional programming but rather using an object oriented approach.
Also, this implementation is tailored towards Monte Carlo based cashflow generation whereas the paper favours more general methods.

## Features
* Composition of financial contracts using elementary contracts `Zero`, `One`, `Give`, `Scale`, `And`, `When`, `Cond`, `Anytime`, `Until` and `Delay`
* Boolean and real valued observables (stochastic processes) to be referenced by contracts
* Cashflow generation for composed contracts given simulation models on fixed dategrids

## Examples
* [Equity Options](examples/Equity%20Options.ipynb)
* [FX Options](examples/FX%20Options.ipynb)
* [Working with Observables](examples/Observables.ipynb)
* [Cashflow types](examples/Cashflows.ipynb)

## Install

With Python 3.8+ on your machine, you can install `monte-carlo-contracts` using `pip` by running (ideally in a [virtual environment](https://docs.python.org/3/glossary.html#term-virtual-environment))

```bash
pip install monte-carlo-contracts
```

which will automatically install the hard dependencies `numpy` and `pandas`.

For development or running the examples, you may instead want to run

```bash
pip install -e .
```

and then

```bash
pip install -r requirements_dev.txt
```

from the root directory of this repository.

## History

See [HISTORY.md](HISTORY.md).

## Credits

Main developer is luphord <luphord@protonmail.com>.

[cookiecutter-pyscript](https://github.com/luphord/cookiecutter-pyscript) was used as project template, but the repository structure has evolved considerably.
