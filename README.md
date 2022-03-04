# Monte Carlo Contracts

[![PyPI package](https://img.shields.io/pypi/v/monte-carlo-contracts)](https://pypi.python.org/pypi/monte-carlo-contracts)
[![Build status](https://github.com/luphord/monte-carlo-contracts/actions/workflows/monte-carlo-contracts-test.yml/badge.svg)](https://github.com/luphord/monte-carlo-contracts/actions)

Composable financial contracts with Monte Carlo valuation.
This module employs ideas from [How to Write a Financial Contract](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7885) by S. L. Peyton Jones and J-M. Eber.
However, the implementation is not based on functional programming but rather using an object oriented approach.
Also, this implementation is tailored towards Monte Carlo based cashflow generation whereas the paper favours more general methods.

## Features
* Composition of financial contracts using elementary contracts `Zero`, `One`, `Give`, `Scale`, `And`, `When`, `Cond`, `Anytime` and `Until`
* Boolean and real valued observables (stochastic processes) to be referenced by contracts
* Cashflow generation for composed contracts given simulation models on fixed dategrids

## Examples
* [Equity Options](examples/Equity%20Options.ipynb)
* [FX Options](examples/FX%20Options.ipynb)
* [Working with Observables](examples/Observables.ipynb)

## Install

You can install `monte-carlo-contracts` using `pip` with

```bash
pip3 install monte-carlo-contracts
```

or you can simply download `mcc.py` and then run it using `python3` with

```bash
python3 mcc.py
```

## History

See [HISTORY.md](HISTORY.md).

## Credits

Main developer is luphord <luphord@protonmail.com>. [cookiecutter-pyscript](https://github.com/luphord/cookiecutter-pyscript) is used as project template.
