# Monte Carlo Contracts

[![PyPI package](https://img.shields.io/pypi/v/monte-carlo-contracts)](https://pypi.python.org/pypi/monte-carlo-contracts)
[![Build status](https://travis-ci.com/luphord/monte-carlo-contracts.svg?branch=master)](https://travis-ci.com/github/luphord/monte-carlo-contracts)

Composable financial contracts with Monte Carlo valuation.
This module employs ideas from [How to Write a Financial Contract](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.7885) by S. L. Peyton Jones and J-M. Eber.
However, the implementation is not based on functional programming but rather using an object oriented approach.
Also, this implementation is tailored towards Monte Carlo based cashflow generation whereas the paper favours more general methods.

## Features
* `monte-carlo-contracts` is a single file Python project

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

### 0.2.0 (2020-10-11)
* Simulation of basic contracts `Zero`, `One`, `Give`, `Scale`, `And`, `When` and `Cond`
* Partial simulation of `Or` contract
* Float observables `Stock` and `FX`
* Boolean observables `At`
* `SimulatedCashflows` and model-bound `IndexedCashflows` to represent cashflows
* Basic `Model` allowing the generation of cashflows for the contracts above

### 0.1.0 (2020-09-22)
* Created using [cookiecutter-pyscript](https://github.com/luphord/cookiecutter-pyscript)

## Credits

Main developer is luphord <luphord@protonmail.com>. [cookiecutter-pyscript](https://github.com/luphord/cookiecutter-pyscript) is used as project template.
