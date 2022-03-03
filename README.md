# Monte Carlo Contracts

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

### 0.5.0 (2020--11-08)

* **BREAKING CHANGE**: Add `simulated_rates` to `Model` (included in constructor);
  pass an empty dict for `simulated_rates` to adapt your code
* **BREAKING CHANGE**: `BrownianMotion` and `GeometricBrownianMotion` generalized to
  dynamic mean/drift; pass `mu_t = lambda t: mu * t` to adapt your code
* `LinearRate` observable supported by `TermStructureModel`
* First steps towards term structure models
* `FixedAfter` observable to keep an observable fixed after a condition is true
* Observables support arithmetic operations (binary `+`, `-`, `*`, `/`, `**` and unary `-`)
  with other observables as well as constants (also right operators work)
* [Working with Observables](examples/Observables.ipynb) example notebook

### 0.4.0 (2020-11-04)

* Discounting (`Model.discount`)
* Evaluation (`Model.evaluate`)
* String representations for contracts and observables

### 0.3.0 (2020-10-23)
* Simulation of basic contract `Until`
* Currency conversion of `IndexedCashflows`
* `Or` contract supports multiple currencies
* `ObservableFloat` supports `<`, `<=`, `>` and `>=` operators with `float` or other `ObservableFloat` instances
* `ObservableBool` supports `~`, `&` and `|` operators for combined conditions
* [Equity Options](examples/Equity%20Options.ipynb) and [FX Options](examples/FX%20Options.ipynb) examples

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
