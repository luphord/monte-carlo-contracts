## History

### 0.10.0 (not yet)
* ToDo: Add new `Exchange(currency, contract)` contract
* ToDo: Modify FX options examples to use `Exchange` for cash settlement
* ToDo: Observables `Sum`, `Product`, `Maximum`, `Minimum`, `AndObservable` and `OrObservable` accept more than two contracts to be combined

### 0.9.0 (not yet)
* ToDo: Split `mcc.py` into multiple modules forming package `mcc`
* ToDo: Split tests into multiple modules
* Remove CLI stub

### 0.8.0 (2023-03-20)
* **BREAKING CHANGE**: `Contract` now inherits from `ResolvableContract` instead of the other way round
* **BREAKING CHANGE**: `And` and `Or` contracts now accept more than two contracts to be combined; these have equivalent semantics to nested `And` or `Or` contracts and allow for flat structures to improve readability
* Add `Delay(observableBool, contract)` contract to delay cashflows to a later point in time (main use case is FX payment offset)
* First steps towards model requirements (yet incomplete)
* Fix cashflow generation for nested contracts

### 0.7.0 (2022-03-13)
* **BREAKING CHANGE**: `ObservableFloat.simulate` and `ObservableBool.simulate` now accept a `DateIndex` `first_observation_idx` as first argument, `Contract` classes will pass `acquisition_idx`; this allows observations to depend on the time of entering a contract, e.g. "maximum spot since acquisition"
* **BREAKING CHANGE**: `FixedAfter` fixes composed observable after (including) `first_observation_idx`, not from the beginning
* Add operator overloading for `Contract` classes, i.e. you can now do `One("USD") - One("EUR") | 1.2 * One("GBP")` instead of `Or(And(One("USD"), Give(One("EUR"))), Scale(1.2, One("GBP")))`
* `Maximum` and `Minimum` observables to observe the larger and smaller value of two observables at the same time on the same path
* `RunningMax` and `RunningMin` observables to observe running extreme values from `first_observation_idx` onwards
* Support Python 3.10
* Make use of type annotations added to numpy

### 0.6.0 (2022-03-04)

* **BREAKING CHANGE**: Make `SimpleCashflows` a `pandas.DataFrame`
* Run notebooks in automated tests using [nbval](https://github.com/computationalmodelling/nbval)
* Migrate from travis-ci to [GitHub Actions](https://github.com/luphord/monte-carlo-contracts/actions)
* Explicitly support Python 3.8 and 3.9
* Move history to HISTORY.md

### 0.5.0 (2020-11-08)

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