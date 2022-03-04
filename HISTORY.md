## History

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