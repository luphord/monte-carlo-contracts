from .model import Model
from .contracts import Contract
from .cashflows import SimulatedCashflows, SimpleCashflows


def generate_cashflows(model: Model, contract: Contract) -> SimulatedCashflows:
    return contract.generate_cashflows(model.eval_date_index, model).apply_index()


def generate_simple_cashflows(model: Model, contract: Contract) -> SimpleCashflows:
    return generate_cashflows(model, contract).to_simple_cashflows()


def generate_simple_cashflows_in_currency(
    model: Model, contract: Contract, currency: str
) -> SimpleCashflows:
    return (
        model.in_currency(
            contract.generate_cashflows(model.eval_date_index, model), currency
        )
        .apply_index()
        .to_simple_cashflows()
    )


def generate_simple_cashflows_in_numeraire_currency(
    model: Model, contract: Contract
) -> SimpleCashflows:
    return generate_simple_cashflows_in_currency(
        model, contract, model.numeraire_currency
    )


def evaluate(model: Model, contract: Contract) -> float:
    cf = contract.generate_cashflows(model.eval_date_index, model)
    return float(model.discount(cf).sum(axis=1).mean(axis=0))
