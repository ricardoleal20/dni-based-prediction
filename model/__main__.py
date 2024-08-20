"""
Module implementation of the problem.
It also includes the ArgParser to decide which section
do we want to run
"""
from argparse import ArgumentParser
# Local imports
from model.implementation import generate_model
from model.create_dni_number import generate_dni

if __name__ == "__main__":
    # Define the parser
    parser = ArgumentParser(description="Parser for the Regression Problem")
    # Define the args
    parser.add_argument("-a", "--algorithm", default="all")
    parser.add_argument("-v", "--verbose", type=bool, default=False)
    parser.add_argument("-l", "--latex", type=bool, default=False)
    parser.add_argument("-f", "--function", default=None)
    parser.add_argument("-d", "--dni", default=None)
    # Parse the arguments
    args = parser.parse_args()
    if not args.function:
        raise ValueError(
            "You should decide an function."+
            " The options are `-f dni` or `-f models`."
        )
    match args.function:
        case "models":
            if not args.dni or not len(args.dni) == 8:
                raise ValueError("You should add an 8 length DNI")
            dni: int = 0
            try:
                dni = int(args.dni)
            except Exception as exc:
                raise TypeError(f"The DNI provided {args.dni} is not a number.") from exc
            # Then, evaluate the model to use
            if args.algorithm not in [
                "all", "regression", "classification",
                "lineal", "elastic", "step_wise",
                "decision_tree", "bagging", "pasting", "random_forest", "gbm"
            ]:
                raise NotImplementedError(
                    f"The algorithm selected was {args.algorithm} "+
                    "but the only available options are: "+
                    '["all", "regression", "classification", ' +
                    '"lineal", "elastic", "step_wise",' +
                    ' "decision_tree", "bagging", "pasting", ' +
                    '"random_forest", "gbm"].'
                    )
            generate_model(
                str(dni),
                args.algorithm,
                args.verbose,
                args.latex
            )
        case "dni":
            DNI = generate_dni()
            print(f"The DNI generated is '{DNI}'.")
        case _:
            raise NotImplementedError(
                f"The function {args.function} is not implemented. "+
                "Use `-f dni` or `-f models`."
            )
