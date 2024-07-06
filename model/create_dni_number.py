"""
Create the DNI number
"""
import numpy as np

def generate_dni() -> int:
    """Generate a DNI number to help to a unique dataset
    creation using Scikit Learn

    Returns:
        - a DNI as an integer number. This is going to be randomly generated
    """
    # First, generate 8 random numbers as a string.
    # We're using the max to ensure that we're not going to have numbers
    # below 2 in the DNI
    random_dni = "".join(
        str(max(2, np.random.randint(0, 9))) for _ in range(0,8)
        )
    # At the end, just convert this to an integer and return it as random
    return int(random_dni)


if __name__ == "__main__":
    generate_dni()
