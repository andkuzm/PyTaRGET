from pathlib import Path

from data_processing.encode_tune_test import Eftt


def encoder_t():
    # Parameters
    annotated_cases_path = Path(r"")
    out_path = Path(r"")
    model = "codet5p"
    train_size = 0.8  # You can adjust this (e.g., 0.7, 0.9, etc.)

    # Initialize and run encoding
    experiment = Eftt(
        annotated_cases_path=annotated_cases_path,
        out_path=out_path,
        model=model,
        train_size=train_size
    )

    print("Encoding starting...")
    experiment.encode()
    print("Encoding completed.")

if __name__ == "__main__":
    encoder_t()