import platform
import sys
import torch


def select_device():
    """
    Select a usable device.
    Prefer CUDA if available AND kernels actually run.
    Fall back to CPU otherwise.
    """
    if torch.cuda.is_available():
        try:
            # Tiny CUDA op to verify kernel support
            x = torch.tensor([1.0], device="cuda")
            y = x * 2
            return torch.device("cuda")
        except RuntimeError as e:
            print("CUDA detected but not usable. Falling back to CPU.")
            print("Reason:", e)
    return torch.device("cpu")


def print_system_info():
    print("=== System Information ===")
    print("Python version:", sys.version.split()[0])
    print("Python executable:", sys.executable)
    print("Operating system:", platform.platform())
    print()


def print_framework_info(device):
    print("=== Framework Versions ===")
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print("Detected GPU:", name)
            print(f"Compute capability: sm_{cap[0]}{cap[1]}")
        except Exception:
            print("GPU detected but details unavailable")

    print("Selected device:", device)
    print()


def run_sanity_check(device):
    print("=== Sanity Check ===")
    # Simple tensor computation on selected device
    a = torch.randn(2, 3, device=device)
    b = torch.randn(3, 2, device=device)

    c = a @ b  # matrix multiplication
    print("Computation successful.")
    print("Result shape:", c.shape)
    print()


def main():
    print_system_info()

    device = select_device()

    print_framework_info(device)
    run_sanity_check(device)

    print("Environment check completed successfully ✔")


if __name__ == "__main__":
    main()