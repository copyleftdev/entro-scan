from numba import cuda
import numpy as np
import math


class EntropyUtils:
    """
    A utility class for calculating various types of entropy.
    """

    @staticmethod
    @cuda.jit
    def shannon_entropy_kernel(freqs, entropy_result):
        """
        CUDA kernel for calculating Shannon entropy.

        Parameters:
            freqs: An array of frequencies on the device.
            entropy_result: An array to store the entropy result on the device.
        """
        i = cuda.grid(1)
        if i < freqs.size:  # Check array boundaries
            entropy_result[0] -= freqs[i] * math.log2(freqs[i])

    @staticmethod
    def shannon_entropy(freqs: np.ndarray) -> float:
        """
        Calculate Shannon entropy using CUDA.

        Parameters:
            freqs (numpy.ndarray): An array of frequencies.

        Returns:
            float: The calculated Shannon entropy.
        """
        freqs_device = cuda.to_device(freqs.astype(np.float32))
        entropy_result_device = cuda.to_device(np.array([0.0], dtype=np.float32))

        # Define block and grid sizes
        threads_per_block = 32
        blocks_per_grid = (freqs.size + (threads_per_block - 1)) // threads_per_block

        # Run the CUDA kernel
        EntropyUtils.shannon_entropy_kernel[blocks_per_grid, threads_per_block](freqs_device, entropy_result_device)

        # Copy the result back to the host
        entropy_result = entropy_result_device.copy_to_host()

        return float(entropy_result[0])

    @staticmethod
    @cuda.jit
    def renyi_entropy_kernel(freqs, alpha, entropy_result):
        """
        CUDA kernel for calculating Rényi entropy.

        Parameters:
            freqs: An array of frequencies on the device.
            alpha: The alpha parameter for Rényi entropy.
            entropy_result: An array to store the entropy result on the device.
        """
        i = cuda.grid(1)
        if i < freqs.size:  # Check array boundaries
            entropy_result[0] += math.pow(freqs[i], alpha)

    @staticmethod
    def renyi_entropy(freqs: np.ndarray, alpha: float) -> float:
        """
        Calculate Rényi entropy using CUDA.

        Parameters:
            freqs (numpy.ndarray): An array of frequencies.
            alpha (float): The alpha parameter for Rényi entropy.

        Returns:
            float: The calculated Rényi entropy.
        """
        if alpha == 1:
            return EntropyUtils.shannon_entropy(freqs)

        freqs_device = cuda.to_device(freqs.astype(np.float32))
        entropy_result_device = cuda.to_device(np.array([0.0], dtype=np.float32))

        # Define block and grid sizes
        threads_per_block = 32
        blocks_per_grid = (freqs.size + (threads_per_block - 1)) // threads_per_block

        # Run the CUDA kernel
        EntropyUtils.renyi_entropy_kernel[blocks_per_grid, threads_per_block](freqs_device, alpha,
                                                                              entropy_result_device)

        # Copy the result back to the host
        entropy_result = entropy_result_device.copy_to_host()

        return (1 / (1 - alpha)) * math.log2(entropy_result[0])

    @staticmethod
    @cuda.jit
    def hartley_entropy_kernel(n, entropy_result):
        """
        CUDA kernel for calculating Hartley entropy.

        Parameters:
            n: The number of possible outcomes.
            entropy_result: An array to store the entropy result on the device.
        """
        entropy_result[0] = math.log2(n)

    @staticmethod
    def hartley_entropy(n: int) -> float:
        """
        Calculate Hartley entropy using CUDA.

        Parameters:
            n (int): The number of possible outcomes.

        Returns:
            float: The calculated Hartley entropy.
        """
        if n <= 0:
            raise ValueError("The number of possible outcomes must be greater than 0.")

        n_device = cuda.to_device(np.array([n], dtype=np.int32))
        entropy_result_device = cuda.to_device(np.array([0.0], dtype=np.float32))

        # Run the CUDA kernel
        EntropyUtils.hartley_entropy_kernel[1, 1](n_device, entropy_result_device)

        # Copy the result back to the host
        entropy_result = entropy_result_device.copy_to_host()

        return float(entropy_result[0])

    @staticmethod
    @cuda.jit
    def tsallis_entropy_kernel(freqs, q, entropy_result):
        """
        CUDA kernel for calculating Tsallis entropy.

        Parameters:
            freqs: An array of frequencies on the device.
            q: The entropic index for Tsallis entropy.
            entropy_result: An array to store the entropy result on the device.
        """
        i = cuda.grid(1)
        if i < freqs.size:  # Check array boundaries
            entropy_result[0] += math.pow(freqs[i], q)

    @staticmethod
    def tsallis_entropy(freqs: np.ndarray, q: float) -> float:
        """
        Calculate Tsallis entropy using CUDA.

        Parameters:
            freqs (numpy.ndarray): An array of frequencies.
            q (float): The entropic index for Tsallis entropy.

        Returns:
            float: The calculated Tsallis entropy.
        """
        if q == 1:
            return EntropyUtils.shannon_entropy(
                freqs)  # Assuming you also have a CUDA-accelerated Shannon entropy method

        freqs_device = cuda.to_device(freqs.astype(np.float32))
        entropy_result_device = cuda.to_device(np.array([0.0], dtype=np.float32))

        # Define block and grid sizes
        threads_per_block = 32
        blocks_per_grid = (freqs.size + (threads_per_block - 1)) // threads_per_block

        # Run the CUDA kernel
        EntropyUtils.tsallis_entropy_kernel[blocks_per_grid, threads_per_block](freqs_device, q, entropy_result_device)

        # Copy the result back to the host
        entropy_result = entropy_result_device.copy_to_host()

        return (1 / (q - 1)) * (entropy_result[0] - 1)

    @staticmethod
    @cuda.jit
    def differential_entropy_kernel(data, n, entropy_result):
        """
        CUDA kernel for estimating differential entropy.

        Parameters:
            data: An array of data points on the device.
            n: The number of data points.
            entropy_result: An array to store the entropy result on the device.
        """
        i = cuda.grid(1)
        if i < n:  # Check array boundaries
            x = data[i]
            pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)  # Simplified PDF for a normal distribution
            if pdf > 0:
                entropy_result[0] -= pdf * math.log(pdf)

    @staticmethod
    def differential_entropy(data: np.ndarray) -> float:
        """
        Estimate differential entropy using CUDA.

        Parameters:
            data (numpy.ndarray): An array of data points.

        Returns:
            float: The estimated differential entropy.
        """
        n = len(data)
        data_device = cuda.to_device(data.astype(np.float32))
        entropy_result_device = cuda.to_device(np.array([0.0], dtype=np.float32))

        # Define block and grid sizes
        threads_per_block = 32
        blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

        # Run the CUDA kernel
        EntropyUtils.differential_entropy_kernel[blocks_per_grid, threads_per_block](data_device, n,
                                                                                     entropy_result_device)

        # Copy the result back to the host
        entropy_result = entropy_result_device.copy_to_host()

        return float(entropy_result[0])
