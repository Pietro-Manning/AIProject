import tensorflow as tf
import subprocess


def get_gpu_name():
    """
    Retrieve the name of the GPU using the `nvidia-smi` command.

    This function executes the `nvidia-smi` command with appropriate arguments
    to extract the GPU name. It attempts to fetch the information and decode the
    output. If an error occurs during execution, a formatted error message
    describing the failure is returned.

    :return: The GPU name as a string if the command executes successfully. If an
             exception occurs, a string containing the error message is returned.
    :rtype: str
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip()
    except Exception as e:
        return f"Error retrieving GPU name: {e}"


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    gpu_name = get_gpu_name()
    print(f"TensorFlow detected the following GPU: {gpu_name}")
else:
    print("TensorFlow did not detect any GPU.")
