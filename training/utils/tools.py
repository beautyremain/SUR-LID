import psutil
import torch
import logging
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import torch.nn as nn
logger = logging.getLogger(__name__)


def calculate_model_flops(model, input_shape):
    """
    计算 PyTorch 模型的 FLOPs（浮点运算次数）
    :param model: PyTorch 模型实例
    :param input_shape: 输入张量的形状，例如 (batch_size, channels, height, width)
    :return: 总 FLOPs，单位为 FLOPs（整数）
    """

    dummy_input = torch.randn(*input_shape)

    # 使用 fvcore 计算 FLOPs
    flops = FlopCountAnalysis(model, dummy_input).total()
    return flops




def print_cpu_gpu_usage(title=None):
    if title is not None:
        logger.info(f"------{title} Begin------")
    # 获取 CPU 占用率
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_total = psutil.virtual_memory().total / (1024 ** 3)  # 转换为 GB
    cpu_used = psutil.virtual_memory().used / (1024 ** 3)    # 转换为 GB

    # 打印 CPU 信息
    logger.info(f"Current CPU Usage: {cpu_usage}%")
    logger.info(f"Total CPU Memory: {cpu_total:.2f} GB")
    logger.info(f"Used CPU Memory: {cpu_used:.2f} GB")

    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # 转换为 MB
        gpu_used = torch.cuda.memory_allocated(0) / (1024 ** 2)                    # 转换为 MB
        gpu_usage = (gpu_used / gpu_total) * 100

        # 打印 GPU 信息
        logger.info(f"Current GPU Usage: {gpu_usage:.2f}%")
        logger.info(f"Total GPU Memory: {gpu_total:.2f} MB")
        logger.info(f"Used GPU Memory: {gpu_used:.2f} MB")
    else:
        logger.info("No available GPU.")
    if title is not None:
        logger.info(f"------{title} End------")

if __name__ == "__main__":
    print_cpu_gpu_usage()
