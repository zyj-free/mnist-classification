from .simple_cnn import SimpleCNN
from .resnet import ResNetMNIST

def get_model(model_name, num_classes=10):
    """根据名称获取模型"""
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return ResNetMNIST(num_classes=num_classes, version=18)
    elif model_name == "resnet34":
        return ResNetMNIST(num_classes=num_classes, version=34)
    else:
        raise ValueError(f"未知模型: {model_name}")

def get_available_models():
    """获取可用模型列表"""
    return ["simple_cnn", "resnet18", "resnet34"]