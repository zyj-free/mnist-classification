import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from models.init import get_model


def evaluate_saved_model(model_name, model_path):
    """评估已保存的模型"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载测试集
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型并加载权重
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 评估
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total

    print(f"模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"测试准确率: {accuracy:.2f}%")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    return accuracy


if __name__ == '__main__':
    # 评估最佳模型
    model_name = "simple_cnn"  # 修改为你训练的模型名称
    model_path = f"checkpoints/{model_name}_best.pth"

    try:
        accuracy = evaluate_saved_model(model_name, model_path)
    except FileNotFoundError:
        print(f"模型文件 {model_path} 不存在，请先训练模型")
    except Exception as e:
        print(f"评估过程中出错: {e}")