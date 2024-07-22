import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_mitbih_data
from model import ECGModel

def load_trained_model(model_path, input_size, num_classes=5):
    model = ECGModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    # Load data
    data_dir = 'path_to_mitbih_data'
    model_path = 'path_to_saved_model.pth'
    
    X_train, X_test, y_train, y_test = load_mitbih_data(data_dir)

    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    input_size = X_test.shape[1]
    model = load_trained_model(model_path, input_size)

    
    evaluate_model(model, test_loader)
