from app.utils.data_loader import load_dataset
X_train, y_train, X_test, y_test = load_dataset()
print(X_train[:5])
print(y_train[:5])
print(X_test[:5])
print(y_test[:5])
print(len(X_train))
print(len(X_test))