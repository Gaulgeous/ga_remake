from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, ConfusionMatrixDisplay, mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score as r2

def calc_final_regression(y_test, predictions):
    RMSE = mse(y_test, predictions, squared=False)
    MAE = mae(y_test, predictions)
    MAPE = mape(y_test, predictions)
    R2 = r2(y_test, predictions)

    print()
    print()
    print(f"Final RMSE: {RMSE}")
    print(f"Final MAE: {MAE}")
    print(f"Final MAPE: {MAPE}")
    print(f"Final R2: {R2}")


def calc_final_classification(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)

    print()
    print()
    print(f"Final Accuracy: {accuracy}")
    print(f"Final f1: {f1}")
    print(f"Final recall: {recall}")
    print(f"Final precision: {precision}")

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()