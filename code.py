import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# Sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y_linear = 2.5 * X + np.random.normal(0, 1, X.shape)
y_logistic = (y_linear > 15).astype(int)
y_tree = y_logistic.ravel()

# Train models
linear_model = LinearRegression().fit(X, y_linear)
logistic_model = LogisticRegression().fit(X, y_logistic)
tree_model = DecisionTreeClassifier().fit(X, y_tree)

# Predictions
linear_pred = linear_model.predict(X)
logistic_pred = logistic_model.predict_proba(X)[:, 1]
tree_pred = tree_model.predict(X)

# Tkinter Application
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Dashboard")

        # Dropdown for model selection
        self.model_var = tk.StringVar(value="Linear Regression")
        models = ["Linear Regression", "Logistic Regression", "Decision Tree"]
        ttk.Label(root, text="Select Model:").grid(row=0, column=0, padx=10, pady=5)
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, values=models, state="readonly")
        self.model_dropdown.grid(row=0, column=1, padx=10, pady=5)
        self.model_dropdown.bind("<>", self.update_plot)

        # Canvas for matplotlib plot
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Initial plot
        self.update_plot()

    def update_plot(self, event=None):
        model = self.model_var.get()

        # Clear previous plot
        self.ax.clear()

        if model == "Linear Regression":
            self.ax.scatter(X, y_linear, color="blue", label="Actual")
            self.ax.plot(X, linear_pred, color="red", label="Predicted")
            self.ax.set_title("Linear Regression")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("y")
        elif model == "Logistic Regression":
            self.ax.scatter(X, y_logistic, color="blue", label="Actual")
            self.ax.plot(X, logistic_pred, color="red", label="Predicted Probability")
            self.ax.set_title("Logistic Regression")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Probability")
        elif model == "Decision Tree":
            self.ax.scatter(X, y_tree, color="blue", label="Actual")
            self.ax.plot(X, tree_pred, color="red", label="Predicted")
            self.ax.set_title("Decision Tree Classifier")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Class")

        self.ax.legend()
        self.canvas.draw()

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
