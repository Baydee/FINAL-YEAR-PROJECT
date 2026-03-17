import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import numpy as np
from fraud_detection_model import load_dataset, build_training_matrix

def generate_visuals():
    print("Loading dataset and training model to generate charts...")
    df = load_dataset('credit_card_fraud_10k.csv')
    
    X, y = build_training_matrix(df)
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    
    # 1. Generate Confusion Matrix Chart
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate (0)', 'Fraudulent (1)'], 
                yticklabels=['Legitimate (0)', 'Fraudulent (1)'],
                annot_kws={"size": 16})
    plt.title('Confusion Matrix - Predicted vs Actual', fontsize=16)
    plt.ylabel('Actual Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Saved 'confusion_matrix.png'!")
    
    # 2. Generate Metrics Table Image
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    table_data = [
        ["Metric", "Score"],
        ["Accuracy", f"{acc:.4f}"],
        ["Precision", f"{prec:.4f}"],
        ["Recall", f"{rec:.4f}"],
        ["F1-Score", f"{f1:.4f}"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    # Make header bold and colored
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
    
    plt.title('Model Performance Metrics', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('metrics_table.png', dpi=300)
    print("Saved 'metrics_table.png'!")

if __name__ == '__main__':
    generate_visuals()