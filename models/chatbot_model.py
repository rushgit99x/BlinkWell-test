import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import string
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChatbotDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out

class NLTKUtils:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def tokenize(self, sentence):
        """Tokenize sentence into array of words/tokens"""
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        """Stemming = find the root form of the word"""
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence, words):
        """Return bag of words array: 1 for each known word that exists in the sentence, 0 otherwise"""
        stemmed_sentence = [self.stem(w) for w in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in stemmed_sentence:
                bag[idx] = 1
        return bag

    def preprocess_data(self, intents_file):
        """Load and preprocess the training data"""
        with open(intents_file, 'r') as f:
            intents = json.load(f)

        all_words = []
        tags = []
        xy = []

        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                w = self.tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))

        ignore_words = ['?', '.', '!', ',', "'", '"']
        all_words = [self.stem(w) for w in all_words if w not in ignore_words]
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        print(f"Unique stemmed words: {len(all_words)}")
        print(f"Tags: {len(tags)}")

        X = []
        y = []
        for (pattern_sentence, tag) in xy:
            bag = self.bag_of_words(pattern_sentence, all_words)
            X.append(bag)
            label = tags.index(tag)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        return X, y, all_words, tags

class AccuracyVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epochs = []
        
        # Create results directory
        self.results_dir = "training_results_chatbot"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def update_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update metrics for visualization"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
    
    def plot_training_history(self, save_path=None):
        """Generate training history plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plots
        ax2.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final accuracy bar chart
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        
        ax3.bar(['Training', 'Validation'], [final_train_acc, final_val_acc], 
                color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black')
        ax3.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate([final_train_acc, final_val_acc]):
            ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Loss vs Accuracy scatter plot
        ax4.scatter(self.train_losses, self.train_accuracies, alpha=0.6, label='Training', s=30)
        ax4.scatter(self.val_losses, self.val_accuracies, alpha=0.6, label='Validation', s=30)
        ax4.set_title('Loss vs Accuracy Relationship', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"training_history_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.show()
        
        return save_path
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """Generate confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"confusion_matrix_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
        
        return save_path
    
    def plot_accuracy_gauge(self, accuracy, title="Model Accuracy", save_path=None):
        """Generate accuracy gauge chart"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a simple circular gauge
        angles = np.linspace(0, np.pi, 100)
        
        # Background arc
        x_bg = np.cos(angles)
        y_bg = np.sin(angles)
        ax.plot(x_bg, y_bg, linewidth=20, color='lightgray', alpha=0.3)
        
        # Accuracy arc
        accuracy_angle = np.pi * (accuracy / 100)
        accuracy_angles = angles[angles <= accuracy_angle]
        
        if accuracy >= 80:
            color = 'green'
        elif accuracy >= 60:
            color = 'orange'
        else:
            color = 'red'
        
        if len(accuracy_angles) > 0:
            x_acc = np.cos(accuracy_angles)
            y_acc = np.sin(accuracy_angles)
            ax.plot(x_acc, y_acc, linewidth=20, color=color, alpha=0.8)
        
        # Add accuracy text in center
        ax.text(0, 0.3, f'{accuracy:.1f}%', ha='center', va='center', 
                fontsize=28, fontweight='bold', color=color)
        ax.text(0, 0.1, 'Accuracy', ha='center', va='center', 
                fontsize=14, color='gray')
        
        # Add scale labels
        scale_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        scale_labels = ['0%', '25%', '50%', '75%', '100%']
        
        for angle, label in zip(scale_angles, scale_labels):
            x = 1.2 * np.cos(angle)
            y = 1.2 * np.sin(angle)
            ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add tick marks
            x_tick_start = 1.05 * np.cos(angle)
            y_tick_start = 1.05 * np.sin(angle)
            x_tick_end = 1.15 * np.cos(angle)
            y_tick_end = 1.15 * np.sin(angle)
            ax.plot([x_tick_start, x_tick_end], [y_tick_start, y_tick_end], 
                   'k-', linewidth=2)
        
        # Customize plot
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.2, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"accuracy_gauge_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy gauge saved to: {save_path}")
        plt.show()
        
        return save_path

class ChatbotTrainer:
    def __init__(self, intents_file='intents.json'):
        self.nltk_utils = NLTKUtils()
        self.intents_file = intents_file
        self.model = None
        self.all_words = []
        self.tags = []
        self.visualizer = AccuracyVisualizer()

    def train(self, num_epochs=1000, batch_size=8, learning_rate=0.001, hidden_size=32, val_split=0.2, patience=50):
        """Train the chatbot model with validation and early stopping"""
        X, y, self.all_words, self.tags = self.nltk_utils.preprocess_data(self.intents_file)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42, stratify=y)

        input_size = len(X_train[0])
        output_size = len(self.tags)
        
        print(f"Training with input_size: {input_size}, output_size: {output_size}, hidden_size: {hidden_size}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Datasets and DataLoaders
        train_dataset = ChatbotDataset(X_train, y_train)
        val_dataset = ChatbotDataset(X_val, y_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model, loss, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        # Store predictions for confusion matrix
        all_val_preds = []
        all_val_true = []

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(torch.long).to(device)
                
                outputs = self.model(words)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, dim=1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            train_accuracy = 100 * correct_train / total_train
            train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for (words, labels) in val_loader:
                    words = words.to(device)
                    labels = labels.to(torch.long).to(device)
                    outputs = self.model(words)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, dim=1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
            
            val_accuracy = 100 * correct_val / total_val
            val_loss = total_val_loss / len(val_loader)
            
            # Update visualizer
            self.visualizer.update_metrics(epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy)
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
                # Store best predictions for confusion matrix
                all_val_preds = val_predictions
                all_val_true = val_true_labels
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    self.model.load_state_dict(best_model_state)
                    break

        # Final metrics
        print(f'Final Train Loss: {train_loss:.4f}, Final Train Accuracy: {train_accuracy:.2f}%')
        print(f'Final Val Loss: {val_loss:.4f}, Final Val Accuracy: {val_accuracy:.2f}%')

        # Generate visualizations
        print("\nGenerating training visualizations...")
        self.visualizer.plot_training_history()
        
        # Generate confusion matrix
        if all_val_preds and all_val_true:
            self.visualizer.plot_confusion_matrix(all_val_true, all_val_preds, self.tags)
        
        # Generate accuracy gauge
        final_val_accuracy = val_accuracy if 'val_accuracy' in locals() else 0
        self.visualizer.plot_accuracy_gauge(final_val_accuracy, "Final Validation Accuracy")
        
        # Generate classification report
        if all_val_preds and all_val_true:
            report = classification_report(all_val_true, all_val_preds, target_names=self.tags)
            print("\nClassification Report:")
            print(report)
            
            # Save classification report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.visualizer.results_dir, f"classification_report_{timestamp}.txt")
            with open(report_path, 'w') as f:
                f.write("BlinkWell Chatbot - Classification Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report)
            print(f"Classification report saved to: {report_path}")

    def save_model(self, filepath='blinkwell_chatbot_model.pth'):
        """Save the trained model and necessary data"""
        data = {
            "model_state": self.model.state_dict(),
            "input_size": len(self.all_words),
            "output_size": len(self.tags),
            "hidden_size": self.model.l1.out_features,
            "all_words": self.all_words,
            "tags": self.tags
        }
        torch.save(data, filepath)
        print(f"Model saved to {filepath}")

class BlinkWellChatbot:
    def __init__(self, model_path='blinkwell_chatbot_model.pth', intents_path='intents.json'):
        self.nltk_utils = NLTKUtils()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(intents_path, 'r') as f:
            self.intents = json.load(f)
        
        data = torch.load(model_path, map_location=self.device)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        
        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        
        print("BlinkWell Chatbot loaded successfully!")

    def get_response(self, user_input):
        """Get chatbot response for user input with confidence score"""
        if not user_input.strip():
            return "I'm sorry, I didn't understand that. Could you please ask me something about eye health?", 0.0
        
        sentence = self.nltk_utils.tokenize(user_input)
        X = self.nltk_utils.bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        confidence = prob.item()
        
        if confidence < 0.7:
            return "I'm not sure about that specific question. Could you please rephrase or ask about dry eyes, eye health tips, BlinkWell features, or prevention methods?", confidence
        
        for intent in self.intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"]), confidence
        
        return "I'm sorry, I couldn't understand your question. Please ask me about eye health, dry eyes, or BlinkWell features.", confidence

    def get_response_with_confidence(self, user_input):
        """Get response with confidence visualization"""
        response, confidence = self.get_response(user_input)
        
        # Create a simple confidence bar
        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        confidence_display = f"Confidence: [{confidence_bar}] {confidence*100:.1f}%"
        
        return response, confidence_display

if __name__ == "__main__":
    trainer = ChatbotTrainer('intents.json')
    print("Starting training...")
    trainer.train(num_epochs=1000, batch_size=8, learning_rate=0.001, hidden_size=32, val_split=0.2, patience=50)
    trainer.save_model('blinkwell_chatbot_model.pth')
    print("Training completed!")
    
    print("\n" + "="*50)
    print("Testing the chatbot...")
    print("="*50)
    
    chatbot = BlinkWellChatbot('blinkwell_chatbot_model.pth', 'intents.json')
    
    test_queries = [
        "Hello",
        "What are dry eye symptoms?",
        "How can I prevent dry eyes?",
        "What is BlinkWell?",
        "Thank you",
        "Goodbye"
    ]
    
    for query in test_queries:
        response, confidence_display = chatbot.get_response_with_confidence(query)
        print(f"User: {query}")
        print(f"Bot: {response}")
        print(f"{confidence_display}")
        print("-" * 30)