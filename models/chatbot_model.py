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

class ChatbotTrainer:
    def __init__(self, intents_file='intents.json'):
        self.nltk_utils = NLTKUtils()
        self.intents_file = intents_file
        self.model = None
        self.all_words = []
        self.tags = []

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
            
            val_accuracy = 100 * correct_val / total_val
            val_loss = total_val_loss / len(val_loader)
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    self.model.load_state_dict(best_model_state)
                    break

        # Final metrics
        print(f'Final Train Loss: {train_loss:.4f}, Final Train Accuracy: {train_accuracy:.2f}%')
        print(f'Final Val Loss: {val_loss:.4f}, Final Val Accuracy: {val_accuracy:.2f}%')

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
        """Get chatbot response for user input"""
        if not user_input.strip():
            return "I'm sorry, I didn't understand that. Could you please ask me something about eye health?"
        
        sentence = self.nltk_utils.tokenize(user_input)
        X = self.nltk_utils.bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        if prob.item() < 0.7:
            return "I'm not sure about that specific question. Could you please rephrase or ask about dry eyes, eye health tips, BlinkWell features, or prevention methods?"
        
        for intent in self.intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
        
        return "I'm sorry, I couldn't understand your question. Please ask me about eye health, dry eyes, or BlinkWell features."

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
        response = chatbot.get_response(query)
        print(f"User: {query}")
        print(f"Bot: {response}")
        print("-" * 30)