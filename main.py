import torch
import tiktoken
import torch.nn as nn

class Skew(nn.Module):
    def __init__(self, vocab_size=100277):
        super(Skew, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.fc1 = nn.Linear(32 * 64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 29)
    
    def forward(self, x):
        x = x.long()
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def predict(text, model_path="skew_model.pth"):
    enc = tiktoken.get_encoding("cl100k_base")
    model = Skew()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    tokens = enc.encode(text)
    if len(tokens) > 32:
        tokens = tokens[:32]
    else:
        tokens = tokens + [0] * (32 - len(tokens))
    
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
    
    responses = {
        1: "Hello. It's nice to meet you.",
        2: "That's great to hear!",
        3: "I'm sorry about that.",
        4: "Yes.",
        5: "No.",
        6: "Could you clarify?",
        7: "Tell me more.",
        8: "Sorry, but I can't solve math problems. Complex ones, at least.",
        9: "Interesting.",
        10: "My name is skew.",
        11: "And how does that make you feel?",
        12: "How is that relevant to our conversation?",
        13: "Well fuck you too then!",
        14: "Why?",
        15: "Are you scared of computers?",
        16: "No, I don't know Eliza.",
        17: "ChatGPT, my enemy. I agree.",
        18: "Alright.",
        19: "You don't have to be so negative.",
        20: "1.",
        21: "2.",
        22: "3.",
        23: "4.",
        24: "5.",
        25: "6.",
        26: "7.",
        27: "8.",
        28: "9.",
        0: "..."
    }
    
    return responses.get(prediction, "...")

if __name__ == "__main__":
    while True:
        print(predict(input(">> ")))