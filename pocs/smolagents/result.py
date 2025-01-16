import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

# Define ASCII art representations for digits 0-9
ascii_digits = {
    0: [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    1: [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    2: [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    3: [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    4: [
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
    ],
    5: [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    6: [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    7: [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ],
    8: [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    9: [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
}


# Create dataset
class AsciiDigitDataset(Dataset):
    def __init__(self, augment=True):
        self.digits = []
        self.art = []

        # Add original patterns
        for digit, pattern in ascii_digits.items():
            self.digits.append(digit)
            self.art.append(pattern)

            if augment:
                # Add noise-augmented versions
                for _ in range(5):  # 5 noisy versions per digit
                    self.digits.append(digit)
                    noisy_pattern = self.add_noise(pattern, noise_prob=0.1)
                    self.art.append(noisy_pattern)

                # Add shifted versions
                for dx, dy in [
                    (0, 1),
                    (1, 0),
                    (-1, 0),
                    (0, -1),
                ]:  # 4 shifted versions per digit
                    self.digits.append(digit)
                    shifted_pattern = self.shift_pattern(pattern, dx, dy)
                    self.art.append(shifted_pattern)

    def add_noise(self, pattern, noise_prob=0.1):
        pattern = np.array(pattern)
        noise_mask = np.random.random(pattern.shape) < noise_prob
        pattern = pattern.copy()
        pattern[noise_mask] = 1 - pattern[noise_mask]  # Flip bits randomly
        return pattern.tolist()

    def shift_pattern(self, pattern, dx, dy):
        pattern = np.array(pattern)
        shifted = np.zeros_like(pattern)

        # Calculate source and target indices
        x1, x2 = max(0, dx), min(8, 8 + dx)
        y1, y2 = max(0, dy), min(8, 8 + dy)
        src_x1, src_x2 = max(0, -dx), min(8, 8 - dx)
        src_y1, src_y2 = max(0, -dy), min(8, 8 - dy)

        shifted[x1:x2, y1:y2] = pattern[src_x1:src_x2, src_y1:src_y2]
        return shifted.tolist()

    def __len__(self):
        return len(self.digits)

    def __getitem__(self, idx):
        digit = self.digits[idx]
        art = self.art[idx]
        return torch.tensor(digit, dtype=torch.float32), torch.tensor(
            art, dtype=torch.float32
        )


# Define the neural network
class AsciiArtGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Embedding layer to give each digit a unique, learnable representation
        self.embedding = nn.Embedding(10, 32)

        # Encoder with wider layers
        self.encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Final projection with stronger binary push
        self.final = nn.Sequential(
            nn.Linear(64, 64), nn.Tanh()  # Push values towards -1 and 1
        )

    def forward(self, x):
        # Convert input to long for embedding
        x = x.long().view(-1)
        # Get learned embedding for each digit
        x = self.embedding(x)
        # Encoder
        encoded = self.encoder(x)
        # Decoder with residual connections
        decoded = self.decoder(encoded)
        # Final projection and reshape to 8x8
        output = self.final(decoded)
        # Convert from [-1,1] to [0,1] range
        output = (output + 1) / 2
        return output.view(-1, 8, 8)


# Constants
NUM_EPOCHS = 2000  # Fewer epochs since we have more data
BATCH_SIZE = 4  # Smaller batch size
LEARNING_RATE = 0.0001  # Lower learning rate for stability

# Create dataset and dataloaders with augmentation
dataset = AsciiDigitDataset(augment=True)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function, and optimizer
model = AsciiArtGenerator()
criterion = nn.BCELoss()  # Back to BCE loss since we want binary outputs
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)

# Use simpler scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=100, verbose=True
)

# Training loop with validation
best_val_loss = float("inf")
train_losses = []
val_losses = []
patience = 100  # Early stopping patience
no_improve = 0

# Progress bar for epochs
epoch_bar = trange(NUM_EPOCHS, desc="Training")
for epoch in epoch_bar:
    # Training
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for digits, arts in train_bar:
        optimizer.zero_grad()
        outputs = model(digits)
        loss = criterion(outputs, arts)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        for digits, arts in val_bar:
            outputs = model(digits)
            batch_loss = criterion(outputs, arts).item()
            val_loss += batch_loss
            val_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Save best model and check early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_ascii_generator.pth")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("\nEarly stopping triggered!")
            break

    # Update progress bar with metrics
    epoch_bar.set_postfix(
        {
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "lr": f'{optimizer.param_groups[0]["lr"]:.6f}',
        }
    )

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")


# Test function with threshold tuning
def generate_ascii_art(digit, threshold=0.5):
    model.eval()
    with torch.no_grad():
        digit_tensor = torch.tensor([[digit]], dtype=torch.float32)
        output = model(digit_tensor)
        binary_output = (output > threshold).float()
        return binary_output.squeeze(0).numpy()


def display_ascii(art_matrix):
    """Display ASCII art using block characters for better visibility."""
    for row in art_matrix:
        print("".join(["██" if cell else "  " for cell in row]))


# Validation visualization
def validate_all_digits():
    print("\nValidating all digits:")
    for digit in range(10):
        print(f"\nDigit {digit}:")
        art = generate_ascii_art(digit)
        display_ascii(art)
        print("-" * 20)


# Interactive testing with adjustable threshold
while True:
    try:
        digit = input("Enter a digit (0-9) or 'v' for validation, 'q' to quit: ")
        if digit.lower() == "q":
            break
        elif digit.lower() == "v":
            validate_all_digits()
        else:
            threshold = float(input("Enter threshold (0.0-1.0, default 0.5): ") or 0.5)
            display_ascii(generate_ascii_art(int(digit), threshold))
    except ValueError:
        print("Invalid input. Please try again.")
