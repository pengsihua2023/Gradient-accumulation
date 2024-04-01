import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os

class MyModel(LightningModule):
    def __init__(self, model, train_dataset, test_dataset, batch_size=2, lr=1e-5, accumulation_steps=10):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, except 'model'
        self.automatic_optimization = False  # 禁用自动优化
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.accumulation_steps = accumulation_steps  # New attribute for gradient accumulation steps

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Gradient accumulation
        loss = loss / self.accumulation_steps  # Scale loss
        self.manual_backward(loss)  # Backward pass (accumulates gradients)
        
        # Condition to check if it is time to update weights
        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader()):
            self.optimizer.step()  # Update weights
            self.optimizer.zero_grad()  # Clear gradients
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)  # Save optimizer as attribute
        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8)

# Assuming the model and dataset are already loaded and prepared as per your requirements
model = GenSLM("genslm_2.5B_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sequences = [str(record.seq) for record in SeqIO.parse("./data/H3N2_upto_2023_01_23.nucleotide.fasta", "fasta")]
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)

# Split the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize LightningModule with the consistent batch_size and accumulation_steps
my_model = MyModel(model, train_dataset, test_dataset, batch_size=2, accumulation_steps=10)

# Define a checkpoint callback to save the best model
checkpoint_dir = './checkpoints/InflunzaA_gradient'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=checkpoint_dir,
    filename='InfluenzaA_2.5B_model_Gradient_accum-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
)

# Initialize the Trainer
trainer = Trainer(
    gpus=-1 if torch.cuda.is_available() else None,
    strategy='ddp',
    max_epochs=3,
    callbacks=[checkpoint_callback],
    default_root_dir=checkpoint_dir,
)

# Start training
trainer.fit(my_model)

# Load the best model checkpoint
best_model_path = checkpoint_callback.best_model_path
best_model = MyModel.load_from_checkpoint(best_model_path)

# Save the best model for inference
torch.save(best_model.model.state_dict(), os.path.join(checkpoint_dir, "Model_2.5B_gradient-accum.pth"))
