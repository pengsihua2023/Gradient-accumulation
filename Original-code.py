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
    def __init__(self, model, train_dataset, test_dataset, batch_size=2, lr=1e-5):
        super().__init__()
        # To save batch_size and lr as class attributes, ignoring 'model'
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)  # Adjust num_workers as needed

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8)  # Adjust num_workers as needed

# Assuming the model and dataset are already loaded and prepared as per your requirements
model = GenSLM("genslm_2.5B_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sequences = [str(record.seq) for record in SeqIO.parse("./data/H3N2_upto_2023_01_23.nucleotide.fasta", "fasta")]
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)  # Access seq_length and tokenizer directly from model

# Split the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize LightningModule with the consistent batch_size
my_model = MyModel(model, train_dataset, test_dataset, batch_size=2)  # Use consistent batch_size

# Define a checkpoint callback to save the best model
checkpoint_dir = './checkpoints/InflunzaA4_25d'  # Checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)  # Create directory if it doesn't exist

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=checkpoint_dir,
    filename='InfluenzaA_2.5B_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # Save only the best model
)

# Initialize the Trainer with the updated strategy parameter
trainer = Trainer(
    gpus=-1 if torch.cuda.is_available() else None,  # Use all available GPUs
    strategy='ddp',  # Updated to use 'strategy' instead of 'accelerator'
    max_epochs=3,
    callbacks=[checkpoint_callback],
    default_root_dir=checkpoint_dir,  # Set default root dir for logs and checkpoints
)

# Start training
trainer.fit(my_model)

# Load the best model checkpoint
best_model_path = checkpoint_callback.best_model_path
best_model = MyModel.load_from_checkpoint(best_model_path)

# Save the best model for inference
torch.save(best_model.model.state_dict(), os.path.join(checkpoint_dir, "best_model_state_dict_2.5B_25day.pth"))
