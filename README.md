# Gradient-accumulation
原始代码：
```
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

```
要在代码中实现梯度累积，首先需要在MyModel类中添加一个参数来指定你想要累积的步数。然后，在training_step方法中实现累积逻辑，仅在累积到指定的步数时执行一次优化器的步骤和梯度清零操作。以下是对你的代码进行的修改：  

1. 在__init__方法中添加一个accumulation_steps参数，并将其保存为一个超参数。  
2. 修改training_step方法以实现梯度累积逻辑。  
3. 你需要确保在累积足够的梯度后手动调用优化器的step方法和zero_grad方法。在PyTorch Lightning中，通常通过self.manual_backward来替代直接在损失上调用backward，并且在适当的时候手动调用self.optimizer.step和self.optimizer.zero_grad（这通常是在optimizer_step钩子中完成的，但在这里为简化我们直接在training_step中处理）。  
下面是修改后的代码：  
```
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
checkpoint_dir = './checkpoints/InflunzaA4_25d'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=checkpoint_dir,
    filename='InfluenzaA_2.5B_model-{epoch:02d}-{val_loss:.2f}',
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
torch.save(best_model.model.state_dict(), os.path.join(checkpoint_dir, "best_model_state_dict_2.5B_25day.pth"))

```
这里的重要变更是添加了accumulation_steps参数并在training_step中实现了梯度累积逻辑。请注意，这个示例假设你在configure_optimizers中创建并返回了一个优化器，并且我们在这里将其保存为类的一个属性，以便在training_step中引用。  

根据你的具体需求和硬件容量调整accumulation_steps的值。此外，确保当batch_idx + 1等于数据加载器长度时执行权重更新，这保证了每个epoch的最后一批次（可能小于accumulation_steps）也会被正确处理。  

这个脚本实现了梯度累积策略，允许在资源有限的情况下使用较小的batch_size同时模拟更大的有效batch_size。请根据实际需求和硬件配置调整accumulation_steps的值。注意，如果你的数据集或模型在执行过程中变化（例如，如果数据加载器的长度不是accumulation_steps的整数倍），你可能需要调整代码以确保最后一个小批次的梯度被正确处理。

## 以上代码出现问题：确保在模型中禁用自动优化，因为我计划使用手动反向传播
要在您的代码中禁用自动优化并使用梯度累积，您需要在MyModel类中进行一些关键的更改。具体来说，您需要设置self.automatic_optimization = False以禁用自动优化，并在training_step中手动管理梯度的反向传播和权重更新。以下是根据您提供的代码做出的修改。    
```
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

```
请注意，在此代码中，我在__init__方法中设置了self.automatic_optimization = False来禁用自动优化，并在training_step方法中手动进行了梯度累积和权重更新。此外，我在configure_optimizers方法中保存了优化器作为类的属性，以便在training_step中可以访问它。这样，您就可以控制梯度的累积和应用，实现更精细的训练控制。  
  
