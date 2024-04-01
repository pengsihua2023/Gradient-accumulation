import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import os

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')

class MyModel(LightningModule):
    def __init__(self, model=None, train_dataset=None, test_dataset=None, batch_size=8, lr=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False  # 禁用自动优化
        if model is not None:
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
        self.manual_backward(loss)  # 使用 manual_backward 进行反向传播
        return loss

    def training_epoch_end(self, outputs):
        optimizer = self.optimizers()  # 获取优化器
        optimizer.step()  # 执行优化器的步骤
        optimizer.zero_grad()  # 清零梯度

    def validation_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def on_load_checkpoint(self, checkpoint):
        # Reinitialize the model here
        model_cache_dir = "/scratch/sp96859/GenSLM"
        self.model = GenSLM("genslm_250M_patric", model_cache_dir=model_cache_dir)
        self.model.train()

# 以下代码假定模型和数据集已按照您的要求加载和准备
model = GenSLM("genslm_250M_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sequences = [str(record.seq) for record in SeqIO.parse("./data/H3N2_upto_2023_01_23.nucleotide.fasta", "fasta")]
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)

# 分割数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 用一致的batch_size初始化LightningModule
my_model = MyModel(model, train_dataset, test_dataset, batch_size=8)

# 定义一个checkpoint回调来保存最佳模型
checkpoint_dir = './checkpoints/InflunzaA5'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=checkpoint_dir,
    filename='InfluenzaA_250M_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
)

# 使用更新后的策略参数初始化训练器
trainer = Trainer(
    gpus=-1 if torch.cuda.is_available() else None,
    strategy=DDPPlugin(find_unused_parameters=False),
    max_epochs=3,
    callbacks=[checkpoint_callback],
    default_root_dir=checkpoint_dir,
)

# 开始训练
trainer.fit(my_model)

# 加载最佳模型checkpoint
best_model_path = checkpoint_callback.best_model_path
if best_model_path:
    best_model = MyModel.load_from_checkpoint(best_model_path)

    # 保存最佳模型以进行推理
    torch.save(best_model.model.state_dict(), os.path.join(checkpoint_dir, "best_model_state_dict_new.pth"))

