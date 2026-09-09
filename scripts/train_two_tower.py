"""Train and sanity-check the multimodal retrieval model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.models.two_tower import InfoNCELoss, TwoTowerNetwork
from src.dataset.dataloader import create_paired_dataloaders


class RetrievalModule(pl.LightningModule):
	def __init__(self, learning_rate: float = 1e-4, pretrained: bool = True) -> None:
		super().__init__()
		self.save_hyperparameters()
		self.model = TwoTowerNetwork(pretrained=pretrained)
		self.loss_fn = InfoNCELoss(temperature=0.07)

	def forward(self, sar_batch: Tensor, ms_batch: Tensor) -> tuple[Tensor, Tensor]:
		return self.model(sar_batch, ms_batch)

	def _step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
		sar_batch, ms_batch = batch[:2]
		sar_embeddings, ms_embeddings = self(sar_batch, ms_batch)
		loss = self.loss_fn(sar_embeddings, ms_embeddings)
		self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
		return self._step(batch, "train")

	def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
		return self._step(batch, "val")

	def configure_optimizers(self) -> dict:
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
		return {"optimizer": optimizer, "lr_scheduler": scheduler}


class RandomPairedDataset(Dataset):
	def __init__(self, size: int) -> None:
		self.sar = torch.randn(size, 2, 224, 224)
		self.ms = torch.randn(size, 3, 224, 224)

	def __len__(self) -> int:
		return len(self.sar)

	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		return self.sar[index], self.ms[index]


def run_random_sanity(samples: int = 200, batch_size: int = 64, epochs: int = 3) -> None:
	dataset = RandomPairedDataset(samples)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	model = RetrievalModule(pretrained=False)
	trainer = pl.Trainer(
		max_epochs=epochs,
		accelerator="auto",
		devices=1,
		logger=False,
		enable_checkpointing=False,
		enable_progress_bar=False,
	)
	trainer.fit(model, train_dataloaders=loader)
	print(f"Random sanity check complete: samples={samples}, epochs={epochs}")


def run_real_training(
	train_csv: Path,
	val_csv: Path,
	s1_root: Path,
	s2_root: Path,
	batch_size: int = 64,
	epochs: int = 20,
	learning_rate: float = 5e-5,
) -> None:
	train_loader, val_loader = create_paired_dataloaders(
		train_csv=train_csv,
		val_csv=val_csv,
		s1_root=s1_root,
		s2_root=s2_root,
		batch_size=batch_size,
		num_workers=0,
		require_s2=True,
	)

	model = RetrievalModule(learning_rate=learning_rate, pretrained=True)
	checkpoint = pl.callbacks.ModelCheckpoint(
		dirpath=PROJECT_ROOT / "outputs" / "models",
		filename="best_model-{epoch:02d}-{val_loss:.4f}",
		monitor="val_loss",
		mode="min",
		save_top_k=1,
	)
	logger = pl.loggers.CSVLogger(PROJECT_ROOT / "outputs" / "logs", name="two_tower")
	trainer = pl.Trainer(
		max_epochs=epochs,
		accelerator="auto",
		devices=1,
		gradient_clip_val=1.0,
		logger=logger,
		callbacks=[checkpoint],
	)
	trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
	print(f"Best checkpoint: {checkpoint.best_model_path}")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--samples", type=int, default=200)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--real-data", action="store_true")
	parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "outputs" / "metadata" / "train_split.csv")
	parser.add_argument("--val-csv", type=Path, default=PROJECT_ROOT / "outputs" / "metadata" / "val_split.csv")
	parser.add_argument("--s1-root", type=Path, default=PROJECT_ROOT / "data" / "s1_patches")
	parser.add_argument("--s2-root", type=Path, default=PROJECT_ROOT / "data" / "s2_val_patches")
	parser.add_argument("--learning-rate", type=float, default=5e-5)
	args = parser.parse_args()
	if args.real_data:
		run_real_training(
			train_csv=args.train_csv,
			val_csv=args.val_csv,
			s1_root=args.s1_root,
			s2_root=args.s2_root,
			batch_size=args.batch_size,
			epochs=args.epochs,
			learning_rate=args.learning_rate,
		)
	else:
		run_random_sanity(args.samples, args.batch_size, args.epochs)


if __name__ == "__main__":
	main()
