import os
import argparse

from datetime import datetime

from torch.nn import MSELoss

from train_utils import SegformerForСraft, compute_metrics, seed_everything, save_training_results

from transformers import SegformerImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback

from sklearn.model_selection import train_test_split
from train_utils import CRAFTDataset, craft_data_collator, collect_data


if __name__ == '__main__':

	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	# Парсим аргументы командной строки
	parser = argparse.ArgumentParser(
		description='Скрипт для finetuning предобученной сегментационной модели на синтетических данных'
	)
	
	parser.add_argument('--input_dir', type=str, default=os.path.join(BASE_DIR, 'syntdata'), help='Директория с синтетическими изображениями')
	parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, 'checkpoints'), help='Директория сохранения результатов обучения')
	parser.add_argument('--model_name', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512', help='Имя предобученной модели')
	parser.add_argument('--num_train_epochs', type=int, default=10, help='Количество эпох обучения')
	parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
	parser.add_argument('--learning_rate', type=float, default=5e-5)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--test_size', type=float, default=0.2, help='Размер тестовой части датасета')
	parser.add_argument('--caching', type=bool, default=False, help='Если True, то все данные датасета кешируются в RAM')

	args = parser.parse_args()

	model_name = args.model_name
	input_dir = args.input_dir
	caching = args.caching

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		eval_strategy="epoch",
		save_strategy="epoch",
		logging_steps=50,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		eval_accumulation_steps=50,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		load_best_model_at_end=True
	)

	seed_everything(seed=37)

	feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
	model = SegformerForСraft.from_pretrained(
		model_name, num_labels=2, ignore_mismatched_sizes=True, loss_fn=MSELoss()
	)

	dataset = collect_data(input_dir)
	train, test = train_test_split(dataset, test_size=0.2, random_state=37)
	train, test = train.reset_index(drop=True), test.reset_index(drop=True)

	if caching:
		start_time = datetime.now()
		print(f'Начало кеширования датасетов в RAM ({start_time.strftime('%Y-%m-%d %H:%M:%S')})')

	train_dataset = CRAFTDataset(feature_extractor, train, caching=caching)
	test_dataset = CRAFTDataset(feature_extractor, test, caching=caching)

	if caching:
		stop_time = datetime.now()
		print(f'Кеширование датасетов завершено ({stop_time.strftime('%Y-%m-%d %H:%M:%S')})')

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		data_collator=craft_data_collator,
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(
			early_stopping_patience=5,
			early_stopping_threshold=0.0
		)]
	)

	start_time = datetime.now()
	print(f'Старт обучения ({start_time.strftime('%Y-%m-%d %H:%M:%S')})')
	
	trainer.train()
	save_training_results(trainer, BASE_DIR)

	stop_time = datetime.now()
	print(f'Обучение завершено ({stop_time.strftime('%Y-%m-%d %H:%M:%S')})')



