
import os
import argparse

import multiprocessing
import concurrent.futures

from datetime import datetime

# from generate_utils import generate_random_image, save_image_and_mask
from generate_utils import generate_random_image, save_image_and_mask
from static.constants import backgrounds, colors, fonts, texts, opacities, sizes


def generate_image_wrapper(fonts, texts, colors, backgrounds, opacities, sizes):
    return generate_random_image(fonts, texts, colors, backgrounds, opacities, sizes)


if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(
        description=
            '''
                Скрипт для создания и сохранения случайных изображений с использованием многопоточности.
                Выполнение функций скрипта происходит в двух созависимых pool-ах: генерации и сохранения.
                Т.е. генерация следующего chunk_size количества изображений начнётся только после их сохранения на диск.
                Это позволяет не заполнять оперативную память изображениями, которые ещё не успели сохраниться на диск.
            '''
    )

    parser.add_argument('--num_images', type=int, default=5000, help='Количество генерируемых изображений. По умолчанию: 5000')
    parser.add_argument('--chunk_size', type=int, default=100, help='Размер порции генерируемых изображений. По умолчанию: 100')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Количество используемых потоков CPU для генерации. По умолчанию: все доступные')
    parser.add_argument('--save_folder', type=str, default=os.path.join(BASE_DIR, 'syntdata'), help='Папка для сохранения изображений. По умолчанию: syntdata')
    parser.add_argument('--show_steps', type=int, default=500, help='Количество итераций, необходимых для вывода статуса выполнения. По умолчанию: 500')
    
    
    args = parser.parse_args()

    num_workers = args.num_workers
    num_images = args.num_images
    chunk_size = args.chunk_size
    show_steps = args.show_steps

    images_folder = os.path.join(args.save_folder, 'images')
    char_masks_folder = os.path.join(args.save_folder, 'character_heatmaps')
    aff_mask_folder = os.path.join(args.save_folder, 'affinity_heatmaps')

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(char_masks_folder, exist_ok=True)
    os.makedirs(aff_mask_folder, exist_ok=True)

    start_time = datetime.now()
    print(f'Начало выполнения генерации ({start_time.strftime('%Y-%m-%d %H:%M:%S')})')

    # Пул процессов для генерации
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as generate_pool:
        # Пул потоков для сохранения
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as save_pool:
            completed_global = 0

            # Идём по чанкам по chunk_size
            for chunk_start in range(0, num_images, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_images)
                current_chunk_size = chunk_end - chunk_start

                # Подготовим задачи на генерацию
                future_to_idx = {
                    generate_pool.submit(
                        generate_image_wrapper, fonts, texts, colors,
                        backgrounds, opacities, sizes
                    ): idx
                    for idx in range(chunk_start, chunk_end)
                }

                # Ждём их выполнения по мере готовности
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        image, mask = future.result()
                        # Сразу кидаем задачу на сохранение в пул потоков
                        save_pool.submit(save_image_and_mask, idx, image, mask,
                                         images_folder, char_masks_folder, aff_mask_folder)
                        completed_global += 1

                        # Выводим статус каждые 500 обработанных
                        if completed_global % show_steps == 0:
                            print(f'Обработано и отправлено на сохранение {completed_global} из {num_images}')
                    except Exception as e:
                        print(f'Ошибка при генерации изображения {idx}: {e}')

    stop_time = datetime.now()
    print(f'Окончание выполнения генерации ({stop_time.strftime('%Y-%m-%d %H:%M:%S')})')
    print(f'Изображений сгенерированно и сохранено: {completed_global}')
    print(f'Полный путь до изображений: {os.path.abspath(args.save_folder)}') 


