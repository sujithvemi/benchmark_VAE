import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from pythae.models import AutoModel
from pythae.samplers import NormalSampler
import os
import numpy as np


class MyHandler(FileSystemEventHandler):
    def __init__(self):
        self.counter = 0
    def on_modified(self, event):
        self.counter += 1
        print(f'event type: {event.event_type}  path : {event.src_path}')
        last_training = sorted(os.listdir('/home/ubuntu/projects/multimodal_fov_robustness_test/benchmark_VAE/examples/notebooks/conv_vae_new_dSprites_v2'))[-1]
        print(last_training)
        os.makedirs("/home/ubuntu/projects/multimodal_fov_robustness_test/benchmark_VAE/conv_vae_new_dSprites_v2_gen_samples/epoch_{}".format(self.counter), exist_ok=True)
        model_rec = AutoModel.load_from_folder(os.path.join('conv_vae_new_dSprites_v1', last_training, 'final_model'))
        sampler = NormalSampler(
            model=model_rec
        )
        gen_data = sampler.sample(
            num_samples=25
        )
        for i in range(5):
            for j in range(5):
                img = gen_data[i*5 +j].detach().cpu().reshape(256, 256).numpy()
                img = img.astype(int)
                if np.max(img) < 255:
                    img = img*255
                cv2.imwrite("/home/ubuntu/projects/multimodal_fov_robustness_test/benchmark_VAE/conv_vae_new_dSprites_v2_gen_samples/epoch_{}/sample_{}_{}.png".format(self.counter, i, j), img)


if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='/home/ubuntu/projects/multimodal_fov_robustness_test/benchmark_VAE/examples/notebooks/conv_vae_new_dSprites_v2/{}'.format(sorted(os.listdir('/home/ubuntu/projects/multimodal_fov_robustness_test/benchmark_VAE/examples/notebooks/conv_vae_new_dSprites_v2'))[-1]), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()