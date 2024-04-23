import cv2 as cv
import numpy as np
import uuid

class Storage:
    def __init__(self, path='storage', npz='data', images='images'):
        if not path.endswith('/'):
            path = path + '/'
        self.root_path = path

        if not npz.endswith('/'):
            npz = npz + '/'
        self.npz_path = self.root_path + npz

        if not images.endswith('/'):
            images = images + '/'
        self.images_path = self.root_path + images

    def generate_uuid_name(self) -> str:
        return str(uuid.uuid4())

    def save_image(self, image, filename=None, ext='.png'):
        if filename is None:
            filename = self.generate_uuid_name()
        file = str(filename) + ext
        resolved_path = self.images_path + file
        cv.imwrite(resolved_path, image)
        return resolved_path
    
    def save_npz(self, data, filename=None):
        if filename is None:
            filename = self.generate_uuid_name()
        file = str(filename) + '.npz'
        resolved_path = self.npz_path + file
        print('NPZ BEFORE')
        a, b = data
        # np.savez(resolved_path, data, allow_pickle=True)
        np.savez(resolved_path, a, b)
        print('RESOLVED_PATH NPZ', resolved_path)
        return resolved_path

    def load_npz(self, path):
        if not path.endswith('.npz'):
            path = path + '.npz'
        data = np.load(path, allow_pickle=True).values()
        return data

    def load_image(self, path, grayscale=True):
        image = cv.imread(path)
        if grayscale:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image

    def __str__(self):
        return "Storage: " + self.path