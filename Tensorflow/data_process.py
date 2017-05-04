import numpy as np
import reader

class DataSet(object):

    def __init__(self, datas):
        # initialization

        self._datas = datas
        self._location = 0
        self._numbers = len(datas)

    def next_batch(self, batch_size):
        #get the batch data for trainning
        start = self._location
        self._location += batch_size
        if self._location > self._numbers:
            np.random.shuffle(self._datas)
            start = 0
            self._location = batch_size
        end = self._location

        this_batch = self._datas[start:end]
        return reader.get_data(this_batch)

    def save_small(self, img, name, num):
        print name
        exit()
    
    def make_small_block(self):
        pos = [0, 256, 512, 768, 1024]
        for batch in self.datas:
            for name in batch:
                for i in range(4):
                    for j in range(4):
                        img = np.load(name)
                        small_img = img[pos[i]: pos[i+1], pos[j]: pos[j+1]]
                        save_small(small_img, name, (4*i+j))

    

    @property
    def datas(self):
        return self._datas

    @property
    def location(self):
        return self._location

    @property
    def numbers(self):
        return self._numbers

#return the Class of datas
def do_it(datas):
    Data = DataSet(datas)
    return Data

if __name__ == '__main__':
    data = do_it(reader.load_data('1', '5'))
    data.make_small_block()
