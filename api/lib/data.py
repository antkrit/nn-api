import numpy as np


class Dataset:

    def __init__(
            self,
            x_data,
            y_data,
            batch_size=1,
            dim=(1, 1),
            shuffle=True,
    ):
        """Constructor method."""
        self.x = x_data
        self.y = y_data

        if len(self.x) != len(self.y):
            raise ValueError(
                f"Cannot match the lengths: {len(self.x)} and {len(self.x)}."
            )

        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.batch_size = min(batch_size, self.indexes.size)
        self.dim = dim

        self.__max = self.__len__()
        self.__inner_state = 0

    def __len__(self):
        return max(len(self.x) // self.batch_size, 1)

    def __getitem__(self, index):
        """Generate batch of data."""
        if self.__inner_state >= self.__max:
            raise IndexError(
                f"Index {index} out of range for {self.__max} batches."
            )

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__generate_data(indexes)

        return x, y

    def __iter__(self):
        self.__inner_state = 0
        return self

    def __next__(self):
        if self.__inner_state >= self.__max:
            self.on_epoch_end()
            self.__inner_state = 0

        result = self.__getitem__(self.__inner_state)

        self.__inner_state += 1
        return result

    def on_epoch_end(self):
        """Must be implemented."""
        # TODO: shuffle on epoch end
        return

    def __generate_data(self, ids):
        """Generate data containing batch_size samples."""
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *np.ones_like(self.dim)))

        # TODO: instead of getting data by index, load it with some method
        # this works only for arrays of data. Need to make this procedure lazy
        for i, id_ in enumerate(ids):
            x[i, ] = self.x[id_]
            y[i, ] = self.y[id_]

        return x, y
