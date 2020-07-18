import numpy as np

class SplitBodyParts():
    @classmethod
    def reorder_data(cls, x):
        # Separa os dados por parte do corpo
        X_trunk = np.concatenate((x[:, :, 15:18], x[:, :, 18:21], x[:, :, 24:27], x[:, :, 27:30]), axis=2)
        X_left_arm = np.concatenate(
            (x[:, :, 81:84], x[:, :, 87:90], x[:, :, 93:96], x[:, :, 99:102], x[:, :, 105:108], x[:, :, 111:114]),
            axis=2)
        X_right_arm = np.concatenate(
            (x[:, :, 84:87], x[:, :, 90:93], x[:, :, 96:99], x[:, :, 102:105], x[:, :, 108:111], x[:, :, 114:117]),
            axis=2)
        X_left_leg = np.concatenate((x[:, :, 33:36], x[:, :, 39:42], x[:, :, 45:48], x[:, :, 51:54], x[:, :, 57:60],
                                     x[:, :, 63:66], x[:, :, 69:72]), axis=2)
        X_right_leg = np.concatenate((x[:, :, 36:39], x[:, :, 42:45], x[:, :, 48:51], x[:, :, 54:57], x[:, :, 60:63],
                                      x[:, :, 66:69], x[:, :, 72:75]), axis=2)
        x_segmented = np.concatenate((X_trunk, X_right_arm, X_left_arm, X_right_leg, X_left_leg), axis=-1)
        return x_segmented
