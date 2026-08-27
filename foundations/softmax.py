import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sm = []
        maxx = max(z)
        for num in z:
            sm.append(np.round(np.exp(num - maxx), 4))
        summ = np.round(sum(sm),4)
        for i in range(len(sm)):
            sm[i] = np.round(sm[i]/summ,4)
        softmax: NDArray[np.float64] = np.array(sm, dtype = np.float64)
        return softmax
