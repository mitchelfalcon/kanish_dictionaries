import numpy as np
import hashlib

# Ring Configuration (Mersenne Prime 2^61 - 1)
PRIME_MOD = 2305843009213693951
MATRIX_DIM = 2

class GematriaVault:
    def __init__(self):
        # Map Oracc GDL operators to UTM(2, p) matrices
        self.operators = {
            '.': self._seed_matrix("OP_BESIDE"),
            'x': self._seed_matrix("OP_CONTAINING"),
            '+': self._seed_matrix("OP_JOINING"),
            '&': self._seed_matrix("OP_ABOVE")
        }

    def _seed_matrix(self, identifier):
        """Generates a deterministic upper triangular matrix."""
        h = int(hashlib.sha256(identifier.encode()).hexdigest(), 16)
        r12 = h % PRIME_MOD
        return np.array([[1, r12], dtype=object)

    def structural_hash(self, signs, operator='.'):
        """Calculates structural hash via non-commutative matrix product."""
        res = np.eye(MATRIX_DIM, dtype=object)
        op_mat = self.operators.get(operator, self.operators['.'])
        for i, sign in enumerate(signs):
            sign_mat = self._seed_matrix(sign)
            if i > 0:
                res = np.matmul(res, op_mat) % PRIME_MOD
            res = np.matmul(res, sign_mat) % PRIME_MOD
        return res
