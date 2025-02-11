from einops import rearrange
import math
import torch

__all__ = ["DCTTransform"]

class DCTTransform:

    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm="ortho"):
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    # Get the closest smallest divisor to the targeted DCT size
                    sc = self.__class__._get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc

                    # Pregenerate DCT basis matrices
                    if sc not in self.f_dict:
                        I = torch.eye(sc)
                        self.f_dict[sc] = self.__class__._dct(I, norm=norm).to(p.dtype).to(p.device)
                        self.b_dict[sc] = self.__class__._idct(I, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def _einsum_2d(x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def _einsum_2d_t(x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.__class__._einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.__class__._einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x):
        if len(x.shape) > 2:  # 2D weights
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w

            x = self.__class__._einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")

        else:  # 1D weights
            n1 = x.shape[1]
            n1w = self.b_dict[n1].to(x.device)
            self.b_dict[n1] = n1w

            x = self.__class__._einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x

    # Code modified and sourced from https://github.com/zh217/torch-dct
    def _dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def _idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

    @classmethod
    def _dct(cls, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = cls._dct_fft_impl(v)

        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == "ortho":
            V[:, 0] /= math.sqrt(N) * 2
            V[:, 1:] /= math.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    @classmethod
    def _idct(cls, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct(dct(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == "ortho":
            X_v[:, 0] *= math.sqrt(N) * 2
            X_v[:, 1:] *= math.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = cls._idct_irfft_impl(V)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, : N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, : N // 2]

        return x.view(*x_shape)

    def _get_prime_divisors(n):
        divisors = []
        while n % 2 == 0:
            divisors.append(2)
            n //= 2
        while n % 3 == 0:
            divisors.append(3)
            n //= 3
        i = 5
        while i * i <= n:
            for k in (i, i + 2):
                while n % k == 0:
                    divisors.append(k)
                    n //= k
            i += 6
        if n > 1:
            divisors.append(n)
        return divisors

    @classmethod
    def _get_divisors(cls, n):
        divisors = []
        if n == 1:
            divisors.append(1)
        elif n > 1:
            prime_factors = cls._get_prime_divisors(n)
            divisors = [1]
            last_prime = 0
            factor = 0
            slice_len = 0
            # Find all the products that are divisors of n
            for prime in prime_factors:
                if last_prime != prime:
                    slice_len = len(divisors)
                    factor = prime
                else:
                    factor *= prime
                for i in range(slice_len):
                    divisors.append(divisors[i] * factor)
                last_prime = prime
            divisors.sort()
        return divisors

    @classmethod
    def _get_smaller_split(cls, n, close_to):
        all_divisors = cls._get_divisors(n)
        for ix, val in enumerate(all_divisors):
            if val == close_to:
                return val
            if val > close_to:
                if ix == 0:
                    return val
                return all_divisors[ix - 1]
        return n
