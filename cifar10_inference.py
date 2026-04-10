from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BatchNormWeights:
    gamma: np.ndarray
    beta: np.ndarray
    moving_mean: np.ndarray
    moving_variance: np.ndarray


class CIFAR10ResNet:
    """Minimal NumPy implementation of the saved CIFAR-10 ResNet."""

    def __init__(
        self,
        weights_path: str | Path = "model.weights.h5",
        config_path: str | Path = "model_config.json",
        architecture_path: str | Path = "config.json",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)
        self.architecture_path = Path(architecture_path)

        with self.config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        self.class_names = list(config["classes"])
        self.mean = np.asarray(config["normalization"]["mean"], dtype=np.float32)
        self.std = np.asarray(config["normalization"]["std"], dtype=np.float32)

        with self.architecture_path.open("r", encoding="utf-8") as fh:
            architecture = json.load(fh)
        self.bn_epsilon = float(architecture["config"]["layers"][2]["config"]["epsilon"])

        self.conv_kernels, self.bn_layers, self.dense_kernel, self.dense_bias = (
            self._load_weights(self.weights_path)
        )

    @staticmethod
    def _layer_group_name(prefix: str, index: int) -> str:
        return prefix if index == 0 else f"{prefix}_{index}"

    @staticmethod
    def _sorted_weight_items(group: h5py.Group) -> list[np.ndarray]:
        vars_group = group["vars"]
        return [vars_group[key][()].astype(np.float32) for key in sorted(vars_group.keys(), key=int)]

    @staticmethod
    def _validate_weights_file(weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        with weights_path.open("rb") as fh:
            header = fh.read(128)

        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{weights_path} is a Git LFS pointer, not the real HDF5 model file. "
                "Railway needs the actual binary weights committed to git. "
                "Remove LFS tracking for model.weights.h5, re-add the file as a normal git blob, "
                "and redeploy."
            )

        if not header.startswith(b"\x89HDF\r\n\x1a\n"):
            raise RuntimeError(
                f"{weights_path} is not a valid HDF5 file. "
                "Make sure the actual model.weights.h5 binary is present in the deployed build."
            )

    def _load_weights(
        self, weights_path: Path
    ) -> tuple[list[np.ndarray], list[BatchNormWeights], np.ndarray, np.ndarray]:
        conv_kernels: list[np.ndarray] = []
        bn_layers: list[BatchNormWeights] = []

        self._validate_weights_file(weights_path)

        with h5py.File(weights_path, "r") as weights_file:
            layers_group = weights_file["layers"]

            for index in range(15):
                conv_group = layers_group[self._layer_group_name("conv2d", index)]
                conv_weights = self._sorted_weight_items(conv_group)
                if len(conv_weights) != 1:
                    raise ValueError(f"Unexpected conv weights in group {conv_group.name}")
                conv_kernels.append(conv_weights[0])

            for index in range(15):
                bn_group = layers_group[self._layer_group_name("batch_normalization", index)]
                bn_weights = self._sorted_weight_items(bn_group)
                if len(bn_weights) != 4:
                    raise ValueError(f"Unexpected BN weights in group {bn_group.name}")
                bn_layers.append(
                    BatchNormWeights(
                        gamma=bn_weights[0],
                        beta=bn_weights[1],
                        moving_mean=bn_weights[2],
                        moving_variance=bn_weights[3],
                    )
                )

            dense_group = layers_group["dense"]
            dense_weights = self._sorted_weight_items(dense_group)
            if len(dense_weights) != 2:
                raise ValueError(f"Unexpected dense weights in group {dense_group.name}")

        return conv_kernels, bn_layers, dense_weights[0], dense_weights[1]

    @staticmethod
    def _same_padding(in_size: int, kernel_size: int, stride: int) -> tuple[int, int]:
        out_size = int(np.ceil(in_size / stride))
        pad_total = max((out_size - 1) * stride + kernel_size - in_size, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after

    def _conv2d(self, x: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
        batch, height, width, in_channels = x.shape
        kernel_h, kernel_w, kernel_in, out_channels = kernel.shape
        if in_channels != kernel_in:
            raise ValueError(
                f"Channel mismatch: input has {in_channels}, kernel expects {kernel_in}"
            )

        pad_top, pad_bottom = self._same_padding(height, kernel_h, stride)
        pad_left, pad_right = self._same_padding(width, kernel_w, stride)
        padded = np.pad(
            x,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=(kernel_h, kernel_w), axis=(1, 2)
        )
        windows = windows[:, ::stride, ::stride, :, :, :]
        output = np.tensordot(windows, kernel, axes=([3, 4, 5], [2, 0, 1]))
        return output.astype(np.float32, copy=False)

    def _batch_norm(self, x: np.ndarray, weights: BatchNormWeights) -> np.ndarray:
        gamma = weights.gamma.reshape(1, 1, 1, -1)
        beta = weights.beta.reshape(1, 1, 1, -1)
        mean = weights.moving_mean.reshape(1, 1, 1, -1)
        variance = weights.moving_variance.reshape(1, 1, 1, -1)
        return gamma * (x - mean) / np.sqrt(variance + self.bn_epsilon) + beta

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _residual_block(
        self,
        x: np.ndarray,
        main_conv_1: int,
        main_bn_1: int,
        main_conv_2: int,
        main_bn_2: int,
        stride: int = 1,
        shortcut_conv: int | None = None,
        shortcut_bn: int | None = None,
    ) -> np.ndarray:
        residual = x
        y = self._conv2d(x, self.conv_kernels[main_conv_1], stride=stride)
        y = self._batch_norm(y, self.bn_layers[main_bn_1])
        y = self._relu(y)
        y = self._conv2d(y, self.conv_kernels[main_conv_2], stride=1)
        y = self._batch_norm(y, self.bn_layers[main_bn_2])

        if shortcut_conv is not None:
            residual = self._conv2d(residual, self.conv_kernels[shortcut_conv], stride=stride)
            residual = self._batch_norm(residual, self.bn_layers[shortcut_bn])

        return self._relu(y + residual)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((32, 32), Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return np.expand_dims(array, axis=0)

    def predict_pil(self, image: Image.Image) -> np.ndarray:
        batch = self.preprocess(image)
        return self.forward(batch)[0]

    def forward(self, batch: np.ndarray) -> np.ndarray:
        x = self._conv2d(batch, self.conv_kernels[0], stride=1)
        x = self._batch_norm(x, self.bn_layers[0])
        x = self._relu(x)

        x = self._residual_block(x, 1, 1, 2, 2)
        x = self._residual_block(x, 3, 3, 4, 4)
        x = self._residual_block(x, 5, 5, 6, 6, stride=2, shortcut_conv=7, shortcut_bn=7)
        x = self._residual_block(x, 8, 8, 9, 9)
        x = self._residual_block(x, 10, 10, 11, 11, stride=2, shortcut_conv=12, shortcut_bn=12)
        x = self._residual_block(x, 13, 13, 14, 14)

        x = x.mean(axis=(1, 2))
        logits = x @ self.dense_kernel + self.dense_bias
        return self._softmax(logits)

    def predict(self, image: Image.Image | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(image) as pil_image:
                return self.predict_pil(pil_image)
        return self.predict_pil(image)

    def top_k(self, image: Image.Image | str | Path, k: int = 5) -> list[tuple[str, float]]:
        probabilities = self.predict(image)
        order = np.argsort(probabilities)[::-1][:k]
        return [(self.class_names[index], float(probabilities[index])) for index in order]
