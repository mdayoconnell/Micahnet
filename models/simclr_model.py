# micahnet_simclr_model.py
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, Model


@dataclass
class MicahNetConfig:
    # Input / model size
    input_shape: tuple = (100, 100, 1)   # grayscale
    width_mult: float = 1.0              # 0.75, 1.0, 1.25 are useful presets
    embedding_dim: int = 256             # backbone output embedding
    proj_dim: int = 128                  # SimCLR projection output
    proj_hidden_dim: int = 512           # hidden size in projection head

    # Regularization
    dropout_rate: float = 0.2
    weight_decay: float = 1e-5           # optional use in optimizer


def _c(channels: int, width_mult: float) -> int:
    """Scale channels by width multiplier while keeping a minimum."""
    return max(8, int(channels * width_mult))


class ConvBNAct(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same", name=None):
        super().__init__(name=name)
        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)


class MicahNetBackbone(Model):
    """
    AlexNet-inspired but lightweight/modernized:
    - 5 conv stages
    - BatchNorm + ReLU
    - Max pooling in early/mid stages
    - Global average pooling instead of giant FC stack
    """
    def __init__(self, cfg: MicahNetConfig, name="micahnet_backbone"):
        super().__init__(name=name)
        wm = cfg.width_mult

        # Stage 1 (larger receptive field early)
        self.block1 = ConvBNAct(_c(48, wm), kernel_size=7, strides=2, padding="same", name="block1")
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Stage 2
        self.block2 = ConvBNAct(_c(96, wm), kernel_size=5, strides=1, padding="same", name="block2")
        self.pool2 = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Stage 3-5 (3x3 stack)
        self.block3 = ConvBNAct(_c(192, wm), kernel_size=3, strides=1, padding="same", name="block3")
        self.block4 = ConvBNAct(_c(192, wm), kernel_size=3, strides=1, padding="same", name="block4")
        self.block5 = ConvBNAct(_c(128, wm), kernel_size=3, strides=1, padding="same", name="block5")
        self.pool5 = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Head
        self.gap = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(cfg.dropout_rate)
        self.fc = layers.Dense(cfg.embedding_dim, kernel_initializer="he_normal", name="embed_fc")
        self.embed_bn = layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.block1(x, training=training)
        x = self.pool1(x)
        x = self.block2(x, training=training)
        x = self.pool2(x)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.pool5(x)

        x = self.gap(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        x = self.embed_bn(x, training=training)  # pre-normalized embedding
        return x


class ProjectionHead(Model):
    """
    SimCLR projection head:
    z = Dense -> BN -> ReLU -> Dense -> (L2 norm outside)
    """
    def __init__(self, cfg: MicahNetConfig, name="projection_head"):
        super().__init__(name=name)
        self.dense1 = layers.Dense(cfg.proj_hidden_dim, use_bias=False, kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        self.dense2 = layers.Dense(cfg.proj_dim, use_bias=True, kernel_initializer="he_normal")

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dense2(x)
        return x


class SimCLRModel(Model):
    """
    Returns:
      h: normalized backbone embedding
      z: normalized projection embedding (used in NT-Xent loss)
    """
    def __init__(self, cfg: MicahNetConfig, name="simclr_model"):
        super().__init__(name=name)
        self.encoder = MicahNetBackbone(cfg)
        self.projector = ProjectionHead(cfg)

    def call(self, x, training=False):
        h = self.encoder(x, training=training)
        z = self.projector(h, training=training)

        # Normalize for cosine similarity / contrastive loss
        h = tf.math.l2_normalize(h, axis=-1)
        z = tf.math.l2_normalize(z, axis=-1)
        return h, z


def build_model(cfg: MicahNetConfig) -> SimCLRModel:
    model = SimCLRModel(cfg)
    # Build by running a dummy batch
    dummy = tf.zeros((2, *cfg.input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


if __name__ == "__main__":
    cfg = MicahNetConfig(
        input_shape=(100, 100, 1),
        width_mult=1.0,
        embedding_dim=256,
        proj_dim=128,
        proj_hidden_dim=512,
    )
    model = build_model(cfg)

    x = tf.random.uniform((8, 100, 100, 1))
    h, z = model(x, training=True)

    print("Backbone embedding h shape:", h.shape)   # (8, 256)
    print("Projection z shape:", z.shape)           # (8, 128)

    # Parameter count sanity check
    model.summary()