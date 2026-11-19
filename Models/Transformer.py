import tensorflow as tf
from tensorflow.keras import layers as L, Model

class PositionalEmbedding2D(L.Layer):
    """Learned 2D positional embeddings for (days, slots)."""
    def __init__(self, days=7, slots=96, d_model=192, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.slots = slots
        self.d_model = d_model

    def build(self, input_shape):
        self.day_emb  = self.add_weight(
            name="day_emb",
            shape=(self.days, self.d_model),
            initializer="random_normal",
            trainable=True
        )
        self.slot_emb = self.add_weight(
            name="slot_emb",
            shape=(self.slots, self.d_model),
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        # x: (B, days, slots, d_model)
        day = self.day_emb[tf.newaxis, :, tf.newaxis, :]   # (1, days, 1, d)
        slot = self.slot_emb[tf.newaxis, tf.newaxis, :, :] # (1, 1, slots, d)
        return x + day + slot

class TransformerCNN:
    def __init__(self, config):
        self.config = config

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = L.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = L.Dropout(dropout)(x)
        x = L.LayerNormalization(epsilon=1e-6)(x)
        res = L.Add()([inputs, x])

        # Feed Forward Network
        x = L.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = L.Dropout(dropout)(x)
        x = L.Conv1D(filters=int(inputs.shape[-1]), kernel_size=1)(x)
        x = L.LayerNormalization(epsilon=1e-6)(x)
        return L.Add()([res, x])

    def build_model(self,
                    input_shape,
                    d_model=192,
                    head_size=128,
                    num_heads=4,
                    ff_dim=256,
                    num_transformer_blocks=4,
                    mlp_units=(256,),
                    n_classes=2,
                    dropout=0.5,
                    mlp_dropout=0.5,
                    ):

        days, slots, _ = input_shape
        T = days * slots

        inputs = L.Input(shape=input_shape)

        x = L.Conv2D(d_model, kernel_size=1, use_bias=False)(inputs)
        x = PositionalEmbedding2D(days=days, slots=slots, d_model=d_model)(x)
        x = L.Reshape((T, d_model))(x)

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = L.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = L.Dense(dim, activation="relu")(x)
            x = L.Dropout(mlp_dropout)(x)
        outputs = L.Dense(n_classes, activation="softmax")(x)
        return Model(inputs, outputs, name="Transformer2DPE")


class TransformerKeras:
    def __init__(self, config):
        self.config = config

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.0, name_prefix="encoder"):
        # Pre-LN
        x_norm = L.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(inputs)

        # Self-attention
        attn_out = L.MultiHeadAttention(key_dim=head_size,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        name=f"{name_prefix}_mha"
                                        )(x_norm, x_norm)
        # Attention dropout
        attn_out = L.Dropout(dropout, name=f"{name_prefix}_attn_dropout1")(attn_out)

        # x = L.LayerNormalization(epsilon=1e-6)(x)

        # Residual
        x = L.Add(name=f"{name_prefix}_attn_residual")([inputs, attn_out])

        # FFN (Pre-LN style)
        y_norm = L.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x)
        y = L.Dense(ff_dim, activation=tf.nn.gelu, name=f"{name_prefix}_ffn_dense1")(y_norm)
        y = L.Dropout(dropout, name=f"{name_prefix}_ffn_dropout1")(y)
        y = L.Dense(int(x.shape[-1]), name=f"{name_prefix}_ffn_dense2")(y)
        y = L.Dropout(dropout, name=f"{name_prefix}_ffn_dropout2")(y)

        # Residual
        return L.Add(name=f"{name_prefix}_ffn_residual")([x, y])

    def build_model(
            self,
            input_shape,                 # (days=7, slots=96, C)
            d_model=192,                 # model/embedding dimension
            head_size=128,
            num_heads=4,
            ff_dim=256,
            num_transformer_blocks=4,
            mlp_units=(256,),
            n_classes=1,
            dropout=0.2,
            mlp_dropout=0.3,
    ):
        days, slots, _ = input_shape
        inputs = L.Input(shape=input_shape) # (T, D)

        # Project to model dim (token embedding per grid cell)
        x = L.Dense(d_model, use_bias=False)(inputs)

        # 2D positional embedding (day + slot)
        x = PositionalEmbedding2D(days=days, slots=slots, d_model=d_model)(x)

        # Flatten grid to sequence of tokens
        x = L.Reshape((days * slots, d_model))(x)       # (B, 672, d_model)

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        # Head
        x = L.LayerNormalization(epsilon=1e-6)(x)
        x = L.GlobalAveragePooling1D()(x)  # pool over tokens
        for dim in mlp_units:
            x = L.Dense(dim, activation=tf.nn.gelu)(x)
            x = L.Dropout(mlp_dropout)(x)

        outputs = L.Dense(n_classes, activation="softmax")(x)  # multi-class

        return Model(inputs, outputs, name="Transformer2DPE")

    @staticmethod
    def build_model_with_aux(days=7,
                             measurements=96,
                             n_classes=2,
                             aux_logits=3,
                             dropout_rate=0.5,
                             d_model=128,
                             head_size=128,
                             num_heads=4,
                             ff_dim=128,
                             num_transformer_blocks=4,
                             mlp_units=(128,),
                             mlp_dropout=0.3,
                             model_name="Transformer2DPE_with_aux"):


        inputs = L.Input(shape=(days, measurements, 1), name="Input")   # (T, D)

        # Project to model dim (token embedding per grid cell)
        x = L.Dense(d_model, use_bias=False,  name="TokenProjection")(inputs)

        # 2D positional embedding (day + slot)
        x = PositionalEmbedding2D(days=days, slots=measurements, d_model=d_model, name="PosEmbedding2D")(x)

        # Flatten grid to sequence of tokens
        x = L.Reshape((days * measurements, d_model), name="FlattenToTokens")(x)   # (B, 672, d_model)

        # Transformer blocks
        for b in range(num_transformer_blocks):
            x = TransformerKeras(None).transformer_encoder(
                x, head_size, num_heads, ff_dim, dropout_rate,
                name_prefix=f"Block{b + 1}"
            )

        # Pooling head (representation)
        x = L.LayerNormalization(epsilon=1e-6, name="HeadLN")(x)
        x = L.GlobalAveragePooling1D(name="GAP")(x)

        # MLP head -> representation vector ("D1" for compatibility)
        for i, dim in enumerate(mlp_units):
            x = L.Dense(dim, activation=tf.nn.gelu, name=f"MLP{i+1}")(x)
            x = L.Dropout(mlp_dropout, name=f"MLP{i+1}_dropout")(x)

        # This is the shared representation (analogous to CNN's D1)
        rep = L.Dense(mlp_units[-1], activation=tf.nn.gelu, name="D1")(x)

        # Real class logits (NO softmax here)
        real = L.Dense(n_classes, activation=None, name="real_logits")(rep)

        # Aux logits (NO softmax here)
        aux = L.Dense(aux_logits, activation=None, name="aux_logit")(rep)
        model = Model(inputs=inputs, outputs=[real, aux], name=model_name)
        return model