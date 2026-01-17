"""
Model architecture definitions.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, Dense, GRU, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
try:
    from .config import (
        LEARNING_RATE, CONV1D_FILTERS_1, CONV1D_FILTERS_2,
        CONV1D_KERNEL_SIZE, GRU_UNITS_1, GRU_UNITS_2,
        DENSE_UNITS, DROPOUT_RATE
    )
except ImportError:
    from config import (
        LEARNING_RATE, CONV1D_FILTERS_1, CONV1D_FILTERS_2,
        CONV1D_KERNEL_SIZE, GRU_UNITS_1, GRU_UNITS_2,
        DENSE_UNITS, DROPOUT_RATE
    )


def create_phys_informed_model(input_shape):
    """
    CNN + GRU model for 4-channel physiological input with BatchNorm and Dropout.
    """
    model = Sequential(
        [
            Conv1D(
                CONV1D_FILTERS_1, CONV1D_KERNEL_SIZE, 
                activation="relu", input_shape=input_shape, padding="same"
            ),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(DROPOUT_RATE),
            Conv1D(CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(DROPOUT_RATE),
            GRU(GRU_UNITS_1, return_sequences=True),
            Dropout(DROPOUT_RATE),
            GRU(GRU_UNITS_2),
            Dropout(DROPOUT_RATE),
            Dense(DENSE_UNITS, activation="relu"),
            Dense(1),
        ]
    )

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=Huber(), metrics=["mae"])
    return model
