"""
Physiology-informed CNN-LSTM model with PAT-based attention mechanism.

Architecture:
1. Multi-channel input: [ECG, PPG, PAT, HR]
2. CNN layers extract local morphology features from all channels
3. PAT channel generates temporal attention weights
4. Attention weights reweight CNN features before LSTM
5. LSTM layers model cardiovascular dynamics
6. Dense layers regress blood pressure
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Dense, LSTM, 
    BatchNormalization, Multiply, Softmax, Lambda, Reshape,
    GlobalAveragePooling1D, Concatenate, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, Loss
import tensorflow.keras.backend as K

try:
    from .config import (
        LEARNING_RATE, GRADIENT_CLIP_NORM,
        CONV1D_FILTERS_1, CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE,
        LSTM_UNITS_1, LSTM_UNITS_2, ATTENTION_UNITS,
        DENSE_UNITS, DROPOUT_RATE
    )
except ImportError:
    from config import (
        LEARNING_RATE, GRADIENT_CLIP_NORM,
        CONV1D_FILTERS_1, CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE,
        LSTM_UNITS_1, LSTM_UNITS_2, ATTENTION_UNITS,
        DENSE_UNITS, DROPOUT_RATE
    )

# L2 Regularization strength (increased to prevent test set overfitting)
L2_REG = 0.01  # Stronger regularization: prevents overfitting by penalizing large weights


class WeightedHuberLoss(Loss):
    """
    Huber loss with BP-dependent sample weighting.
    
    Penalizes errors on extreme BP values more heavily than normal BP.
    This addresses data imbalance where 76.8% of training samples are
    normal BP (90-140 mmHg), causing the model to learn mean prediction.
    
    Weighting strategy:
    - Normal BP (90-140): weight = 1.0 (baseline)
    - High BP (>140): weight increases linearly up to 3.0
    - Low BP (<90): weight increases linearly up to 2.0
    
    This encourages the model to fit extreme values better.
    """
    
    def __init__(self, delta=1.0, normal_bp_range=(90, 140), high_weight=3.0, low_weight=2.0, name='weighted_huber_loss'):
        """
        Args:
            delta: Huber loss delta parameter (threshold for quadratic vs linear)
            normal_bp_range: Tuple (low, high) defining normal BP range in mmHg
            high_weight: Maximum weight for high BP values (>normal_bp_range[1])
            low_weight: Maximum weight for low BP values (<normal_bp_range[0])
            name: Loss function name
        """
        super().__init__(name=name)
        self.delta = delta
        self.normal_low = tf.constant(normal_bp_range[0], dtype=tf.float32)
        self.normal_high = tf.constant(normal_bp_range[1], dtype=tf.float32)
        self.high_weight = tf.constant(high_weight, dtype=tf.float32)
        self.low_weight = tf.constant(low_weight, dtype=tf.float32)
        self.huber = Huber(delta=delta, reduction='none')  # Disable automatic reduction
    
    def call(self, y_true, y_pred):
        """
        Compute weighted Huber loss.
        
        Args:
            y_true: True BP values, shape (batch_size,)
            y_pred: Predicted BP values, shape (batch_size,)
        
        Returns:
            Weighted loss scalar
        """
        # Compute base Huber loss per sample (no reduction)
        huber_loss = self.huber(y_true, y_pred)
        
        # Calculate sample weights based on BP value
        # Weight = 1.0 for normal BP, increases for extreme values
        
        # High BP: weight increases linearly from 1.0 to high_weight
        # Formula: 1.0 + (high_weight - 1.0) * (bp - normal_high) / 60
        # At BP=140: weight=1.0, at BP=200: weight=high_weight
        high_bp_excess = tf.maximum(y_true - self.normal_high, 0.0)
        high_bp_weight = 1.0 + (self.high_weight - 1.0) * tf.minimum(high_bp_excess / 60.0, 1.0)
        
        # Low BP: weight increases linearly from 1.0 to low_weight
        # Formula: 1.0 + (low_weight - 1.0) * (normal_low - bp) / 30
        # At BP=90: weight=1.0, at BP=60: weight=low_weight
        low_bp_deficit = tf.maximum(self.normal_low - y_true, 0.0)
        low_bp_weight = 1.0 + (self.low_weight - 1.0) * tf.minimum(low_bp_deficit / 30.0, 1.0)
        
        # Combine weights: use high weight if BP>normal_high, low weight if BP<normal_low, else 1.0
        sample_weights = tf.where(
            y_true > self.normal_high,
            high_bp_weight,
            tf.where(y_true < self.normal_low, low_bp_weight, 1.0)
        )
        
        # Apply weights and reduce
        weighted_loss = huber_loss * sample_weights
        return tf.reduce_mean(weighted_loss)
    
    def get_config(self):
        """For model serialization."""
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'normal_bp_range': (float(self.normal_low), float(self.normal_high)),
            'high_weight': float(self.high_weight),
            'low_weight': float(self.low_weight)
        })
        return config


def create_pat_attention_layer(pat_channel, cnn_features, name_prefix="pat_attention"):
    """
    Create sigmoid-based attention gates from PAT channel to reweight CNN features.
    
    Unlike softmax attention that normalizes to sum to 1, sigmoid gating allows
    multiple cardiac cycles to contribute simultaneously with independent weights in [0,1].
    This is physiologically appropriate since multiple heartbeats can be significant.
    
    Args:
        pat_channel: PAT time series (batch, timesteps, 1)
        cnn_features: CNN feature maps (batch, timesteps, filters)
        name_prefix: Prefix for layer names
        
    Returns:
        Attended features (batch, timesteps, filters), attention_weights (batch, timesteps, 1)
    """
    # Extract PAT attention weights using sigmoid gating
    # PAT represents cardiovascular timing - use it to weight temporal importance
    attention = Dense(ATTENTION_UNITS, activation='relu', 
                     kernel_regularizer=l2(L2_REG),
                     name=f'{name_prefix}_dense1')(pat_channel)
    attention = Dense(1, activation='linear', 
                     kernel_regularizer=l2(L2_REG),
                     name=f'{name_prefix}_dense2')(attention)
    
    # Apply sigmoid to get independent attention gates in [0, 1]
    # Unlike softmax, these do NOT sum to 1, allowing multiple cardiac cycles to contribute
    attention_weights = Activation('sigmoid', name=f'{name_prefix}_sigmoid')(attention)
    
    # Broadcast attention weights to all feature channels
    # attention_weights shape: (batch, timesteps, 1)
    # cnn_features shape: (batch, timesteps, filters)
    attended_features = Multiply(name=f'{name_prefix}_multiply')([cnn_features, attention_weights])
    
    return attended_features, attention_weights


def create_phys_informed_cnn_lstm_attention(input_shape, return_attention=False):
    """
    Build physiology-informed CNN-LSTM model with PAT-based attention.
    
    Architecture:
    - 4-channel input: [ECG, PPG, PAT, HR]
    - CNN layers extract local features from all channels
    - PAT channel generates attention weights
    - Attention reweights CNN features
    - Bidirectional LSTM models temporal dynamics
    - Dense layers regress blood pressure
    
    Args:
        input_shape: Tuple (timesteps, channels) - expects 4 channels
        return_attention: If True, return attention weights for visualization
        
    Returns:
        Keras Model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input')
    
    # Verify 4 channels: [ECG, PPG, PAT, HR]
    assert input_shape[-1] == 4, "Expected 4 channels: [ECG, PPG, PAT, HR]"
    
    # ===== CNN Feature Extraction =====
    # Extract morphological features from all channels
    x = Conv1D(CONV1D_FILTERS_1, CONV1D_KERNEL_SIZE, 
               activation='relu', padding='same',
               kernel_regularizer=l2(L2_REG),
               name='conv1d_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = MaxPooling1D(2, name='pool_1')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_1')(x)
    
    x = Conv1D(CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE, 
               activation='relu', padding='same',
               kernel_regularizer=l2(L2_REG),
               name='conv1d_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    cnn_features = MaxPooling1D(2, name='pool_2')(x)
    cnn_features = Dropout(DROPOUT_RATE, name='dropout_2')(cnn_features)
    
    # ===== PAT-based Attention Mechanism =====
    # Extract PAT channel (index 2) and downsample to match CNN features
    pat_channel = Lambda(lambda x: x[:, :, 2:3], name='extract_pat')(inputs)
    
    # Downsample PAT to match pooled features (2 pooling layers with stride 2)
    pat_downsampled = MaxPooling1D(4, name='pat_downsample')(pat_channel)
    
    # Generate attention weights from PAT
    attended_features, attention_weights = create_pat_attention_layer(
        pat_downsampled, cnn_features, name_prefix='pat_attention'
    )
    
    # ===== LSTM Temporal Modeling =====
    # Bidirectional LSTM to capture forward and backward cardiovascular dynamics
    lstm_out = LSTM(LSTM_UNITS_1, return_sequences=True,
                    kernel_regularizer=l2(L2_REG),
                    recurrent_regularizer=l2(L2_REG),
                    name='lstm_1')(attended_features)
    lstm_out = Dropout(DROPOUT_RATE, name='dropout_3')(lstm_out)
    
    lstm_out = LSTM(LSTM_UNITS_2, return_sequences=False,
                    kernel_regularizer=l2(L2_REG),
                    recurrent_regularizer=l2(L2_REG),
                    name='lstm_2')(lstm_out)
    lstm_out = Dropout(DROPOUT_RATE, name='dropout_4')(lstm_out)
    
    # ===== Blood Pressure Regression =====
    dense = Dense(DENSE_UNITS, activation='relu',
                 kernel_regularizer=l2(L2_REG),
                 name='dense_1')(lstm_out)
    dense = Dropout(DROPOUT_RATE, name='dropout_5')(dense)
    
    # Dual outputs: Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP)
    # No regularization on output layers to allow full expressiveness
    sbp_output = Dense(1, activation='linear', name='sbp_output')(dense)
    dbp_output = Dense(1, activation='linear', name='dbp_output')(dense)
    
    # Create model
    if return_attention:
        model = Model(inputs=inputs, outputs=[sbp_output, dbp_output, attention_weights], 
                     name='PhysInformed_CNN_LSTM_Attention_Dual')
    else:
        model = Model(inputs=inputs, outputs=[sbp_output, dbp_output], 
                     name='PhysInformed_CNN_LSTM_Attention_Dual')
    
    # Compile with Huber loss and Adam with gradient clipping
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=GRADIENT_CLIP_NORM  # Gradient clipping for LSTM stability
    )
    
    # Use weighted loss for both SBP and DBP
    # SBP: High BP threshold 140 mmHg (hypertension stage 1)
    # DBP: High BP threshold 90 mmHg (hypertension stage 1)
    model.compile(
        optimizer=optimizer,
        loss={
            'sbp_output': WeightedHuberLoss(delta=1.0, normal_bp_range=(90, 140), high_weight=3.0, low_weight=2.0),
            'dbp_output': WeightedHuberLoss(delta=1.0, normal_bp_range=(60, 90), high_weight=3.0, low_weight=2.0)
        },
        loss_weights={
            'sbp_output': 1.0,
            'dbp_output': 1.0
        },
        metrics={
            'sbp_output': ['mae', 'mse'],
            'dbp_output': ['mae', 'mse']
        }
    )
    
    return model


def create_phys_informed_model(input_shape):
    """
    Wrapper function for backward compatibility.
    Creates the physiology-informed CNN-LSTM model with attention.
    """
    return create_phys_informed_cnn_lstm_attention(input_shape, return_attention=False)


def create_attention_visualization_model(input_shape):
    """
    Create model that returns both predictions and attention weights.
    Use this for visualizing what the model is attending to.
    """
    return create_phys_informed_cnn_lstm_attention(input_shape, return_attention=True)


# Legacy function for backward compatibility
def create_simple_cnn_gru_model(input_shape):
    """
    Simple CNN + GRU model (legacy architecture without attention).
    Kept for comparison purposes.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU
    
    model = Sequential([
        Conv1D(CONV1D_FILTERS_1, CONV1D_KERNEL_SIZE, 
               activation="relu", input_shape=input_shape, padding="same",
               kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(DROPOUT_RATE),
        Conv1D(CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE, 
               activation="relu", padding="same",
               kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(DROPOUT_RATE),
        GRU(LSTM_UNITS_1, return_sequences=True,
            kernel_regularizer=l2(L2_REG),
            recurrent_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        GRU(LSTM_UNITS_2,
            kernel_regularizer=l2(L2_REG),
            recurrent_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu",
              kernel_regularizer=l2(L2_REG)),
        Dense(1),
    ], name='Simple_CNN_GRU')
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=Huber(), metrics=["mae"])
    return model
