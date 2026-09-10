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
    Input, Conv1D, MaxPooling1D, Dropout, Dense, LSTM, GRU,
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
        DENSE_UNITS, DROPOUT_RATE, SBP_LOSS_WEIGHT, L2_REG
    )
except ImportError:
    try:
        from config import (
            LEARNING_RATE, GRADIENT_CLIP_NORM,
            CONV1D_FILTERS_1, CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE,
            LSTM_UNITS_1, LSTM_UNITS_2, ATTENTION_UNITS,
            DENSE_UNITS, DROPOUT_RATE, SBP_LOSS_WEIGHT, L2_REG
        )
    except ImportError:
        from config import (
            LEARNING_RATE, GRADIENT_CLIP_NORM,
            CONV1D_FILTERS_1, CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE,
            LSTM_UNITS_1, LSTM_UNITS_2, ATTENTION_UNITS,
            DENSE_UNITS, DROPOUT_RATE, SBP_LOSS_WEIGHT
        )
        L2_REG = 1e-4


class WeightedHuberLoss(Loss):
    """
    Huber loss with mild BP-dependent sample weighting.
    
    Weights are intentionally mild to avoid compounding with data augmentation
    and loss_weights in model.compile(). The total effective weight is:
      augmentation (2x) × WeightedHuberLoss (1.5x) × SBP_LOSS_WEIGHT (1.0x) = 3x
    Previously this was 2x × 3x × 2x = 12x, causing severe SBP over-prediction bias.
    
    - Very Low BP (<75): 2x weight
    - Low BP (75-90): 1.5x weight
    - High BP (>140): 1.5x weight
    - Normal BP (90-140): 1x weight
    """
    
    def __init__(self, delta=1.0, name='weighted_huber_loss', **kwargs):
        """
        Args:
            delta: Huber loss delta parameter (threshold for quadratic vs linear)
            name: Loss function name
            **kwargs: Additional Loss arguments (reduction, etc. for Keras 3 serialization)
        """
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.huber = Huber(delta=delta, reduction='none')  # Disable automatic reduction
    
    def call(self, y_true, y_pred):
        """
        Compute weighted Huber loss with mild BP-dependent sample weighting.
        
        Args:
            y_true: True BP values, shape (batch_size, 1) or (batch_size,)
            y_pred: Predicted BP values, shape (batch_size, 1) or (batch_size,)
        
        Returns:
            Weighted loss scalar
        """
        # Flatten to ensure consistent shape
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Compute base Huber loss per sample (no reduction)
        huber_loss = self.huber(y_true_flat, y_pred_flat)
        
        # Mild sample weights — intentionally conservative to avoid compounding with
        # data augmentation and model compile loss_weights.
        # Very Low BP (<75): 2x weight
        very_low_weight = tf.where(
            y_true_flat < 75.0,
            tf.constant(2.0),
            tf.constant(0.0)
        )
        
        # Low BP (75-90): 1.5x weight
        low_weight = tf.where(
            tf.logical_and(y_true_flat >= 75.0, y_true_flat < 90.0),
            tf.constant(1.5),
            tf.constant(0.0)
        )
        
        # High BP (>140): 1.5x weight
        high_weight = tf.where(
            y_true_flat > 140.0,
            tf.constant(1.5),
            tf.constant(0.0)
        )
        
        # Normal BP (90-140): 1x weight (default)
        normal_weight = tf.where(
            tf.logical_and(y_true_flat >= 90.0, y_true_flat <= 140.0),
            tf.constant(1.0),
            tf.constant(0.0)
        )
        
        # Combine weights (only one condition will be non-zero per sample)
        sample_weights = very_low_weight + low_weight + high_weight + normal_weight
        
        # Apply weights and reduce
        weighted_loss = huber_loss * sample_weights
        return tf.reduce_mean(weighted_loss)
    
    def get_config(self):
        """For model serialization."""
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


def create_pat_attention_layer(pat_channel, cnn_features, name_prefix="pat_attention"):
    """
    Create sigmoid-based attention gates from PAT channel to reweight CNN features.
    
    Uses Conv1D instead of Dense to capture local temporal variations in PAT, 
    preventing the attention weights from being flat constants.
    
    Args:
        pat_channel: PAT time series (batch, timesteps, 1)
        cnn_features: CNN feature maps (batch, timesteps, filters)
        name_prefix: Prefix for layer names
        
    Returns:
        Attended features (batch, timesteps, filters), attention_weights (batch, timesteps, 1)
    """
    # Extract PAT attention weights using temporal convolutions
    # PAT represents cardiovascular timing - use its local variations to weight temporal importance
    attention = Conv1D(ATTENTION_UNITS, kernel_size=5, padding='same', activation='relu', 
                       kernel_regularizer=l2(L2_REG),
                       name=f'{name_prefix}_conv1')(pat_channel)
    attention = Conv1D(1, kernel_size=5, padding='same', activation='linear', 
                       kernel_regularizer=l2(L2_REG),
                       name=f'{name_prefix}_conv2')(attention)
    
    # Apply sigmoid to get independent attention gates in [0, 1]
    # Unlike softmax, these do NOT sum to 1, allowing multiple cardiac cycles to contribute
    attention_weights = Activation('sigmoid', name=f'{name_prefix}_sigmoid')(attention)
    
    # Broadcast attention weights to all feature channels
    # attention_weights shape: (batch, timesteps, 1)
    # cnn_features shape: (batch, timesteps, filters)
    attended_features = Multiply(name=f'{name_prefix}_multiply')([cnn_features, attention_weights])
    
    return attended_features, attention_weights


def create_phys_informed_cnn_lstm_attention(input_shape, return_attention=False, use_attention=True):
    """
    Build physiology-informed CNN-LSTM model with optional PAT-based attention.
    
    Architecture:
    - 4-channel input: [ECG, PPG, PAT, HR]
    - CNN layers extract local features from all channels
    - PAT channel generates attention weights (if use_attention=True)
    - Attention reweights CNN features (if use_attention=True)
    - Bidirectional LSTM models temporal dynamics
    - Dense layers regress blood pressure
    
    Args:
        input_shape: Tuple (timesteps, channels) - expects 4 channels
        return_attention: If True, return attention weights for visualization
        use_attention: If False, ablates the attention mechanism
        
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
    
    # ===== PAT-based Attention Mechanism (or Ablation) =====
    if use_attention:
        # Extract PAT channel (index 2) and downsample to match CNN features
        pat_channel = Lambda(lambda x: x[:, :, 2:3], name='extract_pat')(inputs)
        
        # Downsample PAT to match pooled features (2 pooling layers with stride 2)
        pat_downsampled = MaxPooling1D(4, name='pat_downsample')(pat_channel)
        
        # Generate attention weights from PAT
        attended_features, attention_weights = create_pat_attention_layer(
            pat_downsampled, cnn_features, name_prefix='pat_attention'
        )
    else:
        # Attention ablated: pass CNN features directly to temporal model
        attended_features = cnn_features
        attention_weights = None
    
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
    # Initialize biases to population means (120/70) to speed up convergence
    # since targets are absolute BP values and not standardized.
    import tensorflow.keras.initializers as initializers
    sbp_output = Dense(1, activation='linear', 
                       bias_initializer=initializers.Constant(120.0),
                       name='sbp_output')(dense)
    dbp_output = Dense(1, activation='linear', 
                       bias_initializer=initializers.Constant(70.0),
                       name='dbp_output')(dense)
    
    # Create model
    if return_attention:
        model = Model(inputs=inputs, outputs=[sbp_output, dbp_output, attention_weights], 
                     name='PhysInformed_CNN_LSTM_Attention_Dual')
    else:
        model = Model(inputs=inputs, outputs=[sbp_output, dbp_output], 
                     name='PhysInformed_CNN_LSTM_Attention_Dual')
    
    # Compile with weighted Huber loss and Adam with gradient clipping
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=GRADIENT_CLIP_NORM  # Gradient clipping for LSTM stability
    )
    
    # Use WeightedHuberLoss with built-in sample weighting
    # This replaces both standard Huber loss and separate sample_weight parameter
    model.compile(
        optimizer=optimizer,
        loss={
            'sbp_output': WeightedHuberLoss(delta=1.0),
            'dbp_output': WeightedHuberLoss(delta=1.0)
        },
        loss_weights={
            'sbp_output': SBP_LOSS_WEIGHT,  # SBP weighted 3x more than DBP
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
    Ablation model: Simple CNN + GRU architecture without PAT attention.
    Dual output for SBP and DBP with WeightedHuberLoss for direct comparison.
    
    Args:
        input_shape: Tuple (timesteps, channels) - expects 4 channels
        
    Returns:
        Compiled Keras Model
    """
    inputs = Input(shape=input_shape, name='input')
    
    # ===== CNN Feature Extraction =====
    x = Conv1D(CONV1D_FILTERS_1, CONV1D_KERNEL_SIZE, 
               activation="relu", padding="same",
               kernel_regularizer=l2(L2_REG), name='conv1d_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = MaxPooling1D(2, name='pool_1')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_1')(x)
    
    x = Conv1D(CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE, 
               activation="relu", padding="same",
               kernel_regularizer=l2(L2_REG), name='conv1d_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPooling1D(2, name='pool_2')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_2')(x)
    
    # ===== GRU Temporal Modeling (No Attention) =====
    x = GRU(LSTM_UNITS_1, return_sequences=True,
            kernel_regularizer=l2(L2_REG),
            recurrent_regularizer=l2(L2_REG), name='gru_1')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_3')(x)
    
    x = GRU(LSTM_UNITS_2, return_sequences=False,
            kernel_regularizer=l2(L2_REG),
            recurrent_regularizer=l2(L2_REG), name='gru_2')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_4')(x)
    
    # ===== Blood Pressure Regression =====
    dense = Dense(DENSE_UNITS, activation="relu",
                  kernel_regularizer=l2(L2_REG), name='dense_1')(x)
    dense = Dropout(DROPOUT_RATE, name='dropout_5')(dense)
    
    # Dual outputs: SBP and DBP
    # Initialize biases to population means (120/70)
    import tensorflow.keras.initializers as initializers
    sbp_output = Dense(1, activation='linear', 
                       bias_initializer=initializers.Constant(120.0),
                       name='sbp_output')(dense)
    dbp_output = Dense(1, activation='linear', 
                       bias_initializer=initializers.Constant(70.0),
                       name='dbp_output')(dense)
    
    model = Model(inputs=inputs, outputs=[sbp_output, dbp_output], name='Simple_CNN_GRU_Dual')
    
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=GRADIENT_CLIP_NORM
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'sbp_output': WeightedHuberLoss(delta=1.0),
            'dbp_output': WeightedHuberLoss(delta=1.0)
        },
        loss_weights={
            'sbp_output': SBP_LOSS_WEIGHT,
            'dbp_output': 1.0
        },
        metrics={
            'sbp_output': ['mae', 'mse'],
            'dbp_output': ['mae', 'mse']
        }
    )
    return model
