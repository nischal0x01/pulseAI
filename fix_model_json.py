"""
Fix model.json to be compatible with TensorFlow.js
Converts Keras 3.x inbound_nodes format to Keras 2.x/TFJS compatible format
"""

import json
import sys

def convert_inbound_nodes(layer):
    """Convert Keras 3.x inbound_nodes to Keras 2.x format"""
    if 'inbound_nodes' not in layer or not layer['inbound_nodes']:
        return layer
    
    new_inbound_nodes = []
    
    for node in layer['inbound_nodes']:
        if isinstance(node, dict) and 'args' in node:
            # Keras 3.x format: {"args": [...], "kwargs": {...}}
            # Need to convert to: [[["layer_name", 0, 0, {}]]]
            
            args = node.get('args', [])
            kwargs = node.get('kwargs', {})
            
            # Build new node format
            new_node = []
            
            # Handle args - could be a tensor or list of tensors
            if isinstance(args, list):
                for arg in args:
                    if isinstance(arg, dict):
                        if 'class_name' == '__keras_tensor__' and 'config' in arg:
                            keras_history = arg['config'].get('keras_history', [])
                            if len(keras_history) >= 3:
                                # [layer_name, node_index, tensor_index, kwargs]
                                new_node.append([
                                    keras_history[0],
                                    keras_history[1],
                                    keras_history[2],
                                    kwargs
                                ])
                        elif isinstance(arg, list):
                            # Multiple inputs (like Multiply layer)
                            for sub_arg in arg:
                                if isinstance(sub_arg, dict) and 'config' in sub_arg:
                                    keras_history = sub_arg['config'].get('keras_history', [])
                                    if len(keras_history) >= 3:
                                        new_node.append([
                                            keras_history[0],
                                            keras_history[1],
                                            keras_history[2],
                                            {}
                                        ])
            
            if new_node:
                new_inbound_nodes.append(new_node)
        else:
            # Already in correct format or empty
            new_inbound_nodes.append(node)
    
    layer['inbound_nodes'] = new_inbound_nodes
    return layer

def fix_input_layer(layer):
    """Fix InputLayer configuration"""
    if layer.get('class_name') == 'InputLayer' and 'config' in layer:
        config = layer['config']
        # Change batch_shape to batch_input_shape
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
    return layer

def fix_model_json(input_path, output_path=None):
    """Fix model.json file"""
    if output_path is None:
        output_path = input_path
    
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        model = json.load(f)
    
    print("Fixing model structure...")
    
    # Fix modelTopology layers
    if 'modelTopology' in model and 'model_config' in model['modelTopology']:
        model_config = model['modelTopology']['model_config']
        
        if 'config' in model_config and 'layers' in model_config['config']:
            layers = model_config['config']['layers']
            
            print(f"  Processing {len(layers)} layers...")
            for i, layer in enumerate(layers):
                # Fix InputLayer
                layer = fix_input_layer(layer)
                
                # Fix inbound_nodes
                layer = convert_inbound_nodes(layer)
                
                layers[i] = layer
            
            model_config['config']['layers'] = layers
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(model, f)
    
    print("✓ Model fixed successfully!")
    print("\nChanges made:")
    print("  1. Converted batch_shape → batch_input_shape")
    print("  2. Converted inbound_nodes to TFJS compatible format")
    print("\nYou can now use this model in your Next.js app:")
    print("  const model = await tf.loadLayersModel('/models/model.json');")

if __name__ == '__main__':
    model_path = 'frontend/public/models/model.json'
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    try:
        fix_model_json(model_path)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
