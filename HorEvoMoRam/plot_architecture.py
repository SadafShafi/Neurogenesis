import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def parse_model_from_checkpoint(checkpoint_path):
    # Load the state dict
    # map_location='cpu' to ensure it loads even if trained on GPU
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    architecture = {}
    
    for key, tensor in state_dict.items():
        # Keys look like: 'actor_fc1.base_layer.weight', 'actor_fc1.A_list.0', etc.
        parts = key.split('.')
        layer_name = parts[0]
        
        if layer_name not in architecture:
            architecture[layer_name] = {
                'base': None,
                'adapters': {}
            }
            
        if parts[1] == 'base_layer':
            if parts[2] == 'weight':
                # shape is [out_features, in_features]
                architecture[layer_name]['base'] = {
                    'in_features': tensor.shape[1],
                    'out_features': tensor.shape[0]
                }
        elif parts[1] == 'A_list':
            adapter_idx = parts[2]
            if adapter_idx not in architecture[layer_name]['adapters']:
                architecture[layer_name]['adapters'][adapter_idx] = {}
            # A_list tensor shape is [r, in_features]
            architecture[layer_name]['adapters'][adapter_idx]['r'] = tensor.shape[0]
            architecture[layer_name]['adapters'][adapter_idx]['in_features'] = tensor.shape[1]
        elif parts[1] == 'B_list':
            adapter_idx = parts[2]
            if adapter_idx not in architecture[layer_name]['adapters']:
                architecture[layer_name]['adapters'][adapter_idx] = {}
            # B_list tensor shape is [out_features, r]
            architecture[layer_name]['adapters'][adapter_idx]['out_features'] = tensor.shape[0]

    return architecture

def print_architecture(architecture):
    print("=" * 60)
    print(" HorEvoMoRam Model Architecture (Parsed from Checkpoint) ")
    print("=" * 60)
    
    for layer_name, info in architecture.items():
        if info['base'] is None:
            continue
            
        print(f"\nLayer: {layer_name}")
        base_in = info['base']['in_features']
        base_out = info['base']['out_features']
        print(f"  ├── Base Layer: Linear(in_features={base_in}, out_features={base_out})")
        
        adapters = info['adapters']
        if not adapters:
            print("  └── Adapters: None")
        else:
            print(f"  └── Adapters ({len(adapters)} total):")
            for idx in sorted(adapters.keys(), key=lambda x: int(x)):
                r = adapters[idx].get('r', 'Unknown')
                print(f"      ├── Task {idx}: rank={r}")
    print("\n" + "=" * 60)

def plot_architecture(architecture, output_path="architecture_plot.png", show_adapters=True):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    y_pos = 0.9
    x_actor = 0.2
    x_critic = 0.7
    
    box_width = 0.3
    box_height = 0.1
    
    title = "HorEvoMoRam Architecture" if show_adapters else "Original Model Architecture (No Adapters)"
    plt.text(0.5, 0.98, title, fontsize=16, fontweight='bold', ha='center')
    
    def draw_layer(x, y, layer_name, info):
        base_in = info['base']['in_features']
        base_out = info['base']['out_features']
        adapters = info['adapters']
        num_adapters = len(adapters)
        
        # Base layer box
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        
        text = f"{layer_name}\nLinear({base_in} -> {base_out})"
        plt.text(x + box_width/2, y + box_height/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Adapters
        if show_adapters and num_adapters > 0:
            adapter_y = y - 0.05
            adapter_x_start = x + 0.02
            adapter_width = (box_width - 0.04) / num_adapters
            
            for i, idx in enumerate(sorted(adapters.keys(), key=lambda k: int(k))):
                r = adapters[idx].get('r', '?')
                ax_rect = patches.Rectangle((adapter_x_start + i * adapter_width, adapter_y), 
                                            adapter_width * 0.9, 0.04, 
                                            linewidth=1, edgecolor='darkred', facecolor='salmon')
                ax.add_patch(ax_rect)
                plt.text(adapter_x_start + i * adapter_width + (adapter_width * 0.9)/2, adapter_y + 0.02, 
                         f"A{idx}\nr={r}", ha='center', va='center', fontsize=8)
                
                # Draw connecting line from base
                plt.plot([adapter_x_start + i * adapter_width + (adapter_width * 0.9)/2, adapter_x_start + i * adapter_width + (adapter_width * 0.9)/2],
                         [adapter_y, y], color='gray', linestyle=':')
    
    # Sort layers
    actor_layers = [l for l in architecture.keys() if 'actor' in l]
    critic_layers = [l for l in architecture.keys() if 'critic' in l]
    
    # Sort by suffix typically 'fc1', 'fc2', 'out'
    def sort_key(name):
        if 'fc1' in name: return 1
        elif 'fc2' in name: return 2
        elif 'out' in name: return 3
        else: return 4
        
    actor_layers.sort(key=sort_key)
    critic_layers.sort(key=sort_key)
    
    plt.text(x_actor + box_width/2, y_pos + 0.02, "Actor Network", fontsize=14, fontweight='bold', ha='center')
    plt.text(x_critic + box_width/2, y_pos + 0.02, "Critic Network", fontsize=14, fontweight='bold', ha='center')
    
    step_y = 0.25 if show_adapters else 0.18
    arrow_offset = 0.06 if show_adapters else 0.01
    arrow_length = 0.06 if show_adapters else 0.05
    
    # Draw Actor
    current_y = y_pos - 0.15
    for i, layer_name in enumerate(actor_layers):
        if architecture[layer_name]['base'] is not None:
            draw_layer(x_actor, current_y, layer_name, architecture[layer_name])
            if i < len(actor_layers) - 1:
                # Arrow
                plt.arrow(x_actor + box_width/2, current_y - arrow_offset, 0, -arrow_length, head_width=0.02, head_length=0.02, fc='k', ec='k')
            current_y -= step_y
            
    # Draw Critic
    current_y = y_pos - 0.15
    for i, layer_name in enumerate(critic_layers):
        if architecture[layer_name]['base'] is not None:
            draw_layer(x_critic, current_y, layer_name, architecture[layer_name])
            if i < len(critic_layers) - 1:
                # Arrow
                plt.arrow(x_critic + box_width/2, current_y - arrow_offset, 0, -arrow_length, head_width=0.02, head_length=0.02, fc='k', ec='k')
            current_y -= step_y

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Architecture plot saved to {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_file = os.path.join(script_dir, "models", "ppo_minigrid_horevomoram_final.pt")
    output_plot = os.path.join(script_dir, "architecture_plot.png")
    output_original_plot = os.path.join(script_dir, "architecture_original_plot.png")
    
    if not os.path.exists(checkpoint_file):
        print(f"Error: Could not find model checkpoint at {checkpoint_file}")
        print("Please ensure the model has been trained and saved.")
    else:
        arch = parse_model_from_checkpoint(checkpoint_file)
        print_architecture(arch)
        
        # Plot with adapters
        plot_architecture(arch, output_path=output_plot, show_adapters=True)
        
        # Plot without adapters (original structure)
        plot_architecture(arch, output_path=output_original_plot, show_adapters=False)
