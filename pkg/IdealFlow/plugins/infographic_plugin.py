# plugins/infographic_plugin.py
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow
from PIL import Image, ImageDraw, ImageFont
from Automation import PluginInterface, main_thread_required

class InfographicPlugin(PluginInterface):
    def __init__(self):
        self.num_file = 15  # Default value, ensure it's set correctly in load
        self.font_path = "arial.ttf"
        self.background_image = None
        self.text_overlay_image = None

    def get_actions(self):
        return {
            'draw_fibonacci_circles_and_network': self.draw_fibonacci_circles_and_network,
            'draw_fibonacci_circles': self.draw_fibonacci_circles,
            'draw_strongly_connected_network': self.draw_strongly_connected_network,
            'add_text_overlay': self.add_text_overlay,
            'save_image': self.save_image
        }
    
    def initialize(self, num_file, font_path):
        """Set values for num_file and font_path."""
        self.num_file = num_file
        self.font_path = font_path

    @main_thread_required
    def draw_fibonacci_circles_and_network(self, lighter_colors, darker_colors, radii, centers, nodes, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')  # Hide axes

        # Draw Fibonacci circles
        for r, center in zip(radii, centers):
            circle = Circle(center, r, color=np.random.choice(lighter_colors), alpha=0.4)
            ax.add_patch(circle)

        # Draw strongly connected network
        for i, start in enumerate(nodes):
            for j, end in enumerate(nodes):
                if i != j:
                    color = np.random.choice(darker_colors)
                    ax.add_patch(FancyArrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                                            color=color, width=0.08, head_width=0.25, length_includes_head=True))
        
        # Draw nodes in the network
        for node in nodes:
            ax.plot(*node, marker='o', markersize=20, color=np.random.choice(darker_colors))

        # Save as background image for further processing
        self.background_image = f"background_{self.num_file}.png"
        plt.savefig(self.background_image, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Infographic background with Fibonacci circles and network saved as {self.background_image}")

    def draw_fibonacci_circles(self, colors, radii, centers, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')  # Hide axes

        for r, center in zip(radii, centers):
            circle = Circle(center, r, color=np.random.choice(colors), alpha=0.4)
            ax.add_patch(circle)
        
        # Save as background image for further processing
        self.background_image = f"background_{self.num_file}.png"
        plt.savefig(self.background_image, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Background image with Fibonacci circles saved as {self.background_image}")

    def draw_strongly_connected_network(self, nodes, colors, **kwargs):
        if not self.background_image:
            print("Error: Background image not set.")
            return
        try:
           # Open and convert the image
           img = Image.open(self.background_image).convert("RGB")
           print(f"Image loaded successfully: {self.background_image}")
           draw = ImageDraw.Draw(img)

           # Ensure nodes are tuples of (float, float) and draw a test line
           nodes = [(float(x), float(y)) for x, y in nodes]
           print(f"Processed nodes: {nodes}")

           # Test by drawing a single line
           draw.line([nodes[0], nodes[1]], fill="red", width=3)
           print("Single test line drawn.")
        except Exception as e:
           print(f"Error drawing line: {e}")
           return
        
        
        # Save the modified background with network
        self.background_image = f"network_{self.num_file}.png"
        img.save(self.background_image)
        print(f"Network overlay saved as {self.background_image}")

    @main_thread_required
    def add_text_overlay(self, text_content, **kwargs):
        if not self.background_image:
            print("Error: Background image not set.")
            return
        
        img = Image.open(self.background_image).convert("RGBA")
        draw = ImageDraw.Draw(img)
        
        # Load font and add text elements from content
        for text_item in text_content:
            font = ImageFont.truetype(self.font_path, text_item['size'])
            position = tuple(text_item['position'])

            if "rotate" in text_item and isinstance(text_item["rotate"], (int, float)):
                # Create a new transparent image for rotated text
                text_image = Image.new("RGBA", position, (255, 255, 255, 0))
                text_draw = ImageDraw.Draw(text_image)
                text_draw.text((0, 0), text_item["text"], fill=text_item["color"], font=font)

                # Rotate the text image by specified degrees
                rotated_text_image = text_image.rotate(text_item["rotate"], expand=True)

                # Calculate position offset to paste rotated text correctly
                x_offset = position[0] - rotated_text_image.width // 2
                y_offset = position[1] - rotated_text_image.height // 2

                # Paste rotated text on main image
                img.alpha_composite(rotated_text_image, (x_offset, y_offset))
            else:
                # Draw non-rotated text directly
                draw.text(tuple(text_item['position']), text_item['text'], fill=text_item['color'], font=font)

        # Save overlay image
        self.text_overlay_image = f"infographic_{self.num_file}.png"
        img.save(self.text_overlay_image)
        print(f"Text overlay image saved as {self.text_overlay_image}")

    @main_thread_required
    def save_image(self, **kwargs):
        # Rename final image for clarity if needed
        final_image = f"IFN_Infographic_{self.num_file}.png"
        Image.open(self.text_overlay_image).save(final_image)
        print(f"Final infographic saved as {final_image}")