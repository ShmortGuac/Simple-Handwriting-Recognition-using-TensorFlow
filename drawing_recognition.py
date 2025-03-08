import tkinter as tk
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import random

grid_size = 28  # 28x28 pixels
cell_size = 15  # Size of each cell
canvas_size = grid_size * cell_size


# Initialize a blank canvas (numpy array)
pixel_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

model = load_model("model.h5")

def draw(event):
    x, y = event.x // cell_size, event.y // cell_size  # Get grid position
    if 0 <= x < grid_size and 0 <= y < grid_size:
        pixel_grid[y, x] =  255  # Set pixel to white (255)
        pixel_grid[y+1, x+1] = add_with_limit(pixel_grid[y+1, x+1], 64)
        pixel_grid[y-1, x-1] = add_with_limit(pixel_grid[y-1, x-1], 64)
        pixel_grid[y+1, x-1] = add_with_limit(pixel_grid[y+1, x-1], 64)
        pixel_grid[y-1, x+1] = add_with_limit(pixel_grid[y-1, x+1], 64)
        pixel_grid[y, x+1] = add_with_limit(pixel_grid[y, x+1], 128)
        pixel_grid[y, x-1] = add_with_limit(pixel_grid[y, x-1], 128)
        pixel_grid[y+1, x] = add_with_limit(pixel_grid[y+1, x], 128)
        pixel_grid[y-1, x] = add_with_limit(pixel_grid[y-1, x], 128)
        canvas.create_rectangle(
            x * cell_size, y * cell_size, 
            (x + 3) * cell_size, (y + 3) * cell_size, 
            fill='black', outline='black'
        )

def clear_canvas():
    global pixel_grid
    pixel_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    canvas.delete("all")

def predict_number():
    global pixel_grid
    pixel_grid = np.expand_dims(pixel_grid, axis=0)
    pixel_grid = np.expand_dims(pixel_grid, axis=-1)
    prediction = model.predict(pixel_grid)
    predicted_num = np.argmax(prediction)
    label.config(text=f"Predicted Number: {predicted_num}")
    for i in range(10):
        print(f"{i} : {(prediction[0][i]*100):.4f}%")

def add_with_limit(a, b):
    return min(a+b, 255)


root = tk.Tk()
root.title("28x28 Drawing Pad")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()
canvas.bind("<B1-Motion>", draw)

btn_frame = tk.Frame(root)
btn_frame.pack()

clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas)
clear_btn.pack(side=tk.LEFT)

predict_btn = tk.Button(btn_frame, text="Predict", command=predict_number)
predict_btn.pack(side=tk.LEFT)

label = tk.Label(root, text="", font=("Arial", 16))
label.pack(pady=10) 

root.mainloop()
