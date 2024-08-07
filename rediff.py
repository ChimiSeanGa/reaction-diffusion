import numpy as np
from PIL import Image
from ipywidgets import interact, IntSlider
import imageio
import math
import random
import potrace

# Circular area to initialize grid
def circle_mask(x, y, cen_x, cen_y, rad):
   return (abs(x-cen_x) < rad) & (abs(y-cen_y) < rad)

# n random circular areas to initialize grid
def random_mask(x, y, x_max, y_max, n):
   mask = circle_mask(x, y, 0, 0, 0)
   for i in range(n):
      cen_x = random.uniform(0.1, x_max-0.1)
      cen_y = random.uniform(0.1, y_max-0.1)
      rad = random.uniform(0, 0.2)
      mask |= circle_mask(x, y, cen_x, cen_y, rad)
   return mask

# Initialize the grid values
def init(w, h):
   u = np.ones((h+2, w+2)) # set u to be matrix of ones
   v = np.zeros((h+2, w+2)) # set v to be matrix of zeros

   # Scaled meshgrid dimensions
   x_max = 1
   y_max = h/w

   # Set x and y to be lattice points in a meshgrid
   x, y = np.meshgrid(np.linspace(0, x_max, w+2), np.linspace(0, y_max, h+2))

   # The mask consists of the middle square of length 0.2
   # mask = (x_max/2-.2<x) & (x<x_max/2+.2) & (y_max/2-.2<y) & (y<y_max/2+.2)

   # mask = circle_mask(x, y, x_max/2, y_max/2, 0.2)
   # for i in range(4):
   #    mask = mask | circle_mask(
   #       x, y,
   #       x_max/2 + math.cos(i*math.pi/2)*0.3,
   #       y_max/2 + math.sin(i*math.pi/2)*0.3,
   #       0.05
   #    )
   mask = random_mask(x, y, x_max, y_max, 10)

   u[mask] = 0.50
   v[mask] = 0.25

   return u, v

# Set domain to be periodic
def periodic_bc(u):
   u[0, :] = u[-2, :]
   u[-1, :] = u[1, :]
   u[:, 0] = u[:, -2]
   u[:, -1] = u[:, 1]

# Second order finite difference
def laplacian(u):
   return (u[ :-2, 1:-1] + u[1:-1, :-2] - 4*u[1:-1, 1:-1] + u[1:-1, 2:]
      + u[2: , 1:-1])

# Gray-Scott model
def numpy_grayscott(U, V, Du, Dv, F, k):
   u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]

   Lu = laplacian(U)
   Lv = laplacian(V)

   uvv = u*v*v
   u += Du*Lu - uvv + F*(1-u)
   v += Dv*Lv + uvv - (F+k)*v

   periodic_bc(U)
   periodic_bc(V)

   return U, V

Du, Dv = .1, .05

# Coral growth
F, k = 0.0545, 0.062

# Mitosis
# F, k = 0.0367, 0.0649

# Worm mitosis
# F, k = 0.045, 0.065

# k, F = 0.0573, 0.03162

width, height = 300, 300
U, V = init(width, height)

# Return a mask from an input image
def image_mask(filename):
   image = Image.open('masks/' + filename)
   data = np.asarray(image)

   if data.shape[0] != height+2 or data.shape[1] != width+2:
      return None


   return data[:, :, 3] == 0

# handmask = image_mask('hand.png')

# Modifying matrices for RGB
def rgb_mods(w, h, n):
   R_mod = np.zeros((h+2, w+2))
   G_mod = np.zeros((h+2, w+2))
   B_mod = np.zeros((h+2, w+2))

   # Scaled meshgrid dimensions
   x_max = 1
   y_max = h/w

   # Absolute maximum
   abs_max = max(x_max, y_max)

   # Set x and y to be lattice points in a meshgrid
   x, y = np.meshgrid(np.linspace(0, x_max, w+2), np.linspace(0, y_max, h+2))

   # for i in range(n):
   #    mask = (
   #       (((x-x_max/2)**2 + (y-y_max/2)**2) > (abs_max/2*i/n)**2) &
   #       (((x-x_max/2)**2 + (y-y_max/2)**2) < (abs_max/2*(i+1)/n)**2)
   #    )

   #    R_mod[mask] = math.cos(i * math.pi / n) * 150
   #    G_mod[mask] = math.sin(i * math.pi / n) * 200
   #    B_mod[mask] = math.cos(i * math.pi / n + math.pi) * 120

   return R_mod, G_mod, B_mod

R_mod, G_mod, B_mod = rgb_mods(width, height, 150)

def create_image(grayscott, frame_count):
   global U, V
   for t in range(25):
      U, V = grayscott(U, V, Du, Dv, F, k)
      # V[handmask] = 0
   U_scaled = np.uint8(255*(U-U.min()) / (U.max()-U.min()))
   V_scaled = np.uint8(255*(V-V.min()) / (V.max()-V.min()))

   mask = V_scaled != 0
   R = np.copy(V_scaled)
   G = np.copy(V_scaled)
   B = np.copy(V_scaled)

   R[mask] = np.uint(R[mask] + R_mod[mask])
   G[mask] = np.uint(G[mask] + G_mod[mask])
   B[mask] = np.uint(B[mask] + B_mod[mask])

   rgb_frame = np.stack((R, G, B), axis=-1)

   return rgb_frame

def write_to_svg(frame, outfile):
   data = frame > 128

   bmp = potrace.Bitmap(frame)
   pathlist = bmp.trace()

   with open(f"{outfile}.svg", "w") as fp:
      fp.write(
         f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">''')
      parts = []
      for curve in pathlist:
         fs = curve.start_point
         parts.append(f"M{fs[0]},{fs[1]}")
         for segment in curve.segments:
            if segment.is_corner:
               a = segment.c
               b = segment.end_point
               parts.append(f"L{a[0]},{a[1]}L{b[0]},{b[1]}")
            else:
               a = segment.c1
               b = segment.c2
               c = segment.end_point
               parts.append(f"C{a[0]},{a[1]} {b[0]},{b[1]} {c[0]},{c[1]}")
         parts.append("z")
      fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
      fp.write("</svg>")

def create_frames(n, grayscott):
   return [create_image(grayscott, i) for i in range(n)]

def get_single_frame(n, grayscott):
   global U, V
   for t in range(n):
      U, V = grayscott(U, V, Du, Dv, F, k)
   U_scaled = np.uint8(255*(U-U.min()) / (U.max()-U.min()))
   V_scaled = np.uint8(255*(V-V.min()) / (V.max()-V.min()))

   mask = V_scaled != 0
   R = np.copy(V_scaled)
   G = np.copy(V_scaled)
   B = np.copy(V_scaled)

   R[mask] = np.uint(R[mask] + R_mod[mask])
   G[mask] = np.uint(G[mask] + G_mod[mask])
   B[mask] = np.uint(B[mask] + B_mod[mask])

   rgb_frame = np.stack((R, G, B), axis=-1)

   return rgb_frame

def get_bw_svg(n, outfile):
   rgb_frame = get_single_frame(n, numpy_grayscott)[1:-1,1:-1]
   bw_frame = []
   eps = 50
   for row in rgb_frame:
      bw_row = []
      for pixel in row:
         if pixel[0] > eps or pixel[1] > eps or pixel[2] > eps:
            bw_row.append(np.uint8(0))
         else:
            bw_row.append(np.uint8(255))
      bw_frame.append(bw_row)
   bw_frame = np.array(bw_frame)

   # Save rgb and bw images
   imageio.imwrite("rgb.png", rgb_frame)
   imageio.imwrite("bw.png", bw_frame)

   # Save bw svg
   write_to_svg(bw_frame, outfile)

get_bw_svg(12000, "image.svg")

# Uncomment line below to create image frames
# frames = create_frames(800, numpy_grayscott)

# def display_sequence(iframe):
#    return Image.fromarray(frames[iframe])

# interact(display_sequence,
#          iframe=IntSlider(min=0,
#                            max=len(frames)-1,
#                            step=1,
#                            value=0,
#                            continuous_update=True))

# Uncomment two lines below to generate gif from image frames
# frames_scaled = [np.uint8(255 * frame) for frame in frames]
# imageio.mimsave('movie.gif', frames_scaled, format='gif', fps=60)

# frame_count = 0
# for frame in frames:
#    imageio.mimsave('img' + str(frame_count).zfill(3) + '.png', frame)
#    frame_count += 1
