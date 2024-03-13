import numpy as np
from PIL import Image
from ipywidgets import interact, IntSlider
import imageio
import math

def circle_mask(x, y, cen_x, cen_y, rad):
   return (abs(x-cen_x) < rad) & (abs(y-cen_y) < rad)

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

   mask = circle_mask(x, y, x_max/2, y_max/2, 0.01)
   for i in range(4):
      mask = mask | circle_mask(
         x, y,
         x_max/2 + math.cos(i*math.pi/2)*0.3,
         y_max/2 + math.sin(i*math.pi/2)*0.3,
         0.05
      )

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
# F, k = 0.0545, 0.062

# Mitosis
# F, k = 0.0367, 0.0649

# Worm mitosis
# F, k = 0.045, 0.065

k, F = 0.0573, 0.03162

width, height = 540, 860
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

   for i in range(n):
      mask = (
         (((x-x_max/2)**2 + (y-y_max/2)**2) > (abs_max/2*i/n)**2) &
         (((x-x_max/2)**2 + (y-y_max/2)**2) < (abs_max/2*(i+1)/n)**2)
      )

      R_mod[mask] = math.cos(i * math.pi / n) * 150
      G_mod[mask] = math.sin(i * math.pi / n) * 200
      B_mod[mask] = math.cos(i * math.pi / n + math.pi) * 120

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

def create_frames(n, grayscott):
   return [create_image(grayscott, i) for i in range(n)]

frames = create_frames(800, numpy_grayscott)

# def display_sequence(iframe):
#    return Image.fromarray(frames[iframe])

# interact(display_sequence,
#          iframe=IntSlider(min=0,
#                            max=len(frames)-1,
#                            step=1,
#                            value=0,
#                            continuous_update=True))

frames_scaled = [np.uint8(255 * frame) for frame in frames]
imageio.mimsave('movie.gif', frames_scaled, format='gif', fps=60)
# frame_count = 0
# for frame in frames:
#    imageio.mimsave('img' + str(frame_count).zfill(3) + '.png', frame)
#    frame_count += 1
