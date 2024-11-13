import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def load_image():
	global img_cv
	file_path = filedialog.askopenfilename()
	if file_path:
		img_cv = cv2.imread(file_path)
		display_image(img_cv, original=True)  # Exibe a imagem original
		refresh_canvas()

		# ? Apply filter on image load
		apply_filter(selected_filter.get())

def display_image(img, original=False):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_pil = Image.fromarray(img_rgb)

	# Obtém o tamanho da imagem orifinal
	img_width, img_height = img_pil.size

	# Redimensional a imagem para caber no canvas se for muito grande
	max_size = 500
	img_pil.thumbnail((max_size, max_size))  # Maintain aspect ratio
	img_tk = ImageTk.PhotoImage(img_pil)

	# Calcula a posição para centralizar a imagem dentro do canvas se for menor
	canvas_width, canvas_height = max_size, max_size
	x_offset = (canvas_width - img_pil.width) // 2
	y_offset = (canvas_height - img_pil.height) // 2

	if original:
		original_image_canvas.delete('all')  # Limpa a canvas
		original_image_canvas.image = img_tk  # Mantém a referência viva - garbage collection
		original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
	else:
		edited_image_canvas.delete('all')  # Limapa a canvas
		edited_image_canvas.image = img_tk
		edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def apply_filter(filter_type):
	if img_cv is None:
		return
	if filter_type == 'Low Pass':
		# ? On frequency attempt
		# dft = cv2.dft(np.float32(cv2.cvtColor(np.float32(img_cv), cv2.COLOR_BGR2GRAY)), flags=cv2.DFT_COMPLEX_OUTPUT)
		# dft_shift = np.fft.fftshift(dft)
		# filtered_img = np.uint8(np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])))
		# center = filtered_img.shape
		# center = tuple(int(ti/2) for ti in center)
		# cv2.circle(filtered_img, center, radius.get(), (0,0,0), -1)
		# filtered_img = cv2.cvtColor(np.float32(filtered_img), cv2.COLOR_GRAY2BGR)
		# filtered_img = cv2.idft(np.fft.ifftshift(filtered_img))
		# ? On space
		rad = radius.get() * 2 + 1
		kernel = np.zeros((rad, rad))
		for x in range(rad):
			for y in range(rad):
				if (selected_lpfunction.get() == 'Gaussian'):
					kernel[x][y] = (1 / (2 * np.pi * (rad/5)**2)) * np.exp((-((np.abs((x+.5) - rad/2))**2 + (np.abs((y+.5) - rad/2))**2) / (2 * (rad/5)**2)))
				elif (selected_lpfunction.get() == 'Ideal'):
					if np.sqrt(np.abs((x+.5)-rad/2)**2 + np.abs((y+.5)-rad/2)**2) < rad/2:
						kernel[x][y] = 1
		if (selected_lpfunction.get() == 'Ideal'):
			kernel /= np.count_nonzero(kernel)
		filtered_img = cv2.filter2D(img_cv, -1, kernel)
	elif filter_type == 'High Pass':
		if (selected_hpfunction.get() == 'Sobel'):
			filtered_h = cv2.filter2D(img_cv, -1, np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]))
			filtered_v = cv2.filter2D(img_cv, -1, np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]))
			filtered_img = cv2.add(filtered_h, filtered_v)
		elif (selected_hpfunction.get() == 'Roberts'):
			filtered_x = cv2.filter2D(img_cv, -1, np.array([[1, 0],[0, -1]]))
			filtered_y = cv2.filter2D(img_cv, -1, np.array([[0, 1],[-1, 0]]))
			filtered_img = cv2.add(filtered_x, filtered_y)
		elif (selected_hpfunction.get() == 'Prewitt'):
			filtered_h = cv2.filter2D(img_cv, -1, np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]]))
			filtered_v = cv2.filter2D(img_cv, -1, np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]))
			filtered_img = cv2.add(filtered_h, filtered_v)
		elif (selected_hpfunction.get() == 'Roberts + Prewitt'):
			filtered_x = cv2.filter2D(img_cv, -1, np.array([[1, 0],[0, -1]]))
			filtered_y = cv2.filter2D(img_cv, -1, np.array([[0, 1],[-1, 0]]))
			filtered_h = cv2.filter2D(img_cv, -1, np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]]))
			filtered_v = cv2.filter2D(img_cv, -1, np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]))
			filtered_img = cv2.add(cv2.add(filtered_x, filtered_y), cv2.add(filtered_h, filtered_v))
		else:
			filtered_img = img_cv
		# if (selected_lpfunction.get() == 'Roberts'):
		# 	kernel[x][y] = (1 / (2 * np.pi * (rad/5)**2)) * np.exp((-((np.abs((x+.5) - rad/2))**2 + (np.abs((y+.5) - rad/2))**2) / (2 * (rad/5)**2)))
		# if (selected_lpfunction.get() == 'Prewitt'):
		# 	kernel[x][y] = (1 / (2 * np.pi * (rad/5)**2)) * np.exp((-((np.abs((x+.5) - rad/2))**2 + (np.abs((y+.5) - rad/2))**2) / (2 * (rad/5)**2)))
	elif filter_type == 'Segmentation':
		if (selected_segfunction.get() == 'Manual Simple Gray'):
			filtered_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
			filtered_img = np.where(filtered_img < threshold.get(), filtered_img * 0, (filtered_img * 0) + 255)
		elif (selected_segfunction.get() == 'Manual Simple Color'):
			filtered_img = np.where(img_cv < threshold.get(), img_cv * 0, (img_cv * 0) + 255)
		# elif (selected_segfunction.get() == 'Simple'):
		# 	filtered_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
		# 	hist = np.asarray(cv2.calcHist([filtered_img], [0], None, [256], [0, 256]))
		# 	t = 128
		# 	g1 = hist[:t]
		# 	g2 = hist[t:]
		# 	print('Starting loop')
		# 	while abs(g1.sum() - g2.sum()) > 20:
		# 		print('In loop')
		# 		# print(f'u1 = {u1}; u2 = {u2}')
		# 		# t = int((u1 + u2) / 2)
		# 		g1 = hist[:t].sum()
		# 		g2 = hist[t:].sum()
		# 		print(f't = {t}; g1-g2 = {abs(g1-g2)}')
		# 	print('Ending loop')
		# 	filtered_img = np.where(img_cv < t, img_cv * 0, (img_cv * 0) + 255)
	elif filter_type == 'Morphology':
		if (selected_segfunction.get() == 'Manual Simple Gray'):
			filtered_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
			filtered_img = np.where(filtered_img < threshold.get(), filtered_img * 0, (filtered_img * 0) + 255)
		elif (selected_segfunction.get() == 'Manual Simple Color'):
			filtered_img = np.where(img_cv < threshold.get(), img_cv * 0, (img_cv * 0) + 255)
		if (selected_morphkernel.get() == 'Block 3x3'):
			kernel = np.ones((3, 3))
		elif (selected_morphkernel.get() == 'Block 5x5'):
			kernel = np.ones((5, 5))
		elif (selected_morphkernel.get() == 'Block 7x7'):
			kernel = np.ones((7, 7))
		elif (selected_morphkernel.get() == 'Cross 3x3'):
			kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
		elif (selected_morphkernel.get() == 'Cross 6x6'):
			kernel = np.array([[0,0,1,1,0,0],[0,0,1,1,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,0,0],[0,0,1,1,0,0]])
		elif (selected_morphkernel.get() == 'L 2x2'):
			kernel = np.array([[1,0],[1,1]])
		elif (selected_morphkernel.get() == 'L 4x4'):
			kernel = np.array([[1,1,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]])
		elif (selected_morphkernel.get() == 'L 6x6'):
			kernel = np.array([[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])

		#? Setup
		h, w = filtered_img.shape
		kh, kw = kernel.shape
		pad = kw//2
		filtered_img = np.pad(filtered_img, pad)
		inverted_kernel = np.where(kernel==1, -1, 1)

		#? Operations
		def erode():
			template = filtered_img
			template = np.pad(template, pad)
			for y in range(h):
				for x in range(w):
					roi = template[y:y+kh, x:x+kw] * kernel
					if np.count_nonzero(roi) == np.count_nonzero(kernel):
						filtered_img[y, x] = 255
					else:
						filtered_img[y, x] = 0
		def dilate():
			template = filtered_img
			template = np.pad(template, pad)
			for y in range(h):
				for x in range(w):
					roi = template[y:y+kh, x:x+kw] * kernel
					filtered_img[y, x] = np.max(roi)

		if (selected_morphfunction.get() == 'Erosion'):
			erode()
			filtered_img = filtered_img[pad:-pad, pad:-pad]
		elif (selected_morphfunction.get() == 'Dilation'):
			dilate()
			filtered_img = filtered_img[pad:-pad, pad:-pad]
		elif (selected_morphfunction.get() == 'Opening'):
			erode()
			dilate()
			filtered_img = filtered_img[pad:-pad, pad:-pad]
		elif (selected_morphfunction.get() == 'Closing'):
			dilate()
			erode()
			filtered_img = filtered_img[pad:-pad, pad:-pad]

		#? Experiments
		elif (selected_morphfunction.get() == 'Wrong Erosion'):
			filtered_img = cv2.filter2D(filtered_img, -1, inverted_kernel)
		elif (selected_morphfunction.get() == 'Wrong Dilation'):
			filtered_img = cv2.filter2D(filtered_img, -1, kernel)
		elif (selected_morphfunction.get() == 'Wrong Opening'):
			filtered_img = cv2.filter2D(filtered_img, -1, inverted_kernel)
			filtered_img = cv2.filter2D(filtered_img, -1, kernel)
		elif (selected_morphfunction.get() == 'Wrong Closing'):
			filtered_img = cv2.filter2D(filtered_img, -1, kernel)
			filtered_img = cv2.filter2D(filtered_img, -1, inverted_kernel)
	elif filter_type == 'FFT':
		dft = cv2.dft(np.float32(cv2.cvtColor(np.float32(img_cv), cv2.COLOR_BGR2GRAY)), flags=cv2.DFT_COMPLEX_OUTPUT)
		dft_shift = np.fft.fftshift(dft)
		filtered_img = np.uint8(np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])))
		filtered_img = (filtered_img - filtered_img.min()) * (255 / (filtered_img.max() - filtered_img.min()))
		filtered_img = filtered_img.astype(np.uint8)
	display_image(filtered_img, original=False)  # Exibe a imagem editada

def refresh_canvas():
	edited_image_canvas.delete('all')  # Limpa a canvas para exibir a nova imagem

# Definindo a GUI
root = tk.Tk()
root.title('Image Processing App')

# Define o tamanho da janela da aplicação 1200x800
root.geometry('1384x540')

# Define a cor de fundo da janela
root.config(bg='#111111')

img_cv = None

# Cria o menu da aplicação
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label='File', menu=file_menu)
file_menu.add_command(label='Load Image', command=load_image)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=root.quit)

# Filters menu
# filters_menu = tk.Menu(menu_bar, tearoff=0)
# menu_bar.add_cascade(label='Filters', menu=filters_menu)
# filters_menu.add_command(label='Low Pass Filter', command=lambda: apply_filter('Low Pass'))
# filters_menu.add_command(label='High Pass Filter', command=lambda: apply_filter('High Pass'))
# filters_menu.add_command(label='FFT', command=lambda: apply_filter('FFT'))

# Cria a canvas para a imagem original com borda (sem background)
original_image_canvas = tk.Canvas(root, width=500, height=500, bg='#111111', highlightthickness=1, highlightbackground='#333333')
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

# Cria a canvas para a imagem editada com borda (sem background)
edited_image_canvas = tk.Canvas(root, width=500, height=500, bg='#111111', highlightthickness=1, highlightbackground='#333333')
edited_image_canvas.grid(row=0, column=2, padx=20, pady=20)

# Settings area
settings_area = tk.Frame(root, width=300, height=500, background='#111111', highlightthickness=1, highlightbackground='#333333')
settings_area.grid(row=0, column=1)

# Filter selector
def select():
	if (selected_filter.get() == 'Low Pass'):
		lpfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
		hpfunction_dropdown.grid_forget()
		segfunction_dropdown.grid_forget()
		morphfunction_dropdown.grid_forget()
		morphkernel_dropdown.grid_forget()
		radius.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10))
		threshold.grid_forget()
	elif (selected_filter.get() == 'High Pass'):
		lpfunction_dropdown.grid_forget()
		hpfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
		segfunction_dropdown.grid_forget()
		morphfunction_dropdown.grid_forget()
		morphkernel_dropdown.grid_forget()
		radius.grid_forget()
		threshold.grid_forget()
	elif (selected_filter.get() == 'Segmentation'):
		lpfunction_dropdown.grid_forget()
		hpfunction_dropdown.grid_forget()
		segfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
		morphfunction_dropdown.grid_forget()
		morphkernel_dropdown.grid_forget()
		radius.grid_forget()
		threshold.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10))
	elif (selected_filter.get() == 'Morphology'):
		lpfunction_dropdown.grid_forget()
		hpfunction_dropdown.grid_forget()
		segfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
		morphfunction_dropdown.grid(row=2, column=0, padx=10, pady=(0, 10))
		morphkernel_dropdown.grid(row=2, column=1, padx=(0, 10), pady=(0, 10))
		radius.grid_forget()
		threshold.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10))
		# morph_kernel_size.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10))
		# morph_kernel_area.grid(row=4, column=0, padx=10, pady=(0, 10))
		# mks = int(morph_kernel_size.get())
		# for x in range(10):
		# 	for y in range(10):
		# 		if x >= mks or y >= mks:
		# 			morph_checkboxes[y][x].forget()
		# 		else:
		# 			morph_checkboxes[y][x].grid(row=y, column=x)
	else:
		lpfunction_dropdown.grid_forget()
		hpfunction_dropdown.grid_forget()
		segfunction_dropdown.grid_forget()
		morphfunction_dropdown.grid_forget()
		morphkernel_dropdown.grid_forget()
		radius.grid_forget()
		threshold.grid_forget()
	apply_filter(selected_filter.get())

selected_filter = tk.StringVar(root)
selected_filter.set('Morphology')
filter_options = [ 'Low Pass', 'High Pass', 'Segmentation', 'Morphology', 'FFT' ]
filter_dropdown = tk.OptionMenu(settings_area, selected_filter, *filter_options, command=lambda a: select())
filter_dropdown.grid(row=0, column=0, padx=10, pady=10)
filter_dropdown.config(width=13)

# Low-pass function selector
selected_lpfunction = tk.StringVar(root)
selected_lpfunction.set('Gaussian')
lpfunction_options = [ 'Ideal', 'Gaussian' ]
lpfunction_dropdown = tk.OptionMenu(settings_area, selected_lpfunction, *lpfunction_options, command=lambda a: select())
lpfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
lpfunction_dropdown.config(width=13)

# High-pass function selector
selected_hpfunction = tk.StringVar(root)
selected_hpfunction.set('Sobel')
hpfunction_options = [ 'Sobel', 'Roberts', 'Prewitt', 'Roberts + Prewitt' ]
hpfunction_dropdown = tk.OptionMenu(settings_area, selected_hpfunction, *hpfunction_options, command=lambda a: select())
hpfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
hpfunction_dropdown.config(width=13)

# Segmentation function selector
selected_segfunction = tk.StringVar(root)
selected_segfunction.set('Manual Simple Gray')
segfunction_options = [ 'Manual Simple Gray', 'Manual Simple Color' ]
segfunction_dropdown = tk.OptionMenu(settings_area, selected_segfunction, *segfunction_options, command=lambda a: select())
segfunction_dropdown.grid(row=0, column=1, padx=(0, 10), pady=10)
segfunction_dropdown.config(width=13)

# Morphology function selector
selected_morphfunction = tk.StringVar(root)
selected_morphfunction.set('Erosion')
morphfunction_options = [ 'Erosion', 'Dilation', 'Opening', 'Closing', 'Wrong Erosion', 'Wrong Dilation', 'Wrong Opening', 'Wrong Closing' ]
morphfunction_dropdown = tk.OptionMenu(settings_area, selected_morphfunction, *morphfunction_options, command=lambda a: select())
morphfunction_dropdown.grid(row=2, column=0, padx=10, pady=(0, 10))
morphfunction_dropdown.config(width=13)

# Morphology kernel selector
selected_morphkernel = tk.StringVar(root)
selected_morphkernel.set('Block 3x3')
morphkernel_options = [ 'Block 3x3', 'Block 5x5', 'Block 7x7', 'Cross 3x3', 'Cross 6x6', 'L 2x2', 'L 4x4', 'L 6x6' ]
morphkernel_dropdown = tk.OptionMenu(settings_area, selected_morphkernel, *morphkernel_options, command=lambda a: select())
morphkernel_dropdown.grid(row=2, column=1, padx=(0, 10), pady=(0, 10))
morphkernel_dropdown.config(width=13)

# Filter radius
radius = tk.Scale(settings_area, from_=1, to=100, orient='horizontal', label='Radius', command=lambda a: select(), length=260, highlightthickness=1, highlightbackground='#333333')
radius.set(5)
radius.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10))

# Threshold value
threshold = tk.Scale(settings_area, from_=0, to=255, orient='horizontal', label='Threshold value', command=lambda a: select(), length=280, highlightthickness=1, highlightbackground='#333333')
threshold.set(128)
threshold.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10))

# Morphology kernel grid holder
# morph_kernel_area = tk.Frame(settings_area, width=280, height=500, background='#111111', highlightthickness=1, highlightbackground='#333333')
# morph_kernel_area.grid(row=4, column=0)
# morph_kernel = [[False] * 10] * 10
# morph_checkboxes = [[None] * 10] * 10
# for x in range(10):
# 	for y in range(10):
# 		morph_checkboxes[x][y] = tk.Checkbutton(morph_kernel_area, command=lambda: select())
# 		# morph_checkboxes[y][x] = tk.Label(morph_kernel_area, text=f'{x},{y}')
root.mainloop()
select()