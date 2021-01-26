import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

ells = [Ellipse([261.4118664, 31.87211982],
                width=45.18479809, height=49.01832325,
                angle=358.63),
        Ellipse([6.84277425, 3.41310058],
                width=0.54364185370934259, height=0.8764400643751178,
                angle=118.91284520756146),
        Ellipse([1.64398272, 6.08032175],
                width=2.14505827284187034, height=3.1755193903053578,
                angle=212.19525714110736),
        Ellipse([3.72737711, 1.94535695],
                width=0.86166603358924039, height=1.22543344631533513,
                angle=215.2179399967015)
        ]
# [0,0] #[3.72737711, 1.94535695]
colors = [[0.73733322, 0.59192932, 0.41040727], [0.46288542, 0.75901567, 0.13778732],
          [0.25455402, 0.44709669, 0.1072266],
          [0.57353451, 0.82633143, 0.5778575]]

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
for i, e in enumerate(ells):
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_facecolor(colors[i])

# plt.plot([0, 1.64398272], [0, 6.08032175], 'ro-')
plt.plot([3.72737711, 1.64398272], [1.94535695, 6.08032175], 'ro-')
# plt.plot([0, 6.84277425], [0, 3.41310058], 'ro-')
plt.plot([3.72737711, 6.84277425], [1.94535695, 3.41310058], 'ro-')
# plt.plot([0, 6.6999718], [0, 6.02911497], 'ro-')
plt.plot([3.72737711, 6.6999718], [1.94535695, 6.02911497], 'ro-')

ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)


def get_slope_intercept(x1, x2, y1, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m, b


# ATTEMPT 1
# c_x = 0
# c_y = 0

# ELLIPSE 1
c_x = 3.72737711
c_y = 1.94535695
angle = math.radians(212.19525714110736)
width = 0.86166603358924039 / 2  # h # a
height = 1.22543344631533513 / 2  # v # b

# ELLIPSE 2
c_x2 = 1.64398272
c_y2 = 6.08032175
angle2 = math.radians(215.2179399967015)
width2 = 2.14505827284187034 / 2  # h # a
height2 = 3.1755193903053578 / 2  # v # b

# ELLIPSE 3
c_x3 = 6.84277425
c_y3 = 3.41310058
angle3 = math.radians(118.91284520756146)
width3 = 0.54364185370934259 / 2  # h # a
height3 = 0.8764400643751178 / 2  # v # b

# ELLIPSE 4
c_x4 = 6.6999718
c_y4 = 6.02911497
angle4 = math.radians(107.22442543139232)
width4 = 0.43721214243621875 / 2  # h # a
height4 = 0.9917565566081832 / 2  # v # b


def affline_transformation(cx, cy, theta, px, py, a, b):
    # translate point to origin
    px = px - cx
    py = py - cy
    # rotate point about the negative angle
    # Expand along y axis (or shrink)
    px1 = ((px * math.cos(theta)) + (py * math.sin(theta)))
    py1 = (((-px * math.sin(theta)) + (py * math.cos(theta))) * a / b)
    # find length of vector
    vect_len = math.hypot(px1, py1)
    # find intersection point
    # Shrink back to original shape
    ix = a * px1 / vect_len
    iy = (a * py1 / vect_len) * b / a
    # Rotate back to original point
    # Translate back to center
    x_f = (ix * math.cos(theta)) - (iy * math.sin(theta)) + cx
    y_f = (ix * math.sin(theta)) + (iy * math.cos(theta)) + cy

    return x_f, y_f


x1, y1 = affline_transformation(c_x, c_y, angle, 1.64398272, 6.08032175, width, height)
x2, y2 = affline_transformation(c_x, c_y, angle, 6.84277425, 3.41310058, width, height)
x3, y3 = affline_transformation(c_x, c_y, angle, 6.6999718, 6.02911497, width, height)

x4, y4 = affline_transformation(c_x2, c_y2, angle2, c_x, c_y, width2, height2)
x5, y5 = affline_transformation(c_x3, c_y3, angle3, c_x, c_y, width3, height3)
x6, y6 = affline_transformation(c_x4, c_y4, angle4, c_x, c_y, width4, height4)

# pairs
plt.plot(x1, y1, marker='o', markersize=3, color='k')
plt.plot(x4, y4, marker='o', markersize=3, color='k')
black_distance = math.hypot(x4 - x1, y4 - y1)

plt.plot(x2, y2, marker='o', markersize=3, color='m')
plt.plot(x5, y5, marker='o', markersize=3, color='m')
magenta_distance = math.hypot(x5 - x2, y5 - y2)

plt.plot(x3, y3, marker='o', markersize=3, color='b')
plt.plot(x6, y6, marker='o', markersize=3, color='b')
blue_distance = math.hypot(x6 - x3, y6 - y3)

plt.text(2, 3.5, 'dist = ' + str(round(black_distance, 2)), color='k')
plt.text(4.5, 4, 'dist = ' + str(round(blue_distance, 2)), color='b')
plt.text(4, 2.5, 'dist = ' + str(round(magenta_distance, 2)), color='m')
plt.title('Distance Between Ellipses')
plt.xlabel('Horizontal Distance - X Coordinate (Pxls)')
plt.ylabel('Vertical Distance - Y Coordinate (Pxls)')

# ATTEMPT 2
m1, b1 = get_slope_intercept(c_x, 1.64398272, c_y, 6.08032175)
m2, b2 = get_slope_intercept(c_x, 6.84277425, c_y, 3.41310058)
m3, b3 = get_slope_intercept(c_x, 6.6999718, c_y, 6.02911497)


def line_ellipse_intersection(m, b_1, theta, h, v, cx, cy):
    a = (v ** 2 * (
            (math.cos(theta)) ** 2 + 2 * m * math.cos(theta) * math.sin(theta) + m ** 2 * (math.sin(theta)) ** 2)) \
        + (h ** 2 * (
            m ** 2 * (math.cos(theta)) ** 2 - 2 * m * math.cos(theta) * math.sin(theta) + (math.sin(theta)) ** 2))
    b = (2 * v ** 2 * (b_1 + (m * cx) - cy) * (math.cos(theta) * math.sin(theta) + m * (math.sin(theta)) ** 2)) + \
        (2 * h ** 2 * (b_1 + (m * cx) - cy) * (m * (math.cos(theta)) ** 2 - math.cos(theta) * math.sin(theta)))
    c = (b_1 + (m * cx) - cy) ** 2 * (v ** 2 * (math.sin(theta)) ** 2 + h ** 2 * (math.cos(theta)) ** 2) - (
            h ** 2 * v ** 2)

    x_1 = ((-b + math.sqrt((b ** 2 - 4 * a * c))) / (2 * a)) + cx
    x_2 = ((-b + math.sqrt((b ** 2 - 4 * a * c))) / (2 * a)) + cx

    y_1 = m * x_1 + b_1
    y_2 = m * x_2 + b_1

    return (x_1, y_1), (x_2, y_2)

# i_1, i_2 = line_ellipse_intersection(m1, b1, angle, width, height, c_x, c_y)
# i_3, i_4 = line_ellipse_intersection(m2, b2, angle, width, height, c_x, c_y)
# i_5, i_6 = line_ellipse_intersection(m3, b3, angle, width, height, c_x, c_y)

# plt.plot(i_1[0], i_1[1], marker='o', markersize=3, color='black')
# plt.plot(i_3[0], i_3[1], marker='o', markersize=3, color='black')
# plt.plot(i_6[0], i_6[1], marker='o', markersize=3, color='black')


