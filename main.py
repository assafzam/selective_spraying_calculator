import cProfile
import io
import json
import math
import pstats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MM_PER_PIXEL = 0.5  # 0.5 is default value
FIELDS_NAMES = {
    21: 'Fleming - 18117',
    26: 'Home Farm',
    27: 'Byrne IP',
    28: 'Gervais Claymore',
    30: 'Barn 5',
    31: 'Farm Group 1',
    33: 'Don Radar',
    34: 'Hartman Remple 60',
    35: 'Soybean Site',
    37: 'Sulliovan Obrien',
    38: 'Herdman',
    47: 'Pitel - Weeds',
    147: 'Field 20889',
    169: 'Field 1',
    170: 'Field 41',
    173: 'Field 8, 12'
}
FIELDS_MM_PER_PIXEL = {
    21: 0.4}
"""    
    26: 0.5,
    27: 0.7,
    28: 0.5}

    30: 0.5,
    31: 0.4,
    33: 0.5,
    34: 0.5,
    35: 0.5,
    37: 0.4,
    38: 0.4,
    47: 0.4,
    147: 0.7,
    169: 0.5,
    170: 0.5,
    173: 0.5
}"""
IMAGE_WIDTH_PIXELS = 4864
IMAGE_HEIGHT_PIXELS = 3648


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def convert_cm_2_pixel(cm, mm_per_pixel):
    return convert_mm_2_pixel(10 * cm, mm_per_pixel)


def convert_mm_2_pixel(mm, mm_per_pixel):
    return int(mm // mm_per_pixel)


class Box:
    def __init__(self, data):
        self.x1 = data.get('x')
        self.y1 = data.get('y')
        self.width = data.get('width')
        self.height = data.get('height')
        self.name = data.get('name')
        self.display_name = data.get('displayName')

    def getCoordinates(self):
        return self.x1, self.y1, self.x1 + self.width, self.y1 + self.height


class Cell:
    # cell height and width units is pixels
    def __init__(self, x, y, cell_height_pixels, cell_width_pixels):
        self.height_pixels = cell_height_pixels
        self.width_pixels = cell_width_pixels
        self.x = x
        self.y = y


class Grid:
    def __init__(self, cell_width_pixels, cell_height_pixels, grid_width_pixels=IMAGE_WIDTH_PIXELS,
                 grid_height_pixels=IMAGE_HEIGHT_PIXELS):
        self.cell_width_pixels = cell_width_pixels
        self.cell_height_pixels = cell_height_pixels
        self.height_pixels = grid_height_pixels
        self.width_pixels = grid_width_pixels
        self.base_cell = Cell(x=0, y=0, cell_width_pixels=cell_width_pixels, cell_height_pixels=cell_height_pixels)
        self.cells = {}
        self.right_edge_cell_width = grid_width_pixels % cell_width_pixels
        self.bottom_edge_cell_height = grid_height_pixels % cell_height_pixels

    def build(self):
        cells = {}

        is_there_special_right_edges = self.right_edge_cell_width != 0
        is_there_special_bottom_edges = self.bottom_edge_cell_height != 0

        amount_of_cells = [self.width_pixels // self.cell_width_pixels,
                           self.height_pixels // self.cell_height_pixels]
        if is_there_special_right_edges:
            amount_of_cells = [amount_of_cells[0] + 1, amount_of_cells[1]]
        if is_there_special_bottom_edges:
            amount_of_cells = [amount_of_cells[0], amount_of_cells[1] + 1]

        for x in range(0, amount_of_cells[0]):
            _width = self.cell_width_pixels
            _height = self.cell_height_pixels

            if is_there_special_right_edges and (x == amount_of_cells[0] - 1):
                _width = self.right_edge_cell_width
            for y in range(0, amount_of_cells[1]):
                if is_there_special_bottom_edges and (y == amount_of_cells[1] - 1):
                    _height = self.bottom_edge_cell_height

                cell = Cell(x=x * self.cell_width_pixels, y=y * self.cell_height_pixels, cell_width_pixels=_width,
                            cell_height_pixels=_height)
                key = (cell.x, cell.y)
                cells[key] = cell
        self.cells = cells

        return self


def is_valid_weed_box(weed_box):
    if -10 < weed_box.x1 < 0:
        weed_box.x1 = 0
    if -10 < weed_box.y1 < 0:
        weed_box.y1 = 0
    if weed_box.x1 <= -10 or weed_box.y1 <= 10:
        return False
    if weed_box.x1 + weed_box.width > IMAGE_WIDTH_PIXELS:
        return False
    if weed_box.y1 + weed_box.height > IMAGE_HEIGHT_PIXELS:
        return False
    return True


class Image:
    def __init__(self, image_id, weed_boxes, grid=None):
        self.id = image_id
        self.weed_boxes = weed_boxes
        self.grid = grid

    def find_grid_cells_intersect(self, weed_box):
        grid_width = self.grid.base_cell.width_pixels
        grid_height = self.grid.base_cell.height_pixels

        cells_in_x_axis = math.ceil(weed_box.width / grid_width) + 1
        cells_in_y_axis = math.ceil(weed_box.height / grid_height) + 1

        x_start = weed_box.x1 - (weed_box.x1 % grid_width)
        x_end = weed_box.x1 + weed_box.width - ((weed_box.x1 + weed_box.width) % grid_width)
        x_coordinates = range(x_start, x_end + 1, grid_width)

        y_start = weed_box.y1 - (weed_box.y1 % grid_height)
        y_end = weed_box.y1 + weed_box.height - ((weed_box.y1 + weed_box.height) % grid_height)
        y_coordinates = range(y_start, y_end + 1, grid_height)

        return {(x, y): self.grid.cells.get((x, y)) for x in x_coordinates for y in y_coordinates}

        # return {self.grid.cells.get(coordinate) for coordinate in coordinates}

    def calculate_image_spraying_present(self):
        sprayed_area = self.get_image_sprayed_area()
        image_area = self.grid.height_pixels * self.grid.width_pixels
        return 100 * (sprayed_area / image_area)

    def get_image_sprayed_area(self):
        sprayed_cells = {}
        for weed_box in self.weed_boxes:
            if is_valid_weed_box(weed_box):
                intersected_cells = self.find_grid_cells_intersect(weed_box)
                sprayed_cells.update(intersected_cells)

        sprayed_area = 0
        for cell in sprayed_cells.values():
            sprayed_area += cell.width_pixels * cell.height_pixels

        return sprayed_area


class Field:
    def __init__(self, _field_id, grid_cell_width, grid_cell_height, db):
        self.id = _field_id
        self.grid_cell_width = grid_cell_width
        self.grid_cell_height = grid_cell_height
        self.data_base = db

        self.images = {}

    def build(self):
        field_data = self.data_base[self.data_base['fieldId'] == self.id]

        for index, row in field_data.iterrows():

            image_weed_boxes = []
            image_weeds_data = json.loads(row["weeds"])

            for weed in image_weeds_data:
                box = Box(weed)
                image_weed_boxes.append(box)

            grid = Grid(self.grid_cell_width, self.grid_cell_height, IMAGE_WIDTH_PIXELS, IMAGE_HEIGHT_PIXELS).build()
            image_obj = Image(row['id'], image_weed_boxes, grid)
            self.images.update({image_obj.id: image_obj})
        return self

    def get_field_sprayed_area(self):
        sprayed_area = 0
        for image in self.images.values():
            sprayed_area += image.get_image_sprayed_area()

        return sprayed_area

    def calculate_field_spraying_percent(self):
        sprayed_area = self.get_field_sprayed_area()

        field_area = 0
        for image in self.images.values():
            field_area += image.grid.width_pixels * image.grid.height_pixels

        return 100 * (sprayed_area / field_area)


# @profile
def build_all_cell_sizes_of_field(_field_id):
    mm_per_pixel = FIELDS_MM_PER_PIXEL[_field_id]
    widths = heights = [i for i in range(5, 105, 5)]

    all_sizes = {}

    for cell_height_cm in heights:
        cell_height_pixels = convert_cm_2_pixel(cell_height_cm, mm_per_pixel)
        fields_with_same_cell_width = []
        for cell_width_cm in widths:
            cell_width_pixels = convert_cm_2_pixel(cell_width_cm, mm_per_pixel)
            field = Field(_field_id, cell_width_pixels, cell_height_pixels, data_base).build()
            fields_with_same_cell_width.append(field)
        all_sizes[cell_height_cm] = fields_with_same_cell_width
    return all_sizes


def get_image_percentss(_field_id, image_id):
    all_fields_sizes = fields_dict[_field_id]['all_sizes']
    percentss = {}
    for height, fields_with_same_height_cells in all_fields_sizes.items():
        percents = []
        for field in fields_with_same_height_cells:
            image = list(field.images.values())[image_id]
            percent = image.calculate_image_spraying_present()
            percents.append(percent)

        percentss.update({height: percents})
    return percentss


def get_field_percentss(_field_id, all_sizes):
    all_fields_sizes = all_sizes
    percentss = {}
    for height, fields_with_same_height_cells in all_fields_sizes.items():
        percents = []
        for field in fields_with_same_height_cells:
            percent = field.calculate_field_spraying_percent()
            percents.append(percent)

        percentss.update({height: percents})

    return percentss


def get_title(_field_id, image_id=None):
    if image_id:
        return 'spraying percent - field {} , image {}'.format(_field_id, image_id)
    else:
        return 'spraying percent - field {}'.format(_field_id)


def find_best_image_examples(_field_id):
    return 12, 8


def plot(_field_id, image_id=None):
    # Create a figure with customized size
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    gs = fig.add_gridspec(6, 6)

    # ***************** 2D CHART **********************************************

    chart_2d = fig.add_subplot(gs[:-2, :-2])

    # for each height, plot the chart of percents, with error bar of STDV
    percentss = fields_dict[_field_id]['percentss']
    for height, percents in percentss.items():
        if height % 10 in [0, 1]:
            stdv = round(float(np.std(percents, axis=0, dtype=np.float64)), 2)
            chart_2d.plot([i for i in range(5, 105, 5)], percents, label="height:{} (stdv:{})".format(height, stdv))

    # plot chart properties:
    chart_2d.set_xlabel('cell width', fontsize=14)
    chart_2d.set_ylabel('percent', fontsize=18)
    xaxis_ticks = [i for i in range(5, 105, 5)]
    yaxis_ticks = [i for i in range(5, 105, 5)]
    chart_2d.set_xticks(xaxis_ticks)
    chart_2d.set_yticks(yaxis_ticks)
    chart_2d.legend(ncol=1, loc='upper left', title='cell height', fontsize='large')
    chart_2d.grid()

    # ********************* 3D CHART  *******************************************
    chart_3d = fig.add_subplot(gs[:2, -2:], projection='3d')
    # chart_3d.set_title('3D')

    x = y = [i for i in range(5, 105, 5)]
    x, y = np.meshgrid(x, y)
    z = np.array(list(percentss.values()))
    # Plot the surface.
    surf = chart_3d.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap='viridis')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # configure axes
    chart_3d.set_xlim3d(0, 100)
    chart_3d.set_ylim3d(0, 100)
    chart_3d.set_zlim3d(0, 100)
    chart_3d.set_xlabel("width")
    chart_3d.set_ylabel("height")
    chart_3d.set_zlabel("percents")

    # ******************** symmetry table ***************************************
    symmetry_table = fig.add_subplot(gs[-2:, :2])
    symmetry_table.set_title('Symmetry table', fontsize=20)
    sym_indexes = [20, 40, 60, 80, 100]
    table_data = [[round(x) for x in row][3::4] for row in list(percentss.values())][3::4]  # only 20,40,60,...
    cols = ["h:{}".format(x) for x in list(percentss.keys())[3::4]]  # only 20,40,60,...
    rows = ["w:{}".format(x) for x in list(percentss.keys())[3::4]]  # only 20,40,60,...
    colors = [["w", "tab:orange", "tab:blue", "tab:green", "tab:red"],  # 1
              ["tab:orange", "w", "tab:purple", "tab:brown", "tab:pink"],  # 2
              ["tab:blue", "tab:purple", "w", "tab:gray", "tab:olive"],  # 3
              ["tab:green", "tab:brown", "tab:gray", "w", "tab:cyan"],  # 4
              ["tab:red", "tab:pink", "tab:olive", "tab:cyan", "w"]]  # 5

    the_table = symmetry_table.table(cellText=table_data, colLabels=cols, cellColours=colors, rowLabels=rows,
                                     loc='center')
    symmetry_table.axis('off')
    the_table.scale(1, 3)

    # ****************** image examples ****************************************

    image_example1 = fig.add_subplot(gs[-2:, 2:4])
    image_example2 = fig.add_subplot(gs[-2:, 4:])

    image_example1_id, image_example2_id = find_best_image_examples(_field_id)
    image_examples = [[image_example1, image_example1_id], [image_example2, image_example2_id]]

    for image_example in image_examples:
        image_example[0].set_title(get_title(_field_id, image_example[1]), fontsize=14)

        image_percentss = get_image_percentss(_field_id, image_example[1])

        for height, percents in image_percentss.items():
            if height % 10 in [0, 1]:
                stdv = round(float(np.std(percents, axis=0, dtype=np.float64)), 1)
                image_example[0].plot([i for i in range(5, 105, 5)], percents, label="{}({})".format(height, stdv))

        image_example[0].set_xticks(xaxis_ticks)
        image_example[0].set_yticks(yaxis_ticks)
        image_example[0].legend(ncol=1, loc='upper left', title='cell height', fontsize='xx-small')
        image_example[0].grid()

    # Save image (naming depends if it is a whole field or one image)
    fig.suptitle("{} - {}".format(get_title(_field_id), fields_dict[_field_id]['name']), fontsize=22)
    plt.savefig((get_title(_field_id, image_id)))

    # Show the image, and clear
    plt.tight_layout()
    plt.show()
    plt.clf()


def get_image_pixels(_field_id):
    field_data_entry = data_base[data_base['fieldId'] == _field_id]
    return field_data_entry['width'].iloc[0], field_data_entry['height'].iloc[0]


def main():
    for field_id in list(FIELDS_MM_PER_PIXEL.keys()):
        plot(field_id)


def build_all_fields_with_all_sizes():
    fields_dict_ = {}
    for field_id, mm_per_pixel in list(FIELDS_MM_PER_PIXEL.items()):
        all_sizes = build_all_cell_sizes_of_field(field_id)
        percentss = get_field_percentss(field_id, all_sizes)
        fields_dict_[field_id] = {'mm_per_pixel': mm_per_pixel, 'image_pixels': get_image_pixels(field_id),
                                  'all_sizes': all_sizes, 'percentss': percentss, 'name': FIELDS_NAMES[field_id]}

    return fields_dict_


if __name__ == "__main__":
    # data_base = pd.read_csv('db/farmer_tool_prod_fieldphotos_with_headers.csv')
    data_base = pd.read_csv('db/farmer_tool_prod_fieldphotos_with_headers.csv')
    fields_dict = build_all_fields_with_all_sizes()
    main()
