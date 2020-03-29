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
    21: 0.4,
    # }

    26: 0.5,
    27: 0.7,
    28: 0.5,
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
    173: 0.5}


# """


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
    return round(mm / mm_per_pixel, 2)


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
    def __init__(self, x, y, cell_height_pixels, cell_width_pixels):
        self.height_pixels = cell_height_pixels
        self.width_pixels = cell_width_pixels
        self.x = x
        self.y = y


class Grid:
    def __init__(self, cell_width_pixels, cell_height_pixels, grid_width_pixels, grid_height_pixels):
        self.base_cell_width = cell_width_pixels
        self.base_cell_height = cell_height_pixels
        self.grid_height = grid_height_pixels
        self.grid_width = grid_width_pixels
        self.right_edge_cell_width = grid_width_pixels % cell_width_pixels
        self.bottom_edge_cell_height = grid_height_pixels % cell_height_pixels
        self.cells = self.build_cells()

    def build_cells(self):
        cells = {}

        is_there_special_right_edges = self.right_edge_cell_width != 0
        is_there_special_bottom_edges = self.bottom_edge_cell_height != 0

        amount_of_cells = [math.ceil(self.grid_width / self.base_cell_width),
                           math.ceil(self.grid_height / self.base_cell_height)]

        for x in range(0, amount_of_cells[0]):
            _width = self.base_cell_width
            _height = self.base_cell_height

            if is_there_special_right_edges and (x == amount_of_cells[0] - 1):
                _width = self.right_edge_cell_width
            for y in range(0, amount_of_cells[1]):
                if is_there_special_bottom_edges and (y == amount_of_cells[1] - 1):
                    _height = self.bottom_edge_cell_height

                cell = Cell(x=x * self.base_cell_width, y=y * self.base_cell_height, cell_width_pixels=_width,
                            cell_height_pixels=_height)
                key = (round(cell.x, 2), round(cell.y, 2))
                cells[key] = cell

        return cells

    def find_grid_cells_intersect(self, weed_box):
        grid_cell_width = self.base_cell_width
        grid_cell_height = self.base_cell_height

        x_start = weed_box.x1 - (weed_box.x1 % grid_cell_width)
        x_end = weed_box.x1 + weed_box.width - ((weed_box.x1 + weed_box.width) % grid_cell_width)
        x_coordinates = np.around(np.arange(x_start, x_end + 1, grid_cell_width), 2)

        y_start = weed_box.y1 - (weed_box.y1 % grid_cell_height)
        y_end = weed_box.y1 + weed_box.height - ((weed_box.y1 + weed_box.height) % grid_cell_height)
        y_coordinates = np.around(np.arange(y_start, y_end + 1, grid_cell_height), 2)

        return {(x, y): self.cells.get((x, y)) for x in x_coordinates for y in y_coordinates}

    # def calculate_grid_spraying_present(self, weed_boxes):
    #     sprayed_area = self.get_grid_sprayed_area(weed_boxes)
    #     image_area = self.base_cell_height * self.base_cell_width
    #     return 100 * (sprayed_area / image_area)

    def _is_valid_weed_box(self, weed_box):
        if -10 < weed_box.x1 < 0:
            weed_box.x1 = 0
        if -10 < weed_box.y1 < 0:
            weed_box.y1 = 0
        if weed_box.x1 <= -10 or weed_box.y1 <= -10:
            return False
        if weed_box.x1 + weed_box.width > self.grid_width:
            return False
        if weed_box.y1 + weed_box.height > self.grid_height:
            return False
        return True

    def get_sprayed_area(self, weed_boxes):
        sprayed_cells = {}
        for weed_box in weed_boxes:
            if self._is_valid_weed_box(weed_box):
                sprayed_cells.update(self.find_grid_cells_intersect(weed_box))

        sprayed_area = 0
        for cell in sprayed_cells.values():
            sprayed_area += cell.width_pixels * cell.height_pixels

        return sprayed_area


class Image:
    def __init__(self, image_id, weed_boxes, image_width_pixels, image_height_pixels, mm_per_pixels):
        self.id = image_id
        self.weed_boxes = weed_boxes
        self.width_pixels = image_width_pixels
        self.height_pixels = image_height_pixels
        self.mm_per_pixels = mm_per_pixels
        self.grids = self.build_grids()
        self.spread_area_of_all_grids = self._get_spread_area_of_all_grids_of_image()

    def build_grids(self):
        """
        :returns:
        dict of dicts : {5(h): {5(w): GRID(5x5), 10(w): GRID(10x5), ... , 100(w): GRID(100x5)},
                         10(h): {5(w): GRID(5x10), 10(w): GRID(10x10), ... , 100(w): GRID(100x10)},
                         ...
                         ...
                         ...
                         100(h): {5(w): GRID(5x100), 10(w): GRID(10x100), ... , 100(w): GRID(100x100)}
                       }

        * when GRID(a1xa2) is grid with base cell with a1 width pixels and a2 height pixels
        """
        widths = heights = [(cm, convert_cm_2_pixel(cm, self.mm_per_pixels)) for cm in range(5, 105, 5)]

        return {
            h_cm: {w_cm: Grid(w_pix, h_pix, self.width_pixels, self.height_pixels)
                   for w_cm, w_pix in widths}
            for h_cm, h_pix in heights
        }

    def _get_spread_area_of_all_grids_of_image(self):
        """

        :return: {5(h): {5(w): number(area), 10(w): number(area), ... , 100(w): number(area)},
                  10(h): {5(w): number(area), 10(w): number(area), ... , 100(w): number(area)},
                  ...
                  ...
                  ...
                  100(h): {5(w): number(area), 10(w): number(area), ... , 100(w): number(area)}
                }
        """
        return {h: {w: grid.get_sprayed_area(self.weed_boxes) for w, grid in grids_with_same_height.items()}
                for h, grids_with_same_height in self.grids.items()}

    def get_image_percentss(self):
        return {height: [round(100 * (spread_area / (self.width_pixels * self.height_pixels)), 2)
                         for spread_area in areas_of_same_height.values()]
                for height, areas_of_same_height in self.spread_area_of_all_grids.items()}


class Field:
    def __init__(self, _field_id, mm_per_pixels):
        self.id = _field_id
        self.mm_per_pixels = mm_per_pixels
        self.images = self._build_images()
        self.area = self._get_field_area()
        self.spread_areas = self._get_field_sprayed_area()
        self.percentss = self._get_percentss()

    def _build_images(self):
        """
        images is in the shape: {image_id1: Image1, image_id2: Image2}
        :return:
        """
        field_data = data_base[data_base['fieldId'] == self.id]
        image_width_pixels, image_height_pixels = self._get_image_pixels()

        images = {}
        for index, row in field_data.iterrows():
            images[row['id']] = Image(row['id'], [Box(weed) for weed in json.loads(row["weeds"])], image_width_pixels,
                                      image_height_pixels, self.mm_per_pixels)

        # images = {row['id']: Image(row['id'],
        #                            [Box(weed) for weed in json.loads(row["weeds"])],
        #                            image_width_pixels, image_height_pixels,
        #                            self.mm_per_pixels)
        #           for index, row in field_data.iterrows()}
        return images

    def _get_image_pixels(self):
        field_data_entry = data_base[data_base['fieldId'] == self.id]
        return field_data_entry['width'].iloc[0], field_data_entry['height'].iloc[0]

    def _get_field_sprayed_area(self):
        """
        merges the spreads areas (of all grids) of all images

        :return:
        { 5(h): {5(w): number(area(number), 10(w): number(area(number), ... , 100(w): number(area(number)},
         10(h): {5(w): number(area(number), 10(w): number(area(number), ... , 100(w): number(area(number)},
         ...
         ...
         ...
         100(h): {5(w): number(area(number), 10(w): number(area(number), ... , 100(w): number(area(number)}
        }
        """
        field_spread_area = list(self.images.values())[0].spread_area_of_all_grids

        for image in list(self.images.values())[1:]:
            for height, areas_of_same_heights in image.spread_area_of_all_grids.items():
                for width, area in areas_of_same_heights.items():
                    field_spread_area[height][width] += area

        return field_spread_area

    def _get_percentss(self):
        return {height: [round(100 * (spread_area / self.area), 2)
                         for spread_area in areas_of_same_height.values()]
                for height, areas_of_same_height in self.spread_areas.items()}

    def _get_field_area(self):
        area = 0
        for image in self.images.values():
            area += image.width_pixels * image.height_pixels
        return area


#
# def get_image_percentss(_field_id, image_id):
#     all_fields_sizes = fields_dict[_field_id]['all_sizes']
#     percentss = {}
#     for height, fields_with_same_height_cells in all_fields_sizes.items():
#         percents = []
#         for field in fields_with_same_height_cells:
#             image = list(field.images.values())[image_id]
#             percent = image.calculate_image_spraying_present()
#             percents.append(percent)
#
#         percentss.update({height: percents})
#     return percentss

#
# def get_field_percentss(_field_id, all_sizes):
#     all_fields_sizes = all_sizes
#     percentss = {}
#     for height, all_same_height_cells_of_field in all_fields_sizes.items():
#         percents = []
#         for field in all_same_height_cells_of_field:
#             percent = field.calculate_field_spraying_percent()
#             percents.append(percent)
#
#         percentss.update({height: percents})
#
#     return percentss


def get_title(_field_id, image_id=None):
    if image_id:
        return 'spraying percent - field {} , image {}'.format(_field_id, image_id)
    else:
        return 'spraying percent - field {}'.format(_field_id)


def find_best_image_examples(_field_id):
    return 12, 8


def plot(_field_id, _image_id=None):
    # Create a figure with customized size
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    gs = fig.add_gridspec(6, 6)
    widths = heights = [i for i in range(5, 105, 5)]

    # ***************** 2D CHART **********************************************

    chart_2d = fig.add_subplot(gs[:-2, :-2])

    # for each height, plot the chart of percents, with error bar of STDV
    percentss = fields_dict[_field_id]['percentss']
    for height, percents in percentss.items():
        if height % 10 in [0, 1]:  # plot only the lines with height 10,20,30...
            stdv = round(float(np.std(percents, axis=0, dtype=np.float64)), 2)
            chart_2d.plot(widths, percents, label="height:{} (stdv:{})".format(height, stdv))

    # plot the mean:
    chart_2d.plot(widths, fields_dict[_field_id]['mean'], label='mean of field', lw=2, color='k', ls='--')
    # plot mean of means:
    chart_2d.plot(widths, fields_dict['mean_of_means'], label='mean of all fields', lw=2, color='gray', ls='--')

    # plot chart properties:
    chart_2d.set_xlabel('cell width', fontsize=14)
    chart_2d.set_ylabel('percent', fontsize=18)
    chart_2d.set_xticks(widths)
    chart_2d.set_yticks(heights)
    chart_2d.legend(ncol=1, loc='upper left', title='cell height', fontsize='large')
    chart_2d.grid()

    # ********************* 3D CHART  *******************************************
    chart_3d = fig.add_subplot(gs[:2, -2:], projection='3d')

    # Plot the surface:
    x, y = np.meshgrid(widths, heights)
    z = np.array(list(percentss.values()))
    surf = chart_3d.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap='viridis')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # configure axes
    chart_3d.set_xlabel("width")
    chart_3d.set_xlim3d(0, 100)
    chart_3d.set_ylabel("height")
    chart_3d.set_ylim3d(0, 100)
    chart_3d.set_zlabel("percents")
    chart_3d.set_zlim3d(0, 100)

    # ******************** symmetry table ***************************************
    symmetry_table = fig.add_subplot(gs[-2:, :2])
    symmetry_table.set_title('Symmetry table', fontsize=20)
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

    for image_subplot, _id in image_examples:
        image_subplot.set_title(get_title(_field_id, _id), fontsize=14)

        image_percentss = list(fields_dict[_field_id]['field'].images.values())[_id].get_image_percentss()

        for height, percents in image_percentss.items():
            if height % 10 in [0, 1]:
                stdv = round(float(np.std(percents, axis=0, dtype=np.float64)), 1)
                image_subplot.plot([i for i in range(5, 105, 5)], percents, label="{}({})".format(height, stdv))

        image_subplot.set_xticks(widths)
        image_subplot.set_yticks(heights)
        image_subplot.legend(ncol=1, loc='upper left', title='cell height', fontsize='xx-small')
        image_subplot.grid()

    # Save image (naming depends if it is a whole field or one image)
    fig.suptitle("{} - {}".format(get_title(_field_id), fields_dict[_field_id]['name']), fontsize=22)
    plt.savefig((get_title(_field_id, _image_id)))

    # Show the image, and clear
    # plt.tight_layout()
    plt.show()
    plt.clf()


# def plot_all_fields():


def get_field_mean_line(percents):
    return np.array([row for row in percents.values()]).mean(0)


@profile
def build_all_fields_with_all_sizes():
    fields_dict_ = {}
    for field_id, mm_per_pixel in list(FIELDS_MM_PER_PIXEL.items()):
        # field = build_all_instances_of_field(field_id)
        field = Field(field_id, mm_per_pixel)
        mean = get_field_mean_line(field.percentss)
        fields_dict_[field_id] = {'mm_per_pixel': mm_per_pixel, 'field': field, 'percentss': field.percentss,
                                  'mean': mean, 'name': FIELDS_NAMES[field_id]}

    # calculate mean of means:
    fields_dict_['mean_of_means'] = np.mean([row['mean'] for row in fields_dict_.values()], 0)

    return fields_dict_


def main():
    for field_id in list(FIELDS_MM_PER_PIXEL.keys()):
        plot(field_id)


if __name__ == "__main__":
    data_base = pd.read_csv('db/farmer_tool_prod_fieldphotos_with_headers.csv')
    fields_dict = build_all_fields_with_all_sizes()
    # for id, fields in fields_dict.items():
    #     if isinstance(id, int):
    #         print("*********************** field id: {} - {} ***************************".format(id, fields['name']))
    #         for image_id, image in fields['field'].images.items():
    #             print(" ------------- image id: {} ---------------".format(image_id))
    #             for h, grids in image.grids.items():
    #                 print("{}H - {}".format(h, [len(grid.cells) for grid in grids.values()]))
    #         # for height, instances in fields['all_sizes'].items():
    #         #     for instance in instances:
    #         #         for image in instance.images:
    #         #             print(len(image.grid.cells.vealues()))
    #         # print("num of grid cells: {}".format(
    #         #     [len(image.grid.cells.values()) for instances in fields['all_sizes'].values()
    #         for instance in instances
    #         #      for image in instance.images.values()]))

main()
