from PIL import Image, ImageDraw

if __name__ == '__main__':
    # height = 1000
    height = 3648
    width = 4864
    image = Image.new(mode='L', size=(width,height), color=255)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = 249

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=0,width=3)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=0, width=3)
    #data = [{"x": 1421, "y": 2947, "name": "Amaranthus", "genus": "Amaranthus", "width": 79, "family": "Amaranthaceae", "height": 152, "species": 'null', "leafOrder": "Broad leaf", "customProperties": {"age": "2"}}]
    # data = [{"x": 3174, "y": 526, "name": "Rock fist", "genus": 'null', "width": 446, "family": 'null', "height": 283, "species": 'null', "leafOrder": 'null', "customProperties": {"age": "1"}}]
    # data = [{"x": 1353, "y": 106, "name": "Amaranthus", "genus": "Amaranthus", "width": 36, "family": "Amaranthaceae", "height": 68, "species": 'null', "leafOrder": "Broad leaf", "customProperties": {"age": "2"}}]
    data = [{"x": 2238, "y": 2980, "name": "Object", "genus": 'null', "width": 140, "family": 'null', "height": 82, "species": 'null', "leafOrder": 'null', "customProperties": {"age": "4"}}, {"x": 3303, "y": 1395, "name": "Ambrosia spp.", "genus": "Ambrosia", "width": 31, "family": "Compositae", "height": 45, "species": "Ambrosia spp.", "leafOrder": "Broad leaf", "customProperties": {"age": "4"}}, {"x": 4027, "y": 917, "name": "Ambrosia spp.", "genus": "Ambrosia", "width": 86, "family": "Compositae", "height": 102, "species": "Ambrosia spp.", "leafOrder": "Broad leaf", "customProperties": {"age": "4"}}]
    for box in data:
        draw.rectangle((box['x'],box['y'], box['x'] + box['width'], box['y'] + box['height']), fill=128, outline=10)

    del draw
    image.save('grid.jpg')
    image.show()
