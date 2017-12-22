from PIL import Image


def img(image, rename):
    im = Image.open(image)
    w, h = im.size
    print('原图尺寸: %sx%s' % (w, h))
    im.thumbnail((w // 2, h // 2))
    print('处理后大小: %sx%s' % (w // 2, h // 2))
    im.save("new/" + rename + '_.png', 'png')


color_count = {}


def get_image_max_color(image):
    im = Image.open(image)
    width, height = im.size
    for h in range(height):
        for w in range(width):
            a, r, g, b = im.getpixel((w, h))
            color = format_color_int(a, r, g, b)
            if color in color_count:
                color_count[color] += 1
            else:
                color_count[color] = 0

    return sorted(color_count, key=lambda x: color_count[x])[-1]


def get_image_info(image):
    max_color = get_image_max_color(image)
    im = Image.open(image)
    width, height = im.size
    img_info = []
    for h in range(height):
        for w in range(width):
            a, r, g, b = im.getpixel((w, h))
            color = format_color_int(a, r, g, b)
            if color == max_color:
                img_info.append(0)
            else:
                img_info.append(1)
    return img_info


def format_color_int(a, r, g, b):
    return format_color_single_int(a) + format_color_single_int(r) + \
           format_color_single_int(g) + format_color_single_int(b)


def format_color_single_int(no):
    if no < 10:
        return '0' + '0' + str(no)
    if no < 100:
        return '0' + str(no)
    else:
        return str(no)


if __name__ == '__main__':
    get_image_info('/home/yueguang/develop/project/PythonProjects/machine_learning_studio/train_img/1/1513925673.png')
