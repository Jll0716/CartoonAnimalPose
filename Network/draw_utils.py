import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image

# COCO 17 points
point_name = ["L_eye", "R_eye", "nose", "L_ear", "R_ear",
              "L_F_elbow", "R_F_elbow", "L_B_elbow", "R_B_elbow",
              "L_F_knee", "R_F_knee", "L_B_knee", "R_B_knee",
              "L_F_paw", "R_F_paw", "L_B_paw", "R_B_paw",
              "throat", "withers", "tail", "torso"]
'''
"nose", "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"
'''
point_color = [(203, 192, 255), (255, 0, 255), (60, 20, 220), (203, 192, 255), (255, 0, 255),
               (130, 0, 75), (237, 149, 100), (255, 255, 0),
               (144, 238, 144), (226, 43, 138), (225, 105, 65),
               (139, 139, 0), (143, 188, 143), (139, 61, 72),
               (255, 0, 0), (79, 79, 47), (34, 139, 34), (0, 255, 255),
               (107, 183, 189), (0, 215, 255), (0, 165, 255)]
# 0-4，脸部，红色；粉红、洋红、猩红、粉红、洋红
# 5,9,13，左前肢，紫色；靛青、深紫罗兰的蓝色、深岩暗蓝灰色
# 6,10,14,右前肢，蓝色；矢车菊的蓝色、皇军蓝、纯蓝
# 7,11,15,左后肢，青色；青色、深青色、深石板灰
# 8,12,16,右后肢，绿色；淡绿色、深海洋绿、森林绿
# 17,脖子/喉咙，黄色
# 18，脊柱，深卡其布
# 19，尾巴，金色
# 20，躯干，橙色
'''
(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)
'''

def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 2,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    return img
