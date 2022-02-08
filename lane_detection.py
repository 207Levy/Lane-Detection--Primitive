import numpy as np
import cv2


def makemask():
    vers = np.array([[300, 680], [550, 555], [720, 555], [1100, 680]], dtype=np.int32)
    vers.reshape(-1, 1)
    mask_crop = np.zeros((720, 1280), dtype=np.uint8)
    cv2.fillPoly(mask_crop, [vers], 255)
    return mask_crop


def load_save(name, vtype, fps):
    video_init = cv2.VideoCapture(name + '.' + vtype)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_init = cv2.VideoWriter('final.mp4', fourcc, fps, (1280, 720))
    return out_init, video_init


def filter_img(im, m):
    kernel = np.zeros(shape=(3, 3), dtype=np.uint8)
    kernel[:, 1] = 255
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_blr = cv2.GaussianBlur(im_gray, (3, 3), 0)
    im_mag = cv2.Canny(im_blr, 80, 150)
    crop = cv2.bitwise_and(m, im_mag)
    crop = cv2.dilate(crop, kernel=kernel, iterations=2)
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel, iterations=1)
    crop = cv2.Canny(crop, 80, 150)
    return crop


def write_lane(im, x, y,cos, sin, r, t, high, low, side):
    if side == 'right':
        x1 = int((r- 550*sin)/cos)
        y1 = 550
        x2 = int(x + low * (-sin))
        y2 = int(y + low * cos)
    else:
        x1 = int(x + low * (-sin))
        y1 = int(y + low * cos)
        x2 = int((r- 550*sin)/cos)
        y2 = 550

    im = cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
    return im

out, video = load_save('lane', 'mp4', 27.0)
mask = makemask()

if not video.isOpened():
    print("Error reading video")

while video.isOpened():
    ret, img = video.read()
    filtered_img = filter_img(img, mask)
    TH = 30
    lines = cv2.HoughLines(filtered_img, 1, np.pi / 180, TH)

    if lines is None:
        continue # if no line detected in frame

    # flags - for drawing one line for each lane and if crossing so no lines are drawn.
    cross = False
    found_l = False
    found_r = False
    for r_t in lines:
        rho = r_t[0, 0]
        theta = r_t[0, 1]
        cos = np.cos(theta)
        sin = np.sin(theta)
        x0 = cos * rho
        y0 = sin * rho
        if (found_r and found_l) or cross:
            if cross:
                last = 'cross'
            else:
                last = 'lane'
            break
        if (np.pi / 2 - 20 * np.pi / 180) <= theta <= (np.pi / 2 + 20 * np.pi / 180):
            continue
        if theta >= np.pi - 20 * np.pi / 180 or theta <= 0 + 20 * np.pi / 180:
            cross = True
            img = cv2.putText(img, "crossing lanes", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (100, 100, 255), 3)
            continue
        if np.pi / 2 < theta < np.pi - 20 * np.pi / 180:
            if not found_r:
                found_r = True
                img = write_lane(img, x0, y0, cos, sin, rho, theta, -850, -10000, 'right')
        elif np.pi / 2 > theta > 0 + 20 * np.pi / 180:
            if not found_l:
                found_l = True
                img = write_lane(img, x0, y0,cos, sin,rho, theta, -200, 1000, 'left')

    out.write(img)
video.release()
out.release()
cv2.destroyAllWindows()
