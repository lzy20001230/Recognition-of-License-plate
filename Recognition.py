import cv2
import numpy as np

template_real = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '藏','川','鄂','贵','黑','吉','冀','津','晋','京','辽','鲁','闽',
            '琼','陕','苏','皖','湘','豫','粤','云','浙']
template = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '35','36','37','38','39','40','41','42','43','44','45','46','47',
            '48','49','50','51','52','53','54','55','56']


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def License_plate_location(filename):
    # 读取原图
    img = cv2.imread(filename)
    #cv_show("img",img)

    # 灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv_show("img_gray",img_gray)

    # 高斯
    img_gs = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # cv_show("img_gs",img_gs)

    # sobel算子
    img_Sobel = cv2.Sobel(img_gs, -1, 1, 0, 1)
    # cv_show("img_Sobel",img_Sobel)

    # 图像二值化
    ret, img_binary = cv2.threshold(img_Sobel, 0, 225, cv2.THRESH_OTSU)
    #cv_show("img_binary",img_binary)

    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernelX)
    # cv_show("img_close",img_close)
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernelX)
    #cv_show("img_open", img_open)

    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_dilation = cv2.dilate(img_open, element, iterations=3)
    #cv_show("img_dilation",img_dilation)

    # 获取轮廓
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 测试边框识别结果
    img_draw = img.copy()
    img_draw = cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 3)
    #cv_show("img_draw", img_draw)
    #print(len(contours))

    # 通过长宽比缩小范围
    rect = []
    img_rect = img.copy()
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        radio = w / float(h)
        if radio > 2.2 and radio < 4:
            img_rect = cv2.rectangle(img_rect, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rect.append((x, y, w, h))
    #cv_show("img_rect",img_rect)

    # 用颜色识别出车牌区域
    dist_r = []
    max_mean = 0
    for r in rect:
        block = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        result = cv2.inRange(hsv, lower_blue, upper_blue)
        # 用计算均值的方式找蓝色最多的区块
        mean = cv2.mean(result)
        if mean[0] > max_mean:
            max_mean = mean[0]
            dist_r = r
    img_license_rect = cv2.rectangle(img, (dist_r[0], dist_r[1]), (dist_r[0] + dist_r[2], dist_r[1] + dist_r[3]), (255, 0, 0), 2)
    #cv_show("img_license_rect",img_license_rect)
    img_license = img[dist_r[1] + 5:dist_r[1] + dist_r[3] - 5, dist_r[0] + 5:dist_r[0] + dist_r[2] - 5]
    # img_license = img[dist_r[1] :dist_r[1] + dist_r[3] , dist_r[0] :dist_r[0] + dist_r[2] ]
    img_license = cv2.resize(img_license, None, fx=2, fy=2)
    #cv_show("img_license", img_license)
    return img_license


def Character_segmentation(img):
    # 高斯
    img_gs = cv2.GaussianBlur(img, (3, 3), 0)
    # cv_show("img_gs",img_gs)

    # 灰度图
    img_gray = cv2.cvtColor(img_gs, cv2.COLOR_RGB2GRAY)
    #cv_show("img_gray",img_gray)

    # 图像二值化
    ret, img_binary = cv2.threshold(img_gray, 0, 225, cv2.THRESH_OTSU)
    #cv_show("img_binary", img_binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_dilation = cv2.dilate(img_binary, kernel)
    #cv_show("img_dilation",img_dilation)
    # 获取轮廓
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 测试边框识别结果
    img_draw = img.copy()
    img_draw = cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 3)
    #cv_show("img_draw", img_draw)
    #print(len(contours))

    # 通过长宽比缩小范围
    rect = []
    img_rect = img.copy()
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        radio = h / float(w)
        if radio > 1.7 and radio < 2.8:
            img_rect = cv2.rectangle(img_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)
            rect.append((x, y, w, h))
        if radio > 5 and radio < 8:
            img_rect = cv2.rectangle(img_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)
            rect.append((x, y, w, h))
    rect = sorted(rect, key=lambda s: s[0], reverse=False)
    #cv_show("img_rect", img_rect)
    # print(rect)
    rect_imgs = []
    for (x, y, w, h) in rect:
        img_s = img[y:y + h, x: x + w]
        rect_imgs.append(img_s)

    return rect_imgs


def Character_recognition(word_imgs):
    results = []
    first_words_list = []
    mid_words_list = []
    last_words_list = []
    for i in range(34, 56):
        str1 = 'refer/'
        str2 = template[i]
        str3 = '.jpg'
        filename = str1 + str2 + str3
        word = cv2.imread(filename)
        first_words_list.append((i, word))
    for i in range(0, 34):
        str1 = 'refer/'
        str2 = template[i]
        str3 = '.jpg'
        filename = str1 + str2 + str3
        word = cv2.imread(filename)
        last_words_list.append((i, word))
    for i in range(10, 34):
        str1 = 'refer/'
        str2 = template[i]
        str3 = '.jpg'
        filename = str1 + str2 + str3
        word = cv2.imread(filename)
        mid_words_list.append((i, word))

    for index, word_img in enumerate(word_imgs):
        #cv_show("word_img", word_img)
        word_img_gs = cv2.GaussianBlur(word_img, (3, 3), 0)
        word_img_gray = cv2.cvtColor(word_img_gs, cv2.COLOR_BGR2GRAY)
        ret, word_img_binary = cv2.threshold(word_img_gray, 0, 225, cv2.THRESH_OTSU)
        #cv_show("img_binary", word_img_binary)
        # word_img=cv2.cvtColor(word_img,cv2.COLOR_BGR2GRAY)

        if index == 0:
            scores = []
            for (i, word) in first_words_list:
                # word_img = cv2.imdecode(np.fromfile(word_img, dtype=np.uint8), 1)
                word = cv2.cvtColor(word, cv2.COLOR_BGR2RGB)
                word_img_copy = cv2.resize(word_img_binary, (20, 40))
                word_img_copy = cv2.cvtColor(word_img_copy, cv2.COLOR_BGR2RGB)
                result = cv2.matchTemplate(word_img_copy, word, cv2.TM_CCOEFF) #2
                # print(result[0][0])
                scores.append((i, result[0][0]))
            # print(scores)
            scores = sorted(scores, key=lambda s: s[1], reverse=True)
            # print(scores)
            i = scores[0][0]
            #print(template[i])
            results.append(template_real[i])

        if index == 1:
            scores = []
            for (i, word) in mid_words_list:
                # word_img = cv2.imdecode(np.fromfile(word_img, dtype=np.uint8), 1)
                word = cv2.cvtColor(word, cv2.COLOR_BGR2RGB)
                word_img_copy = cv2.resize(word_img_binary, (20, 40))
                word_img_copy = cv2.cvtColor(word_img_copy, cv2.COLOR_BGR2RGB)
                result = cv2.matchTemplate(word_img_copy, word, cv2.TM_CCOEFF) #2
                # print(result[0][0])
                scores.append((i, result[0][0]))
            # print(scores)
            scores = sorted(scores, key=lambda s: s[1], reverse=True)
            # print(scores)
            i = scores[0][0]
            #print(template[i])
            results.append(template[i])

        if index > 1 and index < 7:
            scores = []
            for (i, word) in last_words_list:
                # word_img = cv2.imdecode(np.fromfile(word_img, dtype=np.uint8), 1)
                word = cv2.cvtColor(word, cv2.COLOR_BGR2RGB)
                word_img_copy = cv2.resize(word_img_binary, (20, 40))
                word_img_copy = cv2.cvtColor(word_img_copy, cv2.COLOR_BGR2RGB)
                result = cv2.matchTemplate(word_img_copy, word, cv2.TM_CCOEFF) #2
                # print(result[0][0])
                scores.append((i, result[0][0]))
            # print(scores)
            scores = sorted(scores, key=lambda s: s[1], reverse=True)
            # print(scores)
            i = scores[0][0]
            #print(template[i])
            results.append(template[i])
    return results



i=0
for i in range(0,3):
    str1="img/car"
    print(str1+str(i + 1) + ".jpg")
    license = License_plate_location(str1+str(i + 1) + ".jpg")
    # cv_show("license",license)
    word_imgs = Character_segmentation(license)
    words = Character_recognition(word_imgs)
    print(words)


