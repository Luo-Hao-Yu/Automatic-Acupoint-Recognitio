import os
import mediapipe as mp
import cv2 as cv
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
if __name__ == "__main__":
    # 图片文件夹路径
    folder_path = "images"  # 替换为你的文件夹路径

    # 保存预测图片的文件夹路径
    output_folder = "outPutimages"  # 替换为你的输出文件夹路径

    #保存人工点的穴位的文件夹路径
    outputPerson_folder="outPersonimages"

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义手检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    # 定义在图中写字的函数
    def cv2ImgAddText(img, text, left, top, textColor, textSize):
        textSize=30
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


    points = []
    # 回调函数，用于获取鼠标点击事件的像素坐标
    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # 在点击位置绘制一个小圆点
            cv.circle(resized_img, (x, y), 5, (0, 0, 255), -1)
            # 输出点击位置的像素坐标
            print("Pixel coordinates (x, y):", x, ",", y)
            # 将点击位置的像素坐标记录到列表中
            points.append((x, y))
            # 更新图像显示
            cv.imshow("Resized Image", resized_img)
            # 如果记录的点达到12个，则结束程序
            if len(points) == 12:
                output_path = os.path.join(outputPerson_folder, filename)
                cv.imwrite(output_path, resized_img)
                print("Saved prediction:", output_path)
                cv.destroyAllWindows()

    #误差列表
    errorList=[]

    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        # 判断文件是否是图片文件（这里简单地以文件名是否以 ".jpg"、".png" 等结尾来判断）
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # 构建完整的文件路径
            img_path = os.path.join(folder_path, filename)

            # 读取图像
            img = cv.imread(img_path)

            #显示并缩放图像
            scale_percent = 30  # 设置缩放比例
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
            # 显示缩放后的图像
            cv.imshow("Resized Image", resized_img)
            # 设置鼠标点击事件的回调函数
            cv.setMouseCallback("Resized Image", click_event)
            cv.waitKey(0)  # 等待按下任意键
            cv.destroyAllWindows()  # 关闭所有窗口

            if img is not None:
                image_height, image_width, _ = np.shape(img)
                # 转换为 RGB
                imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                # 得到检测结果
                results = hands.process(imgRGB)

                # 创建穴位坐标列表
                lmList = []

                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        for i in range(21):
                            # pos_x = hand.landmark[i].x * image_width
                            # pos_y = hand.landmark[i].y * image_height
                            # cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
                            lmList.append([i,int(hand.landmark[i].x * image_width),int(hand.landmark[i].y * image_height)]);

                # 计算手部穴位的函数
                # 手指在上
                if lmList[12][2] < lmList[0][2]:
                    if lmList[4][1] < lmList[20][1]:
                        print('手心')
                        cx11 , cy11 = lmList[11][1],lmList[11][2]
                        cx10,cy10 = lmList[10][1],lmList[10][2]
                        inch=((cx10-cx11)**2+(cy10-cy11)**2)**0.5
                        cx0,cy0 = lmList[0][1],lmList[0][2]
                        cx1 ,cy1 =  lmList[1][1],lmList[1][2]
                        img = cv2ImgAddText(img, "鱼际穴", cx1, cy1, (0,0,0), 30)
                        cv.circle(img, (cx1, cy1), 10, (255, 0, 255), cv.FILLED)
                        cx4,cy4 = lmList[4][1],lmList[4][2]
                        img = cv2ImgAddText(img, "十宣穴1", cx4, cy4, (0,0,0), 30)
                        cv.circle(img, (cx4, int(cy4-0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        cx6,cy6 = lmList[6][1],lmList[6][2]
                        img = cv2ImgAddText(img, "四缝穴1", cx6, cy6, (0,0,0), 30)
                        cv.circle(img, (cx6, cy6), 10, (255, 0, 255), cv.FILLED)
                        cx8,cy8 = lmList[8][1],lmList[8][2]
                        img = cv2ImgAddText(img, "十宣穴2", cx8, cy8, (0,0,0), 30)
                        cv.circle(img, (cx8, int(cy8-0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        cx5,cy5 = lmList[5][1],lmList[5][2]
                        cx9,cy9 = lmList[9][1],lmList[9][2]
                        cv.circle(img, ((3*cx9+cx0)//4, (3*cy9+cy0)//4), 10, (255, 0, 255), cv.FILLED)
                        img = cv2ImgAddText(img, "劳宫穴", (2*cx9+cx5+cx0)//4,(2*cy9+cy5+cy0)//4, (0,0,0), 30)
                        x_center,y_center=(cx9+cx0)//2, (cy9+cy0)//2

                        img = cv2ImgAddText(img, "四缝穴2", cx10, cy10, (0,0,0), 30)
                        cv.circle(img, (cx10, cy10), 10, (255, 0, 255), cv.FILLED)

                        cx12 , cy12 = lmList[12][1],lmList[12][2]
                        img = cv2ImgAddText(img, "十宣穴3", cx12, cy12, (0,0,0), 30)
                        cv.circle(img, (cx12, int(cy12-0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        # img = cv2ImgAddText(img, "中冲穴", (2*cx12+cx11)//3, (2*cy12+cy11)//3, (0,0,0), 30)
                        # cv.circle(img, ((2*cx12+cx11)//3, (2*cy12+cy11)//3), 10, (255, 0, 255), cv.FILLED)
                        cx14,cy14 = lmList[14][1],lmList[14][2]
                        img = cv2ImgAddText(img, "四缝穴3", cx14, cy14, (0,0,0), 30)
                        cv.circle(img, (cx14, cy14), 10, (255, 0, 255), cv.FILLED)
                        cx16,cy16 = lmList[16][1],lmList[16][2]
                        img = cv2ImgAddText(img, "十宣穴4", cx16, cy16, (0,0,0), 30)
                        cv.circle(img, (cx16, int(cy16-0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        cx17,cy17 = lmList[17][1],lmList[17][2]
                        img = cv2ImgAddText(img, "少府穴", (3*cx17+cx0)//4, (3*cy17+cy0)//4, (0,0,0), 30)
                        cv.circle(img, ((3*cx17+cx0)//4, (3*cy17+cy0)//4), 10, (255, 0, 255), cv.FILLED)
                        cx18,cy18 = lmList[18][1],lmList[18][2]
                        img = cv2ImgAddText(img, "四缝穴4", cx18, cy18, (0,0,0), 30)
                        cv.circle(img, (cx18, cy18), 10, (255, 0, 255), cv.FILLED)
                        cx20,cy20 = lmList[20][1],lmList[20][2]
                        img = cv2ImgAddText(img, "十宣穴5", cx20, cy20, (0,0,0), 30)
                        cv.circle(img, (cx20, int(cy20-0.1*inch)), 10, (255, 0, 255), cv.FILLED)

                        #预测穴位坐标
                        CX=[]
                        CX.append((cx1, cy1))
                        CX.append((cx4, int(cy4-0.1*inch)))
                        CX.append((cx6, cy6))
                        CX.append((cx8, int(cy8-0.1*inch)))
                        CX.append(((3*cx9+cx0)//4, (3*cy9+cy0)//4))
                        CX.append((cx10, cy10))
                        CX.append((cx12, int(cy12-0.1*inch)))
                        CX.append((cx14, cy14))
                        CX.append((cx16, int(cy16-0.1*inch)))
                        CX.append(((3*cx17+cx0)//4, (3*cy17+cy0)//4))
                        CX.append((cx18, cy18))
                        CX.append((cx20, int(cy20-0.1*inch)))

                        #误差计算
                        magnitude=[]
                        error=[]
                        for i,point in enumerate(points):
                            x, y=point
                            x=x*(10/3)
                            y=y*(10/3)
                            magnitude.append(math.sqrt(x**2 + y**2))
                            error.append(math.sqrt((x-CX[i][0])**2+(y-CX[i][1])**2))

                        sumError=0
                        for i in range(12):
                            sumError += error[i]/magnitude[i]
                        print(points)
                        print(CX)
                        print(error)
                        print(magnitude)
                        print(sumError/12)
                        errorList.append(sumError/12)
                        points = []




                        # 保存预测图片
                        output_path = os.path.join(output_folder, filename)
                        cv.imwrite(output_path, img)
                        print("Saved prediction:", output_path)

                    else:
                        print('手背')
                        cx0,cy0 = lmList[0][1],lmList[0][2]
                        cx1,cy1 = lmList[1][1],lmList[1][2]
                        x_zhongquan,y_zhongquan = (3*cx0+cx1)//4,(3*cy0+cy1)//4
                        img = cv2ImgAddText(img, "中泉穴", x_zhongquan,y_zhongquan, (0,0,0), 20)
                        cv.circle(img, (x_zhongquan, y_zhongquan), 10, (255, 0, 255), cv.FILLED)
                        cx2 ,cy2 =  lmList[2][1],lmList[2][2]
                        cx3 ,cy3 =  lmList[3][1],lmList[3][2]
                        img = cv2ImgAddText(img, "大骨空穴", cx3,cy3, (0,0,0), 20)
                        cv.circle(img, (cx3, cy3), 10, (255, 0, 255), cv.FILLED)
                        cx4,cy4 = lmList[4][1],lmList[4][2]
                        x_shaoshang,y_shaoshang = (cx3+cx4)//2,(cy3+cy4)//2
                        img = cv2ImgAddText(img, "少商穴", x_shaoshang,y_shaoshang, (0,0,0), 20)
                        cv.circle(img, (x_shaoshang, y_shaoshang), 10, (255, 0, 255), cv.FILLED)
                        img = cv2ImgAddText(img, "十宣穴1", cx4,cy4, (0,0,0), 20)
                        cv.circle(img, (cx4, cy4), 10, (255, 0, 255), cv.FILLED)
                        cx5,cy5 = lmList[5][1],lmList[5][2]
                        x_053,y_053 = (3*cx5+cx0)//4,(3*cy5+cy0)//4
                        x_05,y_05 = (cx5+cx0)//2,(cy5+cy0)//2
                        x_sanjian,y_sanjian = (x_053+cx2)//2,(y_053+cy2)//2
                        x_hegu,y_hegu = (x_05+cx2)//2,(y_05+cy2)//2
                        img = cv2ImgAddText(img, "三间穴", x_sanjian,y_sanjian, (0,0,0), 20)
                        cv.circle(img, (x_sanjian, y_sanjian), 10, (255, 0, 255), cv.FILLED)
                        img = cv2ImgAddText(img, "合谷穴", x_hegu,y_hegu, (0,0,0), 20)
                        cv.circle(img, (x_hegu, y_hegu), 10, (255, 0, 255), cv.FILLED)
                        x_baxie1,y_baxie1 = (cx2+cx5)//2,(cy2+cy5)//2
                        img = cv2ImgAddText(img, "八邪穴1", x_baxie1,y_baxie1, (0,0,0), 20)
                        cv.circle(img, (x_baxie1, y_baxie1), 10, (255, 0, 255), cv.FILLED)
                        x_yaotong1,y_yaotong1=(cx5+cx0)//2,(cy5+cy0)//2
                        img = cv2ImgAddText(img, "腰痛点1", x_yaotong1,y_yaotong1, (0,0,0), 20)
                        cv.circle(img, (x_yaotong1, y_yaotong1), 10, (255, 0, 255), cv.FILLED)
                        cx8,cy8 = lmList[8][1],lmList[8][2]
                        img = cv2ImgAddText(img, "十宣穴2", cx8,cy8, (0,0,0), 20)
                        cv.circle(img, (cx8, cy8), 10, (255, 0, 255), cv.FILLED)
                        cx9 ,cy9 =  lmList[9][1],lmList[9][2]
                        x_baxie2,y_baxie2 = (cx9+cx5)//2,(cy9+cy5)//2
                        img = cv2ImgAddText(img, "八邪穴2", x_baxie2,y_baxie2, (0,0,0), 20)
                        cv.circle(img, (x_baxie2, y_baxie2), 10, (255, 0, 255), cv.FILLED)
                        cx10 ,cy10 =  lmList[10][1],lmList[10][2]
                        img = cv2ImgAddText(img, "中魁穴", cx10,cy10, (0,0,0), 20)
                        cv.circle(img, (cx10, cy10), 10, (255, 0, 255), cv.FILLED)
                        cx12 ,cy12 =  lmList[12][1],lmList[12][2]
                        img = cv2ImgAddText(img, "十宣穴3", cx12,cy12, (0,0,0), 20)
                        cv.circle(img, (cx12, cy12), 10, (255, 0, 255), cv.FILLED)
                        cx13 ,cy13 =  lmList[13][1],lmList[13][2]
                        x_baxie3,y_baxie3 = (cx9+cx13)//2,(cy9+cy13)//2
                        img = cv2ImgAddText(img, "八邪穴3", x_baxie3,y_baxie3, (0,0,0), 20)
                        cv.circle(img, (x_baxie3, y_baxie3), 10, (255, 0, 255), cv.FILLED)
                        x_yaotong2,y_yaotong2 = (cx0+cx13)//2,(cy0+cy13)//2
                        img = cv2ImgAddText(img, "腰痛点2", x_yaotong2,y_yaotong2, (0,0,0), 20)
                        cv.circle(img, (x_yaotong2, y_yaotong2), 10, (255, 0, 255), cv.FILLED)
                        cx16,cy16 =  lmList[16][1],lmList[16][2]
                        img = cv2ImgAddText(img, "十宣穴4", cx16,cy16, (0,0,0), 20)
                        cv.circle(img, (cx16, cy16), 10, (255, 0, 255), cv.FILLED)
                        cx17,cy17 =  lmList[17][1],lmList[17][2]
                        img = cv2ImgAddText(img, "后溪穴", cx17,cy17, (0,0,0), 20)
                        cv.circle(img, (cx17, cy17), 10, (255, 0, 255), cv.FILLED)
                        x_baxie4,y_baxie4 = (cx13+cx17)//2,(cy13+cy17)//2
                        img = cv2ImgAddText(img, "八邪穴4", x_baxie4,y_baxie4, (0,0,0), 20)
                        cv.circle(img, (x_baxie4, y_baxie4), 10, (255, 0, 255), cv.FILLED)
                        cx18,cy18 =  lmList[18][1],lmList[18][2]
                        img = cv2ImgAddText(img, "小骨空穴", cx18,cy18, (0,0,0), 20)
                        cv.circle(img, (cx18, cy18), 10, (255, 0, 255), cv.FILLED)
                        cx20,cy20 =  lmList[20][1],lmList[20][2]
                        img = cv2ImgAddText(img, "十宣穴5", cx20,cy20, (0,0,0), 20)
                        cv.circle(img, (cx20, cy20), 10, (255, 0, 255), cv.FILLED)

                        # 保存预测图片
                        output_path = os.path.join(output_folder, filename)
                        cv.imwrite(output_path, img)
                        print("Saved prediction:", output_path)

                # 手指在下
                else:
                    if lmList[4][1] > lmList[20][1]:
                        print('手心')
                        cx11 , cy11 = lmList[11][1],lmList[11][2]
                        cx10,cy10 = lmList[10][1],lmList[10][2]
                        inch=((cx10-cx11)**2+(cy10-cy11)**2)**0.5
                        cx0,cy0 = lmList[0][1],lmList[0][2]
                        cx1 ,cy1 =  lmList[1][1],lmList[1][2]
                        img = cv2ImgAddText(img, "鱼际穴", cx1, cy1, (0,0,0), 30)
                        cv.circle(img, (cx1, cy1), 10, (255, 0, 255), cv.FILLED)
                        cx4,cy4 = lmList[4][1],lmList[4][2]
                        img = cv2ImgAddText(img, "十宣穴1", cx4, cy4, (0,0,0), 30)
                        cv.circle(img, (cx4, int(cy4+0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        cx6,cy6 = lmList[6][1],lmList[6][2]
                        img = cv2ImgAddText(img, "四缝穴1", cx6, cy6, (0,0,0), 30)
                        cv.circle(img, (cx6, cy6), 10, (255, 0, 255), cv.FILLED)
                        cx8,cy8 = lmList[8][1],lmList[8][2]
                        img = cv2ImgAddText(img, "十宣穴2", cx8, cy8, (0,0,0), 30)
                        cv.circle(img, (cx8, int(cy8+0.1*inch), 10, (255, 0, 255), cv.FILLED))
                        cx9,cy9 = lmList[9][1],lmList[9][2]
                        cv.circle(img, ((3*cx9+cx0)//4, (3*cy9+cy0)//4), 10, (255, 0, 255), cv.FILLED)
                        img = cv2ImgAddText(img, "劳宫穴", (3*cx9+cx0)//4, (3*cy9+cy0)//4, (0,0,0), 30)
                        x_center,y_center=(cx9+cx0)//2, (cy9+cy0)//2
                        cx10,cy10 = lmList[10][1],lmList[10][2]
                        img = cv2ImgAddText(img, "四缝穴2", cx10, cy10, (0,0,0), 30)
                        cv.circle(img, (cx10, cy10), 10, (255, 0, 255), cv.FILLED)
                        cx11 , cy11 = lmList[11][1],lmList[11][2]
                        cx12 , cy12 = lmList[12][1],lmList[12][2]
                        img = cv2ImgAddText(img, "十宣穴3", cx12, cy12, (0,0,0), 30)
                        cv.circle(img, (cx12, int(cy12+0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        # img = cv2ImgAddText(img, "中冲穴", (2*cx12+cx11)//3, (2*cy12+cy11)//3, (0,0,0), 30)
                        # cv.circle(img, ((2*cx12+cx11)//3, (2*cy12+cy11)//3), 10, (255, 0, 255), cv.FILLED)
                        cx14,cy14 = lmList[14][1],lmList[14][2]
                        img = cv2ImgAddText(img, "四缝穴3", cx14, cy14, (0,0,0), 30)
                        cv.circle(img, (cx14, cy14), 10, (255, 0, 255), cv.FILLED)
                        cx16,cy16 = lmList[16][1],lmList[16][2]
                        img = cv2ImgAddText(img, "十宣穴4", cx16, cy16, (0,0,0), 30)
                        cv.circle(img, (cx16, int(cy16+0.1*inch)), 10, (255, 0, 255), cv.FILLED)
                        cx17,cy17 = lmList[17][1],lmList[17][2]
                        img = cv2ImgAddText(img, "少府穴", (3*cx17+cx0)//4, (3*cy17+cy0)//4, (0,0,0), 30)
                        cv.circle(img, ((3*cx17+cx0)//4, (3*cy17+cy0)//4), 10, (255, 0, 255), cv.FILLED)
                        cx18,cy18 = lmList[18][1],lmList[18][2]
                        img = cv2ImgAddText(img, "四缝穴4", cx18, cy18, (0,0,0), 30)
                        cv.circle(img, (cx18, cy18), 10, (255, 0, 255), cv.FILLED)
                        cx20,cy20 = lmList[20][1],lmList[20][2]
                        img = cv2ImgAddText(img, "十宣穴5", cx20, cy20, (0,0,0), 30)
                        cv.circle(img, (cx20, int(cy20+0.1*inch)), 10, (255, 0, 255), cv.FILLED)

                        # 保存预测图片
                        output_path = os.path.join(output_folder, filename)
                        cv.imwrite(output_path, img)
                        print("Saved prediction:", output_path)

                    else:
                        print('手背')
                        cx0,cy0 = lmList[0][1],lmList[0][2]
                        cx1,cy1 = lmList[1][1],lmList[1][2]
                        x_zhongquan,y_zhongquan = (3*cx0+cx1)//4,(3*cy0+cy1)//4
                        img = cv2ImgAddText(img, "中泉穴", x_zhongquan,y_zhongquan, (0,0,0), 20)
                        cv.circle(img, (x_zhongquan, y_zhongquan), 10, (255, 0, 255), cv.FILLED)
                        cx2 ,cy2 =  lmList[2][1],lmList[2][2]
                        cx3 ,cy3 =  lmList[3][1],lmList[3][2]
                        img = cv2ImgAddText(img, "大骨空穴", cx3,cy3, (0,0,0), 20)
                        cv.circle(img, (cx3, cy3), 10, (255, 0, 255), cv.FILLED)
                        cx4,cy4 = lmList[4][1],lmList[4][2]
                        x_shaoshang,y_shaoshang = (cx3+cx4)//2,(cy3+cy4)//2
                        img = cv2ImgAddText(img, "少商穴", x_shaoshang,y_shaoshang, (0,0,0), 20)
                        cv.circle(img, (x_shaoshang, y_shaoshang), 10, (255, 0, 255), cv.FILLED)
                        img = cv2ImgAddText(img, "十宣穴1", cx4,cy4, (0,0,0), 20)
                        cv.circle(img, (cx4, cy4), 10, (255, 0, 255), cv.FILLED)
                        cx5,cy5 = lmList[5][1],lmList[5][2]
                        x_053,y_053 = (3*cx5+cx0)//4,(3*cy5+cy0)//4
                        x_05,y_05 = (cx5+cx0)//2,(cy5+cy0)//2
                        x_sanjian,y_sanjian = (x_053+cx2)//2,(y_053+cy2)//2
                        x_hegu,y_hegu = (x_05+cx2)//2,(y_05+cy2)//2
                        img = cv2ImgAddText(img, "三间穴", x_sanjian,y_sanjian, (0,0,0), 20)
                        cv.circle(img, (x_sanjian, y_sanjian), 10, (255, 0, 255), cv.FILLED)
                        img = cv2ImgAddText(img, "合谷穴", x_hegu,y_hegu, (0,0,0), 20)
                        cv.circle(img, (x_hegu, y_hegu), 10, (255, 0, 255), cv.FILLED)
                        x_baxie1,y_baxie1 = (cx2+cx5)//2,(cy2+cy5)//2
                        img = cv2ImgAddText(img, "八邪穴1", x_baxie1,y_baxie1, (0,0,0), 20)
                        cv.circle(img, (x_baxie1, y_baxie1), 10, (255, 0, 255), cv.FILLED)
                        x_yaotong1,y_yaotong1=(cx5+cx0)//2,(cy5+cy0)//2
                        img = cv2ImgAddText(img, "腰痛点1", x_yaotong1,y_yaotong1, (0,0,0), 20)
                        cv.circle(img, (x_yaotong1, y_yaotong1), 10, (255, 0, 255), cv.FILLED)
                        cx8,cy8 = lmList[8][1],lmList[8][2]
                        img = cv2ImgAddText(img, "十宣穴2", cx8,cy8, (0,0,0), 20)
                        cv.circle(img, (cx8, cy8), 10, (255, 0, 255), cv.FILLED)
                        cx9 ,cy9 =  lmList[9][1],lmList[9][2]
                        x_baxie2,y_baxie2 = (cx9+cx5)//2,(cy9+cy5)//2
                        img = cv2ImgAddText(img, "八邪穴2", x_baxie2,y_baxie2, (0,0,0), 20)
                        cv.circle(img, (x_baxie2, y_baxie2), 10, (255, 0, 255), cv.FILLED)
                        cx10 ,cy10 =  lmList[10][1],lmList[10][2]
                        img = cv2ImgAddText(img, "中魁穴", cx10,cy10, (0,0,0), 20)
                        cv.circle(img, (cx10, cy10), 10, (255, 0, 255), cv.FILLED)
                        cx12 ,cy12 =  lmList[12][1],lmList[12][2]
                        img = cv2ImgAddText(img, "十宣穴3", cx12,cy12, (0,0,0), 20)
                        cv.circle(img, (cx12, cy12), 10, (255, 0, 255), cv.FILLED)
                        cx13 ,cy13 =  lmList[13][1],lmList[13][2]
                        x_baxie3,y_baxie3 = (cx9+cx13)//2,(cy9+cy13)//2
                        img = cv2ImgAddText(img, "八邪穴3", x_baxie3,y_baxie3, (0,0,0), 20)
                        cv.circle(img, (x_baxie3, y_baxie3), 10, (255, 0, 255), cv.FILLED)
                        x_yaotong2,y_yaotong2 = (cx0+cx13)//2,(cy0+cy13)//2
                        img = cv2ImgAddText(img, "腰痛点2", x_yaotong2,y_yaotong2, (0,0,0), 20)
                        cv.circle(img, (x_yaotong2, y_yaotong2), 10, (255, 0, 255), cv.FILLED)
                        cx16,cy16 =  lmList[16][1],lmList[16][2]
                        img = cv2ImgAddText(img, "十宣穴4", cx16,cy16, (0,0,0), 20)
                        cv.circle(img, (cx16, cy16), 10, (255, 0, 255), cv.FILLED)
                        cx17,cy17 =  lmList[17][1],lmList[17][2]
                        img = cv2ImgAddText(img, "后溪穴", cx17,cy17, (0,0,0), 20)
                        cv.circle(img, (cx17, cy17), 10, (255, 0, 255), cv.FILLED)
                        x_baxie4,y_baxie4 = (cx13+cx17)//2,(cy13+cy17)//2
                        img = cv2ImgAddText(img, "八邪穴4", x_baxie4,y_baxie4, (0,0,0), 20)
                        cv.circle(img, (x_baxie4, y_baxie4), 10, (255, 0, 255), cv.FILLED)
                        cx18,cy18 =  lmList[18][1],lmList[18][2]
                        img = cv2ImgAddText(img, "小骨空穴", cx18,cy18, (0,0,0), 20)
                        cv.circle(img, (cx18, cy18), 10, (255, 0, 255), cv.FILLED)
                        cx20,cy20 =  lmList[20][1],lmList[20][2]
                        img = cv2ImgAddText(img, "十宣穴5", cx20,cy20, (0,0,0), 20)
                        cv.circle(img, (cx20, cy20), 10, (255, 0, 255), cv.FILLED)

                        # 保存预测图片
                        output_path = os.path.join(output_folder, filename)
                        cv.imwrite(output_path, img)
                        print("Saved prediction:", output_path)

            else:
                print("Failed to load image:", img_path)

    print(errorList)
    cv.destroyAllWindows()
