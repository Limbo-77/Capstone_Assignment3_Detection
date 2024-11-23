import jetson.inference
import jetson.utils

# 初始化检测网络
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 加载图像
img = jetson.utils.loadImage("workplace.jpg")

# 对图像进行检测
detections = net.Detect(img)

# 遍历检测结果
for detection in detections:
    classid = detection.ClassID
    confidence = detection.Confidence
    left = detection.Left
    top = detection.Top
    right = detection.Right
    bottom = detection.Bottom
    width = detection.Width
    height = detection.Height
    area = detection.Area
    center_x, center_y = detection.Center

    print(f"-- ClassID: {classid}")
    print(f"-- Confidence: {confidence:.6f}")
    print(f"-- Left: {left:.6f}")
    print(f"-- Top: {top:.6f}")
    print(f"-- Right: {right:.6f}")
    print(f"-- Bottom: {bottom:.6f}")
    print(f"-- Width: {width:.6f}")
    print(f"-- Height: {height:.6f}")
    print(f"-- Area: {area}")
    print(f"-- Center: ({center_x:.6f}, {center_y:.6f})\n")

# 如果需要显示结果图像，您可以使用以下代码
jetson.utils.cudaDeviceSynchronize()
jetson.utils.saveImage("output_image.jpg", img)
print("结果图像已保存为 output_image.jpg")
