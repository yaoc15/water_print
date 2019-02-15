from aip import AipOcr
import cv2
import math
import operator
import numpy as np
IMAGE_PATH = './tmp/0001.jpg'
np.set_printoptions(threshold=np.inf)
def is51W(list):
  return not ((list - [255,0,0,255,255,255,0,0,0,0]).any())
def is6W(list):
  return not ((list - [255,0,0,255,255,0,0,0,0,0]).any())
def is71W(list):
  return not ((list - [255,0,0,255,0,0,0,0,0,0]).any())
def is72W(list):
  return not ((list - [255,0,0,0,255,0,0,0,0,0]).any())
def is4W(list):
  return not ((list - [255,0,255,255,255,255,0,0,0,0]).any())
def is32W(list):
  return not ((list - [255,255,255,255,255,255,0,0,0,255]).any())
def is11B(list):
  return not ((list - [0,0,255,255,255,255,255,255,255,0]).any())
def is12B(list):
  return not ((list - [0,255,255,255,255,255,255,255,0,255]).any())
def is2B(list):
  return not ((list - [0,0,255,255,255,255,255,255,0,0]).any())
def is31B(list):
  return not ((list - [0,0,255,255,255,255,255,0,0,0]).any())
def is4B(list):
  return not ((list - [0,0,255,255,255,255,0,0,0,0]).any())
def is52B(list):
  return not ((list - [0,0,255,255,255,0,0,0,0,0]).any())
def find_border_point(image,image_color,t):
  a = image.copy()
  temp = image_color.copy()
  tmp = t
  for i in range(1,image.shape[0]-1):
    for j in range(1,image.shape[1]-1):
      y = np.zeros(10,dtype=int)
      y[0] = image[i][j]
      y[1] = image[i-1][j-1]
      y[2] = image[i][j-1]
      y[3] = image[i+1][j-1]
      y[4] = image[i+1][j]
      y[5] = image[i+1][j+1]
      y[6] = image[i][j+1]
      y[7] = image[i-1][j+1]
      y[8] = image[i-1][j]
      y[9] = y[1]
      # f = 0
      # for k in range(1,9):
      #   f += abs(y[k+1] - y[k])
      # f = f // 255
      # if f == 2:
      if (tmp > 0):
        if (is51W(y) or is6W(y) or is71W(y) or is72W(y) or is4W(y) or is32W(y)):
          print('w',i,j)
          # print(temp[i][j])
          # temp[i][j] = [45,45,45]
          temp[i][j] = [0,0,0]
          # print(image_binarization(temp))
          a[i][j] = 7
          tmp -= 1
      if (tmp < 0):
        if (is11B(y) or is12B(y) or is2B(y) or is31B(y) or is4B(y) or is52B(y)):
          print('B',i,j)
          # temp[i][j] = [210,210,210]
          temp[i][j] = [255,255,255]
          tmp += 1
          a[i][j] = 255
  return temp  
def get_file_content(filePath): #读入图片文件
  with open(filePath,'rb') as fp:
    return fp.read()

def baidu_image_segmentation(): #调用百度ai的api划分文本图片上的文字
  APP_ID = '15546110'
  API_KEY = 'lNuZFA2Co3hHpwG6GM3iMSYA'
  SECRET_KEY = 'cGqgNQkEVNGugtDUjDM5tctP4lXIzDaN'
  client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
  image = get_file_content(IMAGE_PATH)
  options = {}
  options["recognize_granularity"] = "small"
  options["vertexes_location"] = "true"
  return client.general(image,options)

def get_position_list(): #处理baidu返回的结果生成包含每个字的方块数组
  segmentation_result = baidu_image_segmentation()
  print(segmentation_result)
  position_list = []
  for i in range(segmentation_result['words_result_num']):
    for j in range(len(segmentation_result['words_result'][i]['chars'])):
      position_list.append({
        'x':segmentation_result['words_result'][i]['chars'][j]['location']['top'],
        'y':segmentation_result['words_result'][i]['chars'][j]['location']['left'],
        'w':segmentation_result['words_result'][i]['chars'][j]['location']['height'],
        'h':segmentation_result['words_result'][i]['chars'][j]['location']['width']
      })
  return position_list

def image_segmentation(): #读入图片进行文字划分
  image = cv2.imread(IMAGE_PATH)
  position_list = get_position_list()
  for i in range(len(position_list)):
    x = position_list[i]['y'] - 2
    y = position_list[i]['x']
    w = position_list[i]['h'] + 8
    h = position_list[i]['w']
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(image, pt1, pt2, (0, 0, 255))
  cv2.imshow('splited char image', image)
  cv2.waitKey(0)

def save_image_block(): #读入图片,存储文字划分的结果
  image = cv2.imread(IMAGE_PATH)
  position_list = get_position_list()
  for i in range(len(position_list)):
    x = position_list[i]['y']
    y = position_list[i]['x']
    w = position_list[i]['h']
    h = position_list[i]['w']
    tmp_image = image[y+1:y+h,x+1:x+w]
    cv2.imwrite("baidu_result/" + str(i) + ".png",tmp_image)

def image_binarization(image):#图片二值化
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    return binary

def count_black_pixel(image):#统计黑色像素点个数
  num = 0
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      if (image[i][j] == 0):
        num += 1
  return num

def count_gray_pixel(image):#统计一个图像的灰度值和
  num = 0
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      num += (255 - image[i][j])
  return num

def get_average_pixel():#二值图像统计平均像素点
  num = 132#切分的字数个数
  pixel = np.zeros(num,dtype=int)
  new_pixel = []
  average_pixel = 0
  new_average_pixel = 0
  for i in range(0,num):
    image_path = './baidu_result/' + str(i) + '.png'
    color_image = cv2.imread(image_path)
    image_binary = image_binarization(color_image)
    pixel[i] = count_black_pixel(image_binary)
    average_pixel += pixel[i]
  average_pixel = round(average_pixel / (num+1),2)
  for i in range(0,num):
    if (pixel[i] >= average_pixel * 0.3):
      new_pixel.append(i)
      new_average_pixel += pixel[i]
  new_average_pixel = round(new_average_pixel / len(new_pixel),2)
  return (new_pixel,new_average_pixel)

def get_average_gray_pixel():#灰度图像统计平均像素点
  num = 115#切分的字数个数
  pixel = np.zeros(num,dtype=int)
  new_pixel = []
  average_pixel = 0
  new_average_pixel = 0
  for i in range(0,num):
    image_path = './baidu_print_information/' + str(i) + '.png'
    color_image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    pixel[i] = count_gray_pixel(image_gray)
    average_pixel += pixel[i]
  average_pixel = round(average_pixel / (num+1),2)
  for i in range(0,num):
    if (pixel[i] >= average_pixel * 0.3):
      new_pixel.append(i)
      new_average_pixel += pixel[i]
  new_average_pixel = round(new_average_pixel / len(new_pixel),2)
  return (new_pixel,new_average_pixel)

def get_print_image_block():#将每个字二值化后嵌入水印信息
  number = 1
  pixel_list = []
  average_pixel = 0
  pixel_list, average_pixel = get_average_pixel()
  for i in range(len(pixel_list)):
    image_path = './baidu_result/' + str(pixel_list[i]) + '.png'
    color_image = cv2.imread(image_path)
    image_binary = image_binarization(color_image)
    image_binary_pixel = count_black_pixel(image_binary)
    tmp = round(image_binary_pixel / average_pixel /0.15,1)
    x1 = math.floor(tmp)
    x2 = math.floor(tmp + 1)
    x = x2
    if ((number == 1 and x1 % 2 == 1) or(number == 0 and x1 % 2 == 0)):
      x = x1
    change_pixel = int(round(x * 0.15 * average_pixel - image_binary_pixel))
    print('change_pixed',change_pixel)
    new_image = find_border_point(image_binary,color_image,change_pixel)
    cv2.imwrite("baidu_print_information/" + str(i) + ".png",new_image)
    # cv2.imshow('origin',color_image)
    # cv2.imshow('new',new_image)
    # cv2.waitKey(0)

def deposit_water_print_image_block():#提取每个字的水印信息
  number = 1
  pixel_list = []
  average_pixel = 0
  pixel_list, average_pixel = get_average_gray_pixel()
  print('gray_pixel_list',average_pixel)
  for i in range(len(pixel_list)):
    image_path = './baidu_print_information/' + str(pixel_list[i]) + '.png'
    color_image = cv2.imread(image_path)
    image_binary = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    image_binary_pixel = count_gray_pixel(image_binary)
    print(i,image_binary_pixel)
    tmp = round(image_binary_pixel / average_pixel /0.15,0)
    # print('information',tmp)


# get_print_image_block()
deposit_water_print_image_block()