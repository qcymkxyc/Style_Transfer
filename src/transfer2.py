'''
Created on 2018年8月3日

@author: qcymkxyc
'''

import keras.backend as K
import numpy as np
import keras
import os
from keras.utils import plot_model

from keras.applications import VGG16,VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from scipy import misc

def check_img(func):
    """
        图片检验
    """
    def wrapper(obj,img1,img2):
        assert img1.shape == img2.shape,"图片的大小不统一"
        return func(obj,img1,img2)
    return wrapper


class Transfer(object):
    """
        做风格迁移的类
    """
    def __init__(self,style_image_path,content_image_path,style_weight ,generate_image_shape = None):
        """
            @param style_image_path:    风格图片路径
            @param content_image_path:    内容图片路径
            @param generate_image_shape:    tuple或list，生成图片的长和宽，默认为内容图片的长和宽
            @param style_weight:     dict,{层的名字：权重}风格权重 
        """
        self.style_image_path = style_image_path
        self.content_image_path = content_image_path
        
        self.generate_image_shape = generate_image_shape
        
        self.style_weight = style_weight

        
    def init_model(self):
        
        #获取生成图片的长和宽
        n_H,n_W,_ = Transfer.load_img(
                                    img_path = self.content_image_path,
                                    target_shape = self.generate_image_shape).shape
        self.generate_image_shape = n_H,n_W                                     
        
        #得到输入图片
        content_img = Transfer.load_img(img_path =  self.content_image_path,target_shape = self.generate_image_shape) 
        style_img = Transfer.load_img(img_path = self.style_image_path,target_shape = self.generate_image_shape)
        generated_img = K.placeholder(self.__add_img_noise(content_img),name = "generated_img")
        
        #合成模型输入张量
        content_img = K.expand_dims(content_img, axis = 0)
        style_img = K.expand_dims(style_img,axis = 0)
        generated_img = K.expand_dims(generated_img,axis = 0)
        combine_img = K.concatenate([content_img,style_img,generated_img], axis = 0)
        
        #载入预训练模型
        vgg_model = VGG16(include_top = False,input_tensor = combine_img)
        style_loss = self.__style_loss(self.style_weight, vgg_model)
        content_loss = self.__content_loss(content_img, generated_img)
        total_loss = self.__total_loss(style_loss, content_loss)
        self.f_output = K.function([combine_img], [total_loss])   
    
    @check_img
    def __style_layer_loss(self,style_image,generate_image):
        """
            计算单层的风格loss
            @param style_image:    风格图片
            @param generate_image:    生成的图片
            @return: 风格loss
            
        """
        def compute_style_matrix(image):
            """
                计算图片的风格矩阵
            """
            n_H,n_W,n_C = image.shape
            re_image = K.reshape(x = image,shape = (n_H * n_W,n_C))
            return K.dot(K.transpose(re_image), re_image)
        
        n_H,n_W,n_C = style_image.shape
        
        #计算风格矩阵
        style_matrix = compute_style_matrix(style_image)
        generate_matrix = compute_style_matrix(generate_image)
        
#         return K.sum(K.square(style_matrix - generate_matrix)) / 4. * K.flK.square(n_H * n_W * n_C)
#TODO
        return K.sum(K.square(style_matrix - generate_matrix))
    
    def __style_loss(self,layer_weight,model):
        """
            计算总的风格loss
            @param layer_weight:    dict类型，格式为：{"层的名字":层的权重}
            @param model:    模型
        """
        #获取模型的层数
        layer_dict = {layer.name : layer.output for layer in model.layers}
        #检查层名是否正确
        assert set(layer_weight.keys()).issubset(layer_dict.keys()),"层名不匹配"
        
        self.layers = dict()
        for layer in self.model.layers:
            self.layers[layer.name] = layer.output

        style_loss = 0.
        for layer_name,weight in self.style_weight.items():
            output_img = self.layers[layer_name]
            style_arr = output_img[1,:,:,:]
            generate_arr = output_img[-1,:,:,:]  
            style_loss += weight * self.__style_layer_loss(style_arr, generate_arr)
            
            
        return style_loss                  
        
    def __add_img_noise(self,img,noise_rate = 0.6):
        """
            给图片添加噪声
            @param img: 原图片 
        """
        random_noise = np.random.uniform(low = -20. ,high = 20.,size = img.shape).astype("float32")
        return  (random_noise * noise_rate + img * (1 - noise_rate)) / 255.
    
    @check_img
    def __content_loss(self,content_image,generate_image):
        """
            计算内容loss
            @param content_image:    内容图片
            @param generate_image:    生成图片
            @return:    内容loss
        """
        return K.mean(K.square(content_image - generate_image)) / 4
    
    def __total_loss(self,style_loss,content_loss,alpha = 10,beta = 40):
        """
            计算总体loss
            @param style_loss:    风格loss
            @param content_loss:    内容loss
            @param alpha,beta:    内容loss和风格loss的比例    
            @return: 总体的loss
        """
        return alpha * content_loss + beta * style_loss
    
    @staticmethod
    def load_img(img_path,target_shape):
        """
            载入图片
            @param img_path:    图片路径
            @param target_shape:    生成图片的长和宽
            @return: 图片的array形式
        """
        img = load_img(path = img_path,target_size = target_shape)
        img = img_to_array(img,"channels_last")
        return img
    
    @staticmethod
    def save_img(img,path):
        """
            保存图片
            @param img: 图片
            @param path:     保存路径
        """
        path_name,_ = os.path.split(path)
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        misc.imsave(path,img)
    
if __name__ == "__main__":
    p = "../test.png"
#     x = np.arange(9).reshape(3,3)
#     y = x

style_weight = {
    "block1_conv1" : 0.2,
    "block2_conv1" : 0.2,
    "block3_conv1" : 0.2,
    "block4_conv1" : 0.2,
    "block5_conv1" : 0.2
}

# styhle
a = Transfer(content_image_path = p,style_image_path = p,style_weight = style_weight)
a.init_model()
#     a.style_loss(x, y)
