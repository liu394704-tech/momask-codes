#!/usr/bin/env python3
# encoding: utf-8
import os
import sys
import math
import time
import yaml
import ui_add_color
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
from ui_lab_tool import Ui_Form
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
apt_opencv_path = '/usr/lib/python3/dist-packages/'  
if apt_opencv_path not in sys.path:
    sys.path.insert(0, apt_opencv_path)
import cv2

class MainWindow(QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)       
        #################################界面(interface)#######################################
        self.resetServos_ = False
        self.path = '/home/pi/TonyPi/'
        self.lab_file = 'lab_config.yaml'
        self.servo_file = 'servo_config.yaml'
        self.color = 'red'
        self.L_Min = 0
        self.A_Min = 0
        self.B_Min = 0
        self.L_Max = 255
        self.A_Max = 255
        self.B_Max = 255
        self.servo1 = 1500
        self.servo2 = 1500
        self.kernel_erode = 3
        self.kernel_dilate = 3
        self.camera_ui = False
        self.camera_ui_break = False
        
        self.board = rrc.Board()
        self.ctl = Controller(self.board)

        self.horizontal_slider_l_min.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('lmin'))
        self.horizontal_slider_a_min.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('amin'))
        self.horizontal_slider_b_min.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('bmin'))
        self.horizontal_slider_l_max.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('lmax'))
        self.horizontal_slider_a_max.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('amax'))
        self.horizontal_slider_b_max.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('bmax'))
        
        self.horizontal_slider_servo1.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('servo1'))
        self.horizontal_slider_servo2.valueChanged.connect(lambda: self.horizontal_slider_lab_value_change('servo2'))
        
        self.push_button_connect.pressed.connect(lambda: self.on_push_button_action_clicked('connect'))
        self.push_button_disconnect.pressed.connect(lambda: self.on_push_button_action_clicked('disconnect'))
        self.push_button_save_lab.pressed.connect(lambda: self.on_push_button_action_clicked('save_lab'))
        self.push_button_save_servo.pressed.connect(lambda: self.on_push_button_action_clicked('save_servo'))
        self.push_button_add_color.clicked.connect(self.add_color)
        self.push_button_delete_color.clicked.connect(self.delete_color)
               
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_image)
        self.create_config()
        
        self.current_servo_data = self.get_yaml_data(self.path + self.servo_file)
        
        self.servo1 = int(self.current_servo_data['servo1'])
        self.servo2 = int(self.current_servo_data['servo2'])

        self.horizontal_slider_servo1.setValue(self.servo1)
        self.horizontal_slider_servo2.setValue(self.servo2)
        self.label_servo1_value.setNum(self.servo1)
        self.label_servo2_value.setNum(self.servo2)
        
        self.ctl.set_pwm_servo_pulse(1, self.servo1, 500)
        self.ctl.set_pwm_servo_pulse(2, self.servo2, 500)
        
    # 弹窗提示函数(pop-up window prompt function)
    def message_from(self, str):
        try:
            QMessageBox.about(self, '', str)
        except:
            pass

    # 窗口退出(window exit)
    def close_event(self, e):        
        result = QMessageBox.question(self,
                                    "Prompt box",
                                    "quit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            self.camera_ui = True
            self.camera_ui_break = True
            QWidget.closeEvent(self, e)
        else:
            e.ignore()           

    def message_delect(self, string):
        messageBox = QMessageBox()
        messageBox.setWindowTitle('')
        messageBox.setText(string)
        messageBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        messageBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)

        return messageBox.exec_()

    def add_color(self):
        self.qdialog = QDialog()
        self.add_color_dialog = ui_add_color.Ui_dialog()
        self.add_color_dialog.setupUi(self.qdialog)
        self.qdialog.show()
        self.add_color_dialog.push_button_ok.clicked.connect(self.get_color)
        self.add_color_dialog.push_button_cancel.pressed.connect(self.close_qdialog)
    
    def delete_color(self):
        result = self.message_delect('Delect?')
        if not result:
            self.color = self.combo_box_color_select.currentText()
            del self.current_lab_data[self.color]
            self.save_yaml_data(self.current_lab_data, self.path + self.lab_file)
            self.combo_box_color_select.clear()
            self.combo_box_color_select.addItems(self.current_lab_data.keys())
                
    def get_color(self):
        color = self.add_color_dialog.line_edit.text()
        self.combo_box_color_select.addItem(color)
        time.sleep(0.1)
        self.qdialog.accept()
    
    def close_qdialog(self):
        self.qdialog.accept()

    def show_image(self):
        if self.camera_opened:
            ret, origin_frame = self.cap.read()
            if ret:
                origin_frame = cv2.resize(origin_frame, (400, 300))
                frame_gb = cv2.GaussianBlur(origin_frame, (3, 3), 3)
                frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)
                mask = cv2.inRange(frame_lab,
                                   (self.current_lab_data[self.color]['min'][0],
                                    self.current_lab_data[self.color]['min'][1],
                                    self.current_lab_data[self.color]['min'][2]),
                                   (self.current_lab_data[self.color]['max'][0],
                                    self.current_lab_data[self.color]['max'][1],
                                    self.current_lab_data[self.color]['max'][2]))#对原图像和掩模进行位运算(perform bitwise operation to original image and mask)
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_erode, self.kernel_erode)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_dilate, self.kernel_dilate)))
                show_gray_image = QImage(dilated.data, dilated.shape[1], dilated.shape[0], QImage.Format_Indexed8)
                gray_pixmap = QPixmap.fromImage(show_gray_image)
                
                frame_rgb = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
                show_rgb_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
                rgb_pixmap = QPixmap.fromImage(show_rgb_image)
                
                self.label_process.setPixmap(gray_pixmap)
                self.label_orign.setPixmap(rgb_pixmap)

    def horizontal_slider_lab_value_change(self, name):
        if name == 'lmin': 
            self.current_lab_data[self.color]['min'][0] = self.horizontal_slider_l_min.value()
            self.label_l_min.setNum(self.current_lab_data[self.color]['min'][0])
        if name == 'amin':
            self.current_lab_data[self.color]['min'][1] = self.horizontal_slider_a_min.value()
            self.label_a_min.setNum(self.current_lab_data[self.color]['min'][1])
        if name == 'bmin':
            self.current_lab_data[self.color]['min'][2] = self.horizontal_slider_b_min.value()
            self.label_b_min.setNum(self.current_lab_data[self.color]['min'][2])
        if name == 'lmax':
            self.current_lab_data[self.color]['max'][0] = self.horizontal_slider_l_max.value()
            self.label_l_max.setNum(self.current_lab_data[self.color]['max'][0])
        if name == 'amax':
            self.current_lab_data[self.color]['max'][1] = self.horizontal_slider_a_max.value()
            self.label_a_max.setNum(self.current_lab_data[self.color]['max'][1])
        if name == 'bmax':
            self.current_lab_data[self.color]['max'][2] = self.horizontal_slider_b_max.value()
            self.label_b_max.setNum(self.current_lab_data[self.color]['max'][2])
        if name == 'servo1':
            self.current_servo_data['servo1'] = self.horizontal_slider_servo1.value()
            self.label_servo1_value.setNum(self.current_servo_data['servo1'])
            self.servo1 = int(self.current_servo_data['servo1'])
            self.ctl.set_pwm_servo_pulse(1, int(self.servo1), 20)
        if name == 'servo2':
            self.current_servo_data['servo2'] = self.horizontal_slider_servo2.value()
            self.label_servo2_value.setNum(self.current_servo_data['servo2'])
            self.servo2 = int(self.current_servo_data['servo2'])
            self.ctl.set_pwm_servo_pulse(2, int(self.servo2), 20)
    
    def get_yaml_data(self, yaml_file):
        file = open(yaml_file, 'r', encoding='utf-8')
        file_data = file.read()
        file.close()
        
        data = yaml.load(file_data, Loader=yaml.FullLoader)
        
        return data

    def save_yaml_data(self, data, yaml_file):
        file = open(yaml_file, 'w', encoding='utf-8')
        yaml.dump(data, file)
        file.close()
    
    def create_config(self):
        if not os.path.isfile(self.path + self.lab_file):          
            data = {'red': {'max': [255, 255, 255], 'min': [0, 150, 130]},
                    'green': {'max': [255, 110, 255], 'min': [47, 0, 135]},
                    'blue': {'max': [255, 136, 120], 'min': [0, 0, 0]},
                    'black': {'max': [89, 255, 255], 'min': [0, 0, 0]},
                    'white': {'max': [255, 255, 255], 'min': [193, 0, 0]}}
            self.save_yaml_data(data, self.path + self.lab_file)
            self.current_lab_data = data
            
            self.color_list = ['red', 'green', 'blue', 'black', 'white']
            self.combo_box_color_select.addItems(self.color_list)
            self.combo_box_color_select.currentIndexChanged.connect(self.selection_change)       
            self.selection_change()
        else:
            try:
                self.current_lab_data = self.get_yaml_data(self.path + self.lab_file)
                self.color_list = self.current_lab_data.keys()
                self.combo_box_color_select.addItems(self.color_list)
                self.combo_box_color_select.currentIndexChanged.connect(self.selection_change)       
                self.selection_change() 
            except:
                self.message_from('read error！')
        
        if not os.path.isfile(self.path + self.servo_file):          
            data = {'servo1': 1500,
                    'servo2': 1500}
            self.save_yaml_data(data, self.path + self.servo_file)
                          
    def get_color_value(self, color):  
        if color != '':
            self.current_lab_data = self.get_yaml_data(self.path + self.lab_file)
            if color in self.current_lab_data:
                self.horizontal_slider_l_min.setValue(self.current_lab_data[color]['min'][0])
                self.horizontal_slider_a_min.setValue(self.current_lab_data[color]['min'][1])
                self.horizontal_slider_b_min.setValue(self.current_lab_data[color]['min'][2])
                self.horizontal_slider_l_max.setValue(self.current_lab_data[color]['max'][0])
                self.horizontal_slider_a_max.setValue(self.current_lab_data[color]['max'][1])
                self.horizontal_slider_b_max.setValue(self.current_lab_data[color]['max'][2])
            else:
                self.current_lab_data[color] = {'max': [255, 255, 255], 'min': [0, 0, 0]}
                self.save_yaml_data(self.current_lab_data, self.path + self.lab_file)
                
                self.horizontal_slider_l_min.setValue(0)
                self.horizontal_slider_a_min.setValue(0)
                self.horizontal_slider_b_min.setValue(0)
                self.horizontal_slider_l_max.setValue(255)
                self.horizontal_slider_a_max.setValue(255)
                self.horizontal_slider_b_max.setValue(255)

    def selection_change(self):
        self.color = self.combo_box_color_select.currentText()      
        self.get_color_value(self.color)
        
    def on_push_button_action_clicked(self, buttonName):
        if buttonName == 'save_lab':
            try:              
                self.save_yaml_data(self.current_lab_data, self.path + self.lab_file)
            except Exception as e:
                self.message_from('save failed！')
                return
            self.message_from('save success！')
        elif buttonName == 'save_servo':
            try:               
                self.save_yaml_data(self.current_servo_data, self.path + self.servo_file)
            except Exception as e:
                self.message_from('save failed！')
                return
            self.message_from('save success！')                      
        elif buttonName == 'connect':
            self.cap = cv2.VideoCapture(-1)
            if not self.cap.isOpened():
                self.label_process.setText('Can\'t find camera')
                self.label_orign.setText('Can\'t find camera')
                self.label_process.setAlignment(Qt.AlignCenter|Qt.AlignVCenter)
                self.label_orign.setAlignment(Qt.AlignCenter|Qt.AlignVCenter)
            else:
                self.camera_opened = True
                self.timer.start(20)
        elif buttonName == 'disconnect':
            self.camera_opened = False
            self.timer.stop()
            self.label_process.setText(' ')
            self.label_orign.setText(' ')           
            self.cap.release()

if __name__ == "__main__":  
    app = QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
