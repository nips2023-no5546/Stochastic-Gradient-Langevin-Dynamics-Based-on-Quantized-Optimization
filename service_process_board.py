#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# NeurIPS Research Test
# Working Directory :
###########################################################################
import shutil

_description = '''\
====================================================
service_process_board.py : 
                    Written by ******** @ 2023-04-25
====================================================
Example : service_process_board.py
'''
import pickle
import os
import io
import argparse
import textwrap
import numpy as np
import torch
import time

import matplotlib.pyplot as plt
import yaml
import smtplib
import socket
from email.mime.text import MIMEText

import my_debug as DBG
# =================================================================
# Parsing the Argument
# =================================================================
g_first_level_info  = ['Information of Operation', 'Quantization Parameters']
g_second_level_info = ['Learning algorithm', 'Learning rate', 'Learning Schedule', 'eta', 'warmp_up_period']

def _ArgumentParse(_intro_msg, L_param, bUseArgParser=True):
    parser = argparse.ArgumentParser(
        prog='test pytorch_inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-rd', '--result_dir',
                        help="Depict result directory (Default: result)",
                        type=str, default='result')
    parser.add_argument('-t', '--training',
                        help="[0] test [(1)] training",
                        type=int, default=1)
    parser.add_argument('-pr', '--processing',
                        help="[(0)] single file processing [1] multiple files processing",
                        type=int, default=0)
    parser.add_argument('-gp', '--graphic',
                        help="[0] no plot graph  [(1)] active plot graph",
                        type=int, default=1)
    parser.add_argument('-an', '--analysis',
                        help="[0] algorithm analysis  [1] learning rate analysis [2] Scheduler analysis [3] eta analysis",
                        type=int, default=0)
    parser.add_argument('-cf', '--config',
                        help='Config file for Service Process (default) config_service.yaml',
                        type=str, default='config_service.yaml')

    if bUseArgParser or len(L_param)==0:
        args = parser.parse_args()
    else:
        args = parser.parse_args(L_param)

    args.training   = True if args.training == 1 else False
    args.graphic    = True if args.graphic == 1  else False
    return args

def read_config_file(config_file='config_service.yaml'):
    try:
        with open(config_file) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
        exit()
    return config_data

# =================================================================
# Processing Classes
# =================================================================
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            pass
        return super().find_class(module, name)

class process_data_storage:
    def __init__(self):
        # For save
        self._dump_data     = []
        self._load_data     = []
        self.b_gpu_proc     = torch.cuda.is_available()

    # Support mutiple function input (*_data)
    def write_data_on_board(self, *_data):
        _sub_dump_data = []
        for _k, in_data in enumerate(_data):
            _sub_dump_data.append(in_data)

        self._dump_data.append(_sub_dump_data)

    def save_process_data(self, _file_name):
        with open(_file_name, 'wb') as f:
            pickle.dump(self._dump_data, f)

    def load_process_data(self, _file_name):
        with open(_file_name, 'rb') as f:
            while True:
                try:
                    if self.b_gpu_proc:
                        _data = pickle.load(f)
                    else:
                        _data = CPU_Unpickler(f).load()
                except EOFError:
                    break
                if len(_data) == 1:
                    self._load_data.append(_data)
                else:
                    self.data_arrange(_data)

    def data_arrange(self, in_data, b_debug=False):
        _num_data       = len(in_data)
        _num_atoms      = len(in_data[0])
        _arranged_data  = [[0 for _j in range(_num_data)] for _k in range(_num_atoms)]
        DBG.dbg("Re-arrange data", _active=b_debug)
        for _k, _data in enumerate(in_data):
            for _i, _atom in enumerate(_data):
                _arranged_data[_i][_k] = _atom

        DBG.dbg("data casting to float", _active=b_debug)
        for _i, _data_list in enumerate(_arranged_data):
            for _j, _arranged_atom in enumerate(_data_list):
                _arranged_data[_i][_j] = float(_arranged_atom)

        self._load_data = _arranged_data
        print("Finish Data Re-arranging and Casting")

    def print_loaded_data(self, b_print=True):
        if b_print:
            print("--------------------------------------")
            print("Loaded Data Description")
            print("Length of Data : ", len(self._load_data))
            print("--------------------------------------")
            for _k, _data in enumerate(self._load_data):
                print("[%3d] " %_k, _data )
        else:
            pass

class plot_graph:
    def __init__(self, _active=True):
        _config_data       = read_config_file()
        _class_config_data = _config_data['plot_graph']

        self._data      = 0
        self._fig_size  = (_class_config_data['fig_size_x'], _class_config_data['fig_size_y'])
        self.l_legend_info = []
        self.b_active   = _active
        self.graph_file = _class_config_data['graph_file']
        self.graph_dpi  = _class_config_data['graph_dpi']

        self.first_level_info    = g_first_level_info
        self.second_level_info   = g_second_level_info

    def data_analysis(self, _data_info, _legend_presnt):
        # Basic Processing
        [load_data, data_set] = _data_info
        _data_name = data_set['data_name']
        _plot_data = load_data
        _data_spec = len(_data_name)
        _data_dim  = len(load_data[0]) if len(load_data) > 1 else _data_spec

        if _legend_presnt >= 0 and _legend_presnt < 3:
            _data_info  = data_set[self.first_level_info[0]]
            _opt_name   = _data_info[self.second_level_info[0]]
            _lr_data    = _data_info[self.second_level_info[1]]
            _scheduler  = _data_info[self.second_level_info[2]]
        elif _legend_presnt == 3:
            _data_info  = data_set[self.first_level_info[1]]
            _eta_values = _data_info[self.second_level_info[3]]
        else:
            DBG.dbg("Parameter Error !!!")
            exit()

        if _legend_presnt == 0:
            self.l_legend_info.append(_opt_name)
        elif _legend_presnt == 1:
            self.l_legend_info.append(_lr_data)
        elif _legend_presnt == 2:
            self.l_legend_info.append(_scheduler)
        elif _legend_presnt == 3:
            self.l_legend_info.append(str(_eta_values))
        else:
            # NO legend
            pass

        return _data_name, _data_spec, _plot_data

    def operation(self, list_data_info, _legend_presnt=0):
        if self.b_active is False: return

        plt.figure(figsize=self._fig_size)

        for _k, _data_info in enumerate(list_data_info):
            _data_name, _data_spec, _plot_data = self.data_analysis(_data_info,_legend_presnt)

            for _j, _name in enumerate(_data_name):
                plt.subplot(1, _data_spec, 1 + _j)
                plt.plot(_plot_data[_j], label=("%d " % _k) + self.l_legend_info[-1])
                plt.title(_name)
                plt.grid(True)
                plt.legend()

        plt.tight_layout()
        plt.savefig(self.graph_file, dpi=self.graph_dpi)
        plt.show()

class operation_class:
    def __init__(self, L_param, bUseArgParser=True):
        self.file_suffix    = '.pickle'
        self.yaml_suffix    = '.yaml'
        self.text_suffix    = '.txt'
        self.pt_suffix      = '.pt'
        self._args          = _ArgumentParse(_description, L_param=L_param, bUseArgParser=bUseArgParser)
        self.data_dir       = os.path.join(os.getcwd(), self._args.result_dir)
        self.data_obj       = process_data_storage()

        self.pickle_file_list   = [_file for _file in os.listdir(self.data_dir) if _file.endswith(self.file_suffix)]
        self.yaml_file_list     = [_file for _file in os.listdir(self.data_dir) if _file.endswith(self.yaml_suffix)]
        self.text_file_list     = [_file for _file in os.listdir(self.data_dir) if _file.endswith(self.text_suffix)]
        self.pt_file_list       = [_file for _file in os.listdir(self.data_dir) if _file.endswith(self.pt_suffix)]
        self.num_op_files       = len(self.pickle_file_list)
        self._file_index        = 0

        #self._legend_presnt     = self._args.equal_learning_rate*2 + self._args.equal_algorithm
        self._legend_presnt     = self._args.analysis

        # Represent the pickcle and the YAML file lists
        print("-----------------------------------------------")
        print("Pickle Files :")
        for _k, _filename in enumerate(self.pickle_file_list):
            print("%2d  %s " % (_k, _filename))
        print("Yaml Files :")
        for _k, _filename in enumerate(self.yaml_file_list):
            print("%2d  %s " % (_k, _filename))
        print("Text Files :")
        for _k, _filename in enumerate(self.text_file_list):
            print("%2d  %s " % (_k, _filename))
        print("-----------------------------------------------")

        if len(self.pickle_file_list) == 0:
            print("There is not any files from the Learning TEST. Process finish automatically")
            exit()
        else: pass

        # Get Config file and Initialization
        d_config_data           = read_config_file()
        _class_config_data      = d_config_data['operation_class']
        self.result_yaml_file   = _class_config_data['result_yaml_file']

        print("Configuration is finished. Operation Begin")
        print("-----------------------------------------------")

    def single_file_processing(self, _input_file_id):
        self._file_index    = _input_file_id
        _file_name          = self.pickle_file_list[self._file_index]
        print("Target Data File : %s" % (_file_name))
        #Pickcle File Processing
        self.data_obj.load_process_data(os.path.join(self.data_dir, _file_name))
        self.data_obj.print_loaded_data(b_print=False)

        #YAML File Processing
        _file_name      = self.yaml_file_list[self._file_index]
        _yaml_file_name = os.path.join(self.data_dir, _file_name)
        with open(_yaml_file_name) as f:
            _data_set = yaml.load(f, Loader=yaml.FullLoader)
        print("Target YAML File : %s" % (_file_name))
        return  self.data_obj._load_data, _data_set

    def merge_yaml_files(self, list_data_info):
        total_yaml_str  = ''
        _str_divider    = '-------------------------------------------\n'
        for _k, _data_info in enumerate(list_data_info):
            total_yaml_str = total_yaml_str + _str_divider
            yaml_str  = yaml.dump(_data_info[1])
            total_yaml_str = total_yaml_str + yaml_str

        with open(self.result_yaml_file, 'w') as f:
            f.write(total_yaml_str)

        print("-----------------------------------------------")
        print(" Result YANL File : %s" %self.result_yaml_file)
        print("-----------------------------------------------")


    def sort_list_data(self, _list_data):
        first_level_info    = g_first_level_info
        second_level_info   = g_second_level_info
        _key_list           = []

        for _k, _data in enumerate(_list_data):
            _dict_info = _data[1]
            if      self._legend_presnt >= 0 and self._legend_presnt < 3:
                _first_level    = _dict_info[first_level_info[0]]
            elif    self._legend_presnt >= 3:
                _first_level    = _dict_info[first_level_info[1]]
            else:
                _first_level    = 0
                DBG.dbg("Abnormal analysis parameter !!!")
                exit()
                pass

            _key_index      = second_level_info[self._legend_presnt]
            _key_data       = _first_level[_key_index]
            _key_list.append((_key_data, _data))

        _dict_info      = dict(_key_list)
        _sorted__data   = dict(sorted(_dict_info.items()))
        _ret_data       = list(_sorted__data.values())
        return _ret_data


class send_notify:
    def __init__(self, recvEmail="jnwseok@etri.re.kr"):
        _config_data           = read_config_file()
        _class_config_data     = _config_data['send_notify']

        # E-mail Notify
        self.sendEmail  = _class_config_data['sendEmail']
        self.recvEmail  = _class_config_data['recvEmail']
        self._id        = _class_config_data['identification']
        self.password   = _class_config_data['password']

        self.smtpName   = _class_config_data['smtpName']  # smtp 서버 주소
        self.smtpPort   = _class_config_data['smtpPort']  # smtp 포트 번호  587
        self.etri_mail  = _class_config_data['etri_mail']
        self.text       = ''
        self.msg        = ''
        self._real_msg  = []
        self._head_msg  = _class_config_data['head_msg']

    def set_msg(self, _msg):
        self.text = _msg
        self.msg = MIMEText(self.text)         # MIMEText(text , _charset = "utf8")

        self.msg['Subject'] = self._head_msg
        self.msg['From']    = self.sendEmail
        self.msg['To']      = self.recvEmail
        print("=============================================================")
        print("message :")
        print(self.msg.as_string())
        print("=============================================================")

    def send_noti_mail(self, _msg):
        self.set_msg(_msg=_msg)
        s = smtplib.SMTP_SSL(self.smtpName, self.smtpPort)  # 메일 서버 연결

        try:
            s.ehlo()
        except Exception as e:
            DBG.dbg("E-mail Noti system get Error!!! at s.ehlo()")
            print(e)
            s.close()  # smtp 서버 연결을 종료합니다.
            return
            #exit()

        if self.etri_mail: pass
        else:
            try:
                if s.has_extn('STARTTLS'):
                    s.starttls()  # TLS 보안 처리
            except Exception as e:
                DBG.dbg("E-mail Noti system get Error!!! at s.starttls()")
                print(e)
                s.close()  # smtp 서버 연결을 종료합니다.
                return

        try:
            s.login(self._id, self.password)  # 로그인
        except Exception as e:
            print("log in Fail")
            print(e)
            exit()

        s.sendmail(self.sendEmail, self.recvEmail, self.msg.as_string())  # 메일 전송, 문자열로 변환하여 보냅니다.
        s.close()  # smtp 서버 연결을 종료합니다.
        #s.quit()

    def extract_batfile(self, _batfile="work_02.bat"):
        self._real_msg.clear()
        _ret_msg = '-------------------------------------------------------------\n'
        if os.path.isfile(_batfile):
            with open(_batfile, 'r') as f:
                _file_lines = f.readlines()
                for _k, _line_msg in enumerate(_file_lines):
                    _line   = _line_msg.strip()
                    _idx    = _line.find("::")
                    if  _idx < 0:
                        self._real_msg.append(_line)
                    else:
                        pass
                        ## print("there exist ::")
        else:
            _ret_msg += ("There is not the file : %s" %_batfile)

        for s in self._real_msg:
            if s == '': pass
            else: _ret_msg += s +'\n'

        now = time
        _ret_msg += "from : \n"
        _ret_msg += "Host Name  " + socket.gethostname() + "   "
        _ret_msg += "IP address " + socket.gethostbyname(socket.gethostname())
        _ret_msg += "Notification Time : %s" %now.strftime('%Y-%m-%d %H:%M:%S')
        return _ret_msg

class config_yaml:
    def __init__(self):
        self.config_files   = dict(learning     ="config_learning.yaml",
                                   quantization ="config_quantization.yaml",
                                   service      ="config_service.yaml")
        self.list_yaml_data = []

    def get_config_file_name(self, _category):
        return self.config_files[_category]

    def read_yaml_file(self, _yaml_file_name):
        try:
            with open(_yaml_file_name) as f:
                _data_sets = yaml.load(f, Loader=yaml.FullLoader)
            DBG.dbg("Target Read YAML File : %s" % (_yaml_file_name))
        except Exception as e:
            DBG.dbg(e)
            exit()
        return _data_sets

    def write_yaml_file(self, _data, _yaml_file_name):
        try :
            with open(_yaml_file_name, 'w') as f:
                yaml.dump(_data, f)
            DBG.dbg("Target Write YAML File : %s" % (_yaml_file_name))
        except Exception as e:
            DBG.dbg(e)
            exit()

#=============================================================
# Test Processing
#=============================================================
if __name__ == "__main__":
    list_data_info  = []
    operation_class_param = []
    op_obj          = operation_class(L_param=operation_class_param)
    op_plot         = plot_graph(_active=op_obj._args.graphic)
    c_noti          = send_notify()

    if op_obj._args.processing == 0:
        # Single Processing
        _if_idx = int(input("File ID : ")) if len(op_obj.pickle_file_list) > 1 else 0
        list_data_info.append(op_obj.single_file_processing(_if_idx))
        _analysis_data = list_data_info
    elif op_obj._args.processing == 1:
        # Multiple Processing
        for _k in range(op_obj.num_op_files):
            list_data_info.append(op_obj.single_file_processing(_k))
        sorted_data_info = op_obj.sort_list_data(_list_data=list_data_info)
        op_obj.merge_yaml_files(sorted_data_info)
        _analysis_data = sorted_data_info
    else:
        # Test Processing
        DBG.dbg(" NO Processing Thank You~")
        exit()

    #-------------------------------------------------------------
    # Plotting the Result
    #-------------------------------------------------------------
    op_plot.operation(_analysis_data, _legend_presnt=op_obj._legend_presnt)
    #-------------------------------------------------------------
    # Notification Test
    #-------------------------------------------------------------
    #_noti_msg = "Test Process is finished. Please Check"
    #c_noti.send_noti_mail(_msg=_noti_msg)
    #=============================================================
    # Final Stage
    #=============================================================
    print("Process Finished!!")



