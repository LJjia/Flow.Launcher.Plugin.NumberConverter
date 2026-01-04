# -*- coding: utf-8 -*-

import sys,os

parent_folder_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_folder_path)
sys.path.append(os.path.join(parent_folder_path, 'lib'))
sys.path.append(os.path.join(parent_folder_path, 'plugin'))

from flowlauncher import FlowLauncher
import re
import math


def create_float_class(name, s, e, m):
    class FloatIR():
        '''
        1 singned
        4 exponent
        3 significand
        '''
        SIGNED=s
        EXPONENT=e
        MANTISSA=m
        def __init__(self, data: float) -> None:
            # (-1)^sign * 2^(E-bias) * (1+M/(2**3))
            # bias = 2^(4 - 1) - 1 = 7
            self.bias = 2**(self.EXPONENT-1)-1
            # exp range [1-bias, 0] and [1, bias]
            min_val = 2**(1-self.bias)
            max_val = 2**self.bias * (2-2**(-self.MANTISSA))
            # we use nearest round-off
            # max: 2^7 * (1+7*1/8) (0b 1110 111) = 240

            if type(data) == float or type(data) == int:
                data = float(data)
                # singned
                signed = bin(int(data<0))
                data=abs(data)
                if data>max_val:
                    self.E_bin = '1'*self.EXPONENT
                    self.M_bin = '0'*self.MANTISSA
                elif data<min_val:
                    self.E_bin = '0'*self.EXPONENT
                    self.M_bin = math.floor(data/(2**(1-self.bias-self.MANTISSA)))
                else:
                    # here we need to think how to convert a float data to fp8
                    # for simple. we used round to zero
                    rtz_exp = math.floor(math.log2(min(max(data, min_val), max_val)))
                    rtz_m = math.floor((data - 2**rtz_exp) / (2**(rtz_exp -self.MANTISSA)))
                    # rtz_exp = (E-bias)
                    E = rtz_exp+self.bias
                    M = rtz_m
                    self.E_bin = bin(E).split('b')[-1].zfill(self.EXPONENT)
                    self.M_bin = bin(M).split('b')[-1].zfill(self.MANTISSA)
                self.S_bin = signed.split('b')[-1]
            elif type(data) == str or type(data) == int:
                # include bin(0b10), hex(0x20, or 0X20), int(2)
                if type(data) == str:
                    if (data[:2] == "0x" or data[:2] == "0X"):
                        data = int(data, base=16)
                    elif (data[:2] == "0b" or data[:2] == "0B"):
                        data = int(data, base=2)
                # we need trans all format to 0b format
                data = bin(data)
                if len(data) - 2 >= self.EXPONENT+self.MANTISSA:
                    # truncate
                    self.E_bin = data[-self.EXPONENT-self.MANTISSA:-self.MANTISSA]
                    self.M_bin = data[-self.MANTISSA:]
                    # if not specify sign, we default "0"
                    self.S_bin = data[-self.EXPONENT-self.MANTISSA -1] if len(data)-2 > self.EXPONENT+self.MANTISSA else "0"
                else:
                    self.S_bin = '0'
                    min_mantissa = min(self.MANTISSA, len(data) - 2)
                    min_exponent = len(data) - 2 - min_mantissa
                    self.M_bin = data[-min_mantissa:]
                    self.E_bin = data[2:2+min_exponent] if min_exponent>0 else "0"
                # print(f"{name}, s {self.S_bin} e {self.E_bin} m {self.M_bin}")


        def toFloat(self):
            if(self.E_bin == "0"*self.EXPONENT):
                return (-1)**int(self.S_bin, base=2) * 2**(1-self.bias) * (int(self.M_bin, base=2)/(2**self.MANTISSA)) 
            elif(self.E_bin == "1"*self.EXPONENT):
                return ("-" if int(self.S_bin, base=2) else "+") + "inf"
            else:
                return (-1)**int(self.S_bin, base=2) * 2**(int(self.E_bin, base=2) - self.bias) * (1+int(self.M_bin, base=2)/(2**self.MANTISSA))


        def toBin(self):
            return f"0b{self.S_bin}_{self.E_bin}_{self.M_bin}"


        def toHex(self):
            return hex(int(f"0b{self.S_bin}{self.E_bin}{self.M_bin}", base=2))
    FloatIR.__name__ = name
    return FloatIR


HALF = create_float_class("HALF E5M10", 1, 5, 10)
BF16 = create_float_class("BF16 E8M7", 1, 8, 7)
FP32 = create_float_class("FP32 E8M23", 1, 8, 23)
FP8E4M3 = create_float_class("FP8 E4M3", 1, 4, 3)
FP8E5M2 = create_float_class("FP8 E5M2", 1, 5, 2)



def intToDec(dec):
    return (f"{int(dec)}", "Dec")


def intToHex(dec):
    return (f"{hex(dec)}", "Hex")


def intToBin(dec):
    return (f"{bin(dec)}", "Bin")


def is_float_strict_regex(value: str) -> bool:
    pattern = r'^[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?$'
    return bool(re.match(pattern, value))


def getDecOrFloat(arg):
    '''
    get str the true mean
    '''
    if arg.startswith("0x") or arg.startswith("0X"):
        return (int(arg, 16), "Hex")
    elif arg.startswith("0b") or arg.startswith("0B"):
        return (int(arg, 2), "Bin")
    elif is_float_strict_regex(arg):
        # 0. response float
        return (float(arg), "Float")
    else:
        # 0 response int
        return (int(arg, 10), "Dec")


class NumberConverter(FlowLauncher):
    i2i_converters = [intToDec, intToHex, intToBin]
    fp_converters = [HALF, BF16, FP8E4M3, FP8E5M2, FP32]

    def query(self, query):
        results = []
        if len(query) == 0:
            return results
        try:
            decResult = getDecOrFloat(query)
            if decResult[1] == "Float":
                # float format
                for func in self.fp_converters:
                    fp = func(decResult[0])
                    results.append({
                        "Title": f"Hex {fp.toHex()} Bin {fp.toBin()}",
                        "SubTitle": f"{func.__name__} {fp.toFloat()}",
                        "IcoPath": "Images/app.png",
                        'JsonRPCAction': {
                            'method': 'copyToClipboard',
                            'parameters': [0],
                            'dontHideAfterAction': False
                        }
                    })
            elif decResult[1] == "Hex" or decResult[1] == "Bin":
                # hex or int format, we need trans to other int format and float
                for func in self.i2i_converters:
                    result = func(decResult[0])
                    results.append({
                        "Title": result[0],
                        "SubTitle": f"{result[1]}",
                        "IcoPath": "Images/app.png",
                        'JsonRPCAction': {
                            'method': 'copyToClipboard',
                            'parameters': [result[0]],
                            'dontHideAfterAction': False
                        }
                    })
                for func in self.fp_converters:
                    fp = func(query)
                    results.append({
                        "Title": f"{fp.toFloat()}",
                        "SubTitle": f"{func.__name__} Bin {fp.toBin()}",
                        "IcoPath": "Images/app.png",
                        'JsonRPCAction': {
                            'method': 'copyToClipboard',
                            'parameters': [0],
                            'dontHideAfterAction': False
                        }
                    })
            else:
                # Dec only need dec->dec convert
                for func in self.i2i_converters:
                    result = func(decResult[0])
                    results.append({
                        "Title": result[0],
                        "SubTitle": f"{result[1]}",
                        "IcoPath": "Images/app.png",
                        'JsonRPCAction': {
                            'method': 'copyToClipboard',
                            'parameters': [result[0]],
                            'dontHideAfterAction': False
                        }
                    })
        except Exception as e:
            results = []
            results.append({
                "Title": f"Invalid parameters {e} {decResult}",
                "SubTitle": "Please try again",
                "IcoPath": 'Images/app.png'
            })
            # print("error", result)
            return results
        # print("finnal", result)
        return results

    def copyToClipboard(self, value):
        command = 'echo|set /p={v}|clip'.format(v=value.strip())
        os.system(command)


if __name__ == "__main__":
    NumberConverter()
