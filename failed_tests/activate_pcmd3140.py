import time
from smbus2 import SMBus

def activate_mics():
    with SMBus(1) as bus:
        if bus.read_byte_data(int('4E', 16), int('75', 16)) != int('60', 16):
            print('\nActivating converters')
            
            bus.write_byte_data(int('4E', 16), int('2', 16), int('81', 16))
            time.sleep(1e-3)
            bus.write_byte_data(int('4E', 16), int('7', 16), int('60', 16))
            bus.write_byte_data(int('4E', 16), int('B', 16), int('0', 16))
            bus.write_byte_data(int('4E', 16), int('C', 16), int('20', 16))
            bus.write_byte_data(int('4E', 16), int('22', 16), int('41', 16))
            bus.write_byte_data(int('4E', 16), int('2B', 16), int('40', 16))
            bus.write_byte_data(int('4E', 16), int('73', 16), int('C0', 16))
            bus.write_byte_data(int('4E', 16), int('74', 16), int('C0', 16))
            bus.write_byte_data(int('4E', 16), int('75', 16), int('60', 16))
        else:
            print('\nConverters were already activated')