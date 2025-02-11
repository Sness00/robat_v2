from smbus2 import SMBus

def activate_mics():
    with SMBus(1) as bus:
        if bus.read_byte_data(int('4c', 16), int('75', 16)) != int('60', 16):
            print('Activating converters')
            
            bus.write_byte_data(int('4c', 16), int('2', 16), int('85', 16))
            bus.write_byte_data(int('4d', 16), int('2', 16), int('85', 16))
            bus.write_byte_data(int('4e', 16), int('2', 16), int('85', 16))
            bus.write_byte_data(int('4f', 16), int('2', 16), int('85', 16))

            addr = '4c'
            bus.write_byte_data(int(addr, 16), int('7', 16), int('60', 16))
            bus.write_byte_data(int(addr, 16), int('b', 16), int('0', 16))
            bus.write_byte_data(int(addr, 16), int('c', 16), int('20', 16))
            bus.write_byte_data(int(addr, 16), int('22', 16), int('41', 16))
            bus.write_byte_data(int(addr, 16), int('2b', 16), int('40', 16))
            bus.write_byte_data(int(addr, 16), int('3c', 16), int('40', 16))
            bus.write_byte_data(int(addr, 16), int('41', 16), int('40', 16))
            bus.write_byte_data(int(addr, 16), int('73', 16), int('c0', 16))
            bus.write_byte_data(int(addr, 16), int('74', 16), int('c0', 16))
            bus.write_byte_data(int(addr, 16), int('75', 16), int('60', 16))
        else:
            print('Converters were already activated')
