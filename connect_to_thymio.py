from thymiodirect import Connection, Thymio
import time
import numpy as np

try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    print(th.variables(robot_id))
    robot = th[robot_id]
    try:
        th.disconnect()
    except Exception as e:
            print(e)
except Exception as e:
    print(e)