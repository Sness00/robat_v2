from thymiodirect import Connection, Thymio
import time

try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'\nThymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    print(th.variables(robot_id))
    time.sleep(1)
    th.disconnect()
except Exception as e:
    print(e)