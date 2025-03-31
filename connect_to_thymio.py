from thymiodirect import Connection, Thymio
import time
import numpy as np

try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'\nThymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    print(th.variables(robot_id))
    robot = th[robot_id]
    robot['motor.left.target'] = 100
    robot['motor.right.target'] = 100
    try:
        while True:
            print('#'*80)
            print(robot['prox.ground.reflected'])
            print(robot['prox.ground.ambiant'])
            print(robot['prox.ground.delta'])
            print('#'*80)
            time.sleep(0.1)
    except KeyboardInterrupt:
         
        try:
            robot['motor.left.target'] = 0
            robot['motor.right.target'] = 0
            time.sleep(1)
            th.disconnect()
        except Exception as e:
                print(e)
except Exception as e:
    print(e)