from thymiodirect import Connection, Thymio
import time

try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'\nThymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    robot = th[robot_id]
    try:
        robot['motor.left.target'] = 200
        robot['motor.right.target'] = 200
        while True:
            print(robot['prox.ground.reflected'][0])
            
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
    print(th.variables(robot_id))
    time.sleep(1)
    th.disconnect()
except Exception as e:
    print(e)