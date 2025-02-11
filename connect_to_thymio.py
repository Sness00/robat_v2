from thymiodirect import Connection, Thymio
try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    # print(th.variables(robot_id))
    robot = th[robot_id]
    try:
        # robot['motor.left.target'] = 200
        # robot['motor.right.target'] = 200
        while True:
            print(robot['prox.ground.ambiant'], robot['prox.ground.reflected'], robot['prox.ground.delta'])

    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        print('Basta')
    try:
        th.disconnect()
    except Exception as e:
            print(e)
except Exception as e:
    print(e)