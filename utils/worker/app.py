from resources.resources import (
    get_rpc_worker, get_wq_worker, get_msg_worker
)
import os
import sys
import signal
import asyncio


def signal_handler(signal, action):

    os.write(2, 'Received SIGINT. Stop!\n'.encode())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


loop = asyncio.get_event_loop()
loop.create_task(get_rpc_worker().run())
loop.create_task(get_wq_worker().run())
loop.create_task(get_msg_worker().run())
loop.run_forever()

