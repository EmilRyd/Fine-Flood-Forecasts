import pycuda
from pycuda import compiler
import pycuda.driver as drv

drv.init()

print('%d devices found', drv.Device.count())

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print(ordinal, dev.name())
