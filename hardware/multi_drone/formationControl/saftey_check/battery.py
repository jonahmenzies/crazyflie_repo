import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.console import Console
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig


def _tolerant_incoming(self, packet):
    try:
        text = packet.data.decode('UTF-8', errors='replace')
    except Exception:
        return
    self.receivedChar.call(text)


Console._incoming = _tolerant_incoming

URIS = [
    'radio://0/10/2M/E7E7E7E701',
    'radio://0/60/2M/E7E7E7E706',
    'radio://0/80/2M/E7E7E7E708',
    'radio://1/90/2M/E7E7E7E709',
    'radio://1/100/2M/E7E7E7E710',
]

# Resting-voltage to state-of-charge breakpoints for a 1S LiPo.
# Approximate, and only valid at rest — under motor load the cell sags
# 0.2-0.3 V, so a resting 3.85 V reads far lower in flight.
_SOC_CURVE = [
    (3.00,   0),
    (3.50,   5),
    (3.70,  15),
    (3.80,  30),
    (3.85,  40),
    (3.90,  50),
    (3.95,  60),
    (4.00,  70),
    (4.05,  80),
    (4.10,  90),
    (4.20, 100),
]


def soc_percent(v):
    if v <= _SOC_CURVE[0][0]:
        return 0
    if v >= _SOC_CURVE[-1][0]:
        return 100
    for i in range(len(_SOC_CURVE) - 1):
        v0, p0 = _SOC_CURVE[i]
        v1, p1 = _SOC_CURVE[i + 1]
        if v0 <= v <= v1:
            return int(round(p0 + (v - v0) * (p1 - p0) / (v1 - v0)))
    return 0


def status_for(v):
    if   v >= 4.10: return 'FULL'
    elif v >= 3.95: return 'GOOD'
    elif v >= 3.85: return 'MARGINAL'
    else:           return 'LOW - CHARGE'


cflib.crtp.init_drivers()

print()
for uri in URIS:
    vbat = {'v': None}

    def cb(ts, msg, conf):
        vbat['v'] = msg['pm.vbat']

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            lg = LogConfig(name='Battery', period_in_ms=100)
            lg.add_variable('pm.vbat', 'float')
            scf.cf.log.add_config(lg)
            lg.data_received_cb.add_callback(cb)
            lg.start()

            start = time.time()
            while vbat['v'] is None and time.time() - start < 3.0:
                time.sleep(0.05)

            # Stop the block and let in-flight packets drain before the link
            # closes, otherwise cflib prints "no LogEntry to handle".
            lg.stop()
            time.sleep(0.3)

        v = vbat['v']
        if v is None:
            print(f'{uri}   no data')
        else:
            print(f'{uri}   {v:.2f} V   {soc_percent(v):3d}%   {status_for(v)}')

    except Exception as ex:
        print(f'{uri}   offline ({type(ex).__name__})')
print()
