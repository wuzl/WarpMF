def free_memory():
    total = 0
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            line = line.strip()
            if any(line.startswith(field) for field in ('MemFree', 'Buffers', 'Cached')):
                field, amount, unit = line.split()
                amount = int(amount)
                if unit != 'kB':
                    raise ValueError(
                        'Unknown unit {u!r} in /proc/meminfo'.format(u = unit))
                total += amount
    return total

def chunks(data,length):
    for i in range(0, len(data), length):
        yield data[i:i + length]
