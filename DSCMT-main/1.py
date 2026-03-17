import os
appear_pair_env = os.getenv('APPEAR_PAIR', 'RGBD').upper()
print(f"APPEAR_PAIR is set to: {appear_pair_env}")
