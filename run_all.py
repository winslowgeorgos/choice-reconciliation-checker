import subprocess

scripts = [
    'localcurrencyadjustment.py',
    'foreignCurrencyadjustment.py',
    'verify_transfers.py'
]

for script in scripts:
    print(f'Running {script} ...')
    subprocess.run(['python', script], check=True)
print('All done!')
