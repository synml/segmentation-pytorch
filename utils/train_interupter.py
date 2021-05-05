def train_interupter():
    with open('train_interupter.ini', 'r', encoding='utf-8') as f:
        flag = f.read().strip()

    if flag == '0':
        return False
    elif flag == '1':
        with open('train_interupter.ini', 'w', encoding='utf-8') as f:
            f.write('0')
        return True
    else:
        raise ValueError('Wrong flag value.')
