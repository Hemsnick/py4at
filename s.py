import operator
group = {'paper':3,
        'scissors':3,
        'stone':3}
while (group['paper'],group['scissors'],group['stone']) != (0,0,0):
    input_s = int(input('chose one : paper =1 , scissors =2 , stone = 3 ?'))
    if input_s == 1 :
        if group['paper'] <= 0:
            print('Out of rule')
            continue
        else:
            group['paper'] -= 1
    elif input_s == 2:
        if group['scissors'] <= 0:
            print('Out of rule')  
            continue
        else:
            group['scissors'] -= 1 
    else:
        if group['stone'] <= 0:
            print('Out of rule')
            continue
        else:
            group['stone'] -= 1
    choose = max(group.items(), key=operator.itemgetter(1))[0]

    print('computer:', group)
    if choose == 'paper':
        print('We choose scissors')
    elif choose =='scissors':
        print('We choose stone')
    else:
        print('We choose paper')