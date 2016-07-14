total = 1

for i in range(2000):

    if i % 16 == 0:

        total  = 1+ total*0.99
    else:
        total = total*0.99



print total