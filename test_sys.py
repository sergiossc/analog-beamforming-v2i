import sys

if __name__ == "__main__":
    pass
    my_arg = None
    try:
        my_arg = sys.argv[1]
    except:
        print ('where is my arg')
        exit()
    print (my_arg)
