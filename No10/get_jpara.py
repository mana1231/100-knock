import sys

def get_jp(jpara_file, jpara_ja, jpara_en):
    with open(jpara_file) as file:
        text = file.readlines()

    data = [x.split('\t') for x in text]
    data = [x for x in data if len(x) >= 4]
    data = [[x[-1].strip(), x[-2].strip()] for x in data]

    with open(jpara_ja, 'w') as f, open(jpara_en, 'w') as g:
        for j, e in data:
            print(j, file=f)
            print(e, file=g)

def main():
    jpara_file = sys.argv[1]
    jpara_ja = sys.argv[2]
    jpara_en = sys.argv[3]
    get_jp(jpara_file, jpara_ja, jpara_en)

if __name__ == '__main__':
    main()
    