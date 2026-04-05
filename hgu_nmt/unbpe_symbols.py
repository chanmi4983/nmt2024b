import argparse
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=str, default='')
    parser.add_argument("--output", type=str, default='')
    args = parser.parse_args()

    in_file = open(args.input, "r", encoding="utf-8")
    out_file = open(args.output, "w")

    line_num = 0
    while True:
        line = in_file.readline().strip() 
        if line == '':
            break

        line = line.replace("__P@@ ", "__P")
        out_file.write(line.strip() + '\n')

    in_file.close()
    out_file.close()

    print('Unbpe symbols. Done!!')
