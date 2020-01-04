import sys
import getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile>')
      sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
    print ('Input file is "', inputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
