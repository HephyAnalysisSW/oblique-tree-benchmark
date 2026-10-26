"""Print the exact affine functions in a portable binary oblique model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import argparse, json
from tao.printing import TreePrinter

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='Portable standardized or raw model.json')
    parser.add_argument('--tree',type=int,help='Zero-based tree index; omit to print the whole forest')
    parser.add_argument('--precision',type=int,default=17)
    parser.add_argument('--output',help='Write text here instead of stdout')
    args=parser.parse_args();printer=TreePrinter(json.loads(Path(args.model).read_text()),args.precision)
    text=printer.format_forest() if args.tree is None else printer.format_tree(args.tree)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    else:sys.stdout.write(text)

if __name__=='__main__':main()
