"""Verify frozen files, compile standalone C++ reference, and check golden vectors.

Python standard library only; no training stack is needed.
"""
import argparse,hashlib,json,math,os,shutil,subprocess,tempfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',help='Optional new verification report path; existing frozen reports are preserved');parser.add_argument('--bundle',help='Default: bundle pinned by CURRENT_MODEL.json');parser.add_argument('--compiler',default=os.environ.get('CXX','c++'))
    args=parser.parse_args()
    repository=Path(__file__).resolve().parents[1]
    selected=json.loads((repository/'fpga_handoff/CURRENT_MODEL.json').read_text())
    bundle=Path(args.bundle).resolve() if args.bundle else (repository/selected['manifest']).parent
    manifest=json.loads((bundle/'manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        assert hashlib.sha256((bundle/relative).read_bytes()).hexdigest()==expected,'Checksum mismatch: '+relative
    compiler=shutil.which(args.compiler)
    if not compiler:raise RuntimeError('A C++17 compiler is required; pass --compiler')
    reference=Path(__file__).resolve().parents[1]/'fpga_handoff/reference.cpp';checks=[]
    with tempfile.TemporaryDirectory() as temporary:
        for case in manifest['candidates']:
            for space in ['standardized','raw']:
                folder=bundle/case['id']/space;binary=Path(temporary)/(case['id']+'_'+space)
                subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-I',str(folder),str(reference),'-o',str(binary)],check=True)
                result=subprocess.run([str(binary)],input=(folder/'vectors.tsv').read_text(),text=True,capture_output=True,check=True)
                actual=[line.split() for line in result.stdout.splitlines()]
                expected=[line.split() for line in (folder/'expected.tsv').read_text().splitlines()]
                assert len(actual)==len(expected)
                error=0.
                for a,b in zip(actual,expected):
                    assert len(a)==len(b) and a[1:]==b[1:],'Leaf routing mismatch'
                    error=max(error,abs(float(a[0])-float(b[0])))
                    assert math.isclose(float(a[0]),float(b[0]),rel_tol=1e-10,abs_tol=1e-9),'Signal margin mismatch'
                checks.append(dict(candidate=case['id'],input_space=space,vectors=len(actual),max_absolute_margin_error=error))
    report=dict(compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],all_passed=True,checks=checks)
    report_path=Path(args.report) if args.report else bundle/'cpp_verification.json'
    if args.report or not report_path.exists():report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
