"""Standard-library verification of frozen table models and golden vectors."""
import argparse,hashlib,json,math,os,shutil,struct,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiler',default=os.environ.get('CXX','c++'));p.add_argument('--promote',action='store_true');p.add_argument('--report',help='Write a new report without replacing the frozen verification');args=p.parse_args()
    bundle=ROOT/'fpga_handoff/comparison_v1';manifest=json.loads((bundle/'manifest.json').read_text())
    for path,digest in manifest['files'].items():assert hashlib.sha256((bundle/path).read_bytes()).hexdigest()==digest,path
    checks=[];compiler=shutil.which(args.compiler)
    if not compiler:raise ValueError('C++17 compiler required')
    with tempfile.TemporaryDirectory() as temp:
        for case in manifest['candidates']:
            folder=(ROOT/case['model']).parent
            expected=[line.split() for line in (folder/'expected.tsv').read_text().splitlines()]
            if case['kind']=='tao':
                binary=Path(temp)/case['id']
                subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-I',str(folder),str(ROOT/'fpga_handoff/reference.cpp'),'-o',str(binary)],check=True)
                run=subprocess.run([str(binary)],input=(folder/'vectors.tsv').read_text(),text=True,capture_output=True,check=True)
                actual=[line.split() for line in run.stdout.splitlines()];method='standalone C++17'
            else:
                spec=json.loads((ROOT/case['model']).read_text());actual=[]
                for line in (folder/'vectors.tsv').read_text().splitlines():
                    x=[struct.unpack('f',struct.pack('f',float(v)))[0] for v in line.split()];score=spec['init_predict'][0];routes=[]
                    for group in spec['trees']:
                        t=group[0];node=0
                        while t['feature'][node]>=0:
                            node=t['children_left'][node] if x[t['feature'][node]]<t['threshold'][node] else t['children_right'][node]
                        score+=t['value'][node];routes.append(node)
                    actual.append([str(score)]+[str(v) for v in routes])
                method='independent Python exported-tree walk; full C++/native checks in benchmark report'
            assert len(actual)==len(expected);error=0.
            for a,b in zip(actual,expected):
                assert a[1:]==b[1:],'Route mismatch: '+case['id']
                delta=abs(float(a[0])-float(b[0]));error=max(error,delta)
                assert math.isclose(float(a[0]),float(b[0]),rel_tol=1e-10,abs_tol=1e-9),case['id']
            checks.append(dict(id=case['id'],vectors=len(actual),max_margin_error=error,method=method))
    report=dict(all_passed=True,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],checks=checks)
    report_path=Path(args.report) if args.report else bundle/'verification.json'
    if args.report or not report_path.exists():report_path.write_text(json.dumps(report,indent=2)+'\n')
    if args.promote:
        pointer=dict(schema='tree-benchmark.current-comparison.v1',as_of='2026-10-29',manifest='fpga_handoff/comparison_v1/manifest.json',
            manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),verification='fpga_handoff/comparison_v1/verification.json',
            verification_sha256=hashlib.sha256((bundle/'verification.json').read_bytes()).hexdigest(),
            family_winners=dict(axis_linear='axis_linear_d4_n3',oblique_constant='oblique_constant_d4_n5',oblique_linear='oblique_linear_performance_winner'),
            compact_reference='compact_refit_3t_d4',reference='xgboost',hardware_winner=None,
            note='Final ten-model paper comparison. Standardized inputs. No synthesized resource winner; historical CURRENT_MODEL remains the Compact refit default.')
        (ROOT/'fpga_handoff/CURRENT_COMPARISON.json').write_text(json.dumps(pointer,indent=2)+'\n')
        old=ROOT/'fpga_handoff/CURRENT_CANDIDATES.json';legacy=json.loads(old.read_text());legacy['current_comparison']='fpga_handoff/CURRENT_COMPARISON.json';legacy['note']='Historical two-model pair retained for reproducing its emulator study. The final three-family comparison is pinned by CURRENT_COMPARISON.json.';old.write_text(json.dumps(legacy,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
