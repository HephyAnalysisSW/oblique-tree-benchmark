"""Resource instrumentation and evaluated-checkpoint archiving for one scan case."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json,os,resource,time
from data_utils import resolve
from scripts import train_tao_support_refit as worker
from scripts.support_refit_artifacts import install_archiver


def main():
    config=sys.argv[sys.argv.index('--config')+1];plan=json.loads(resolve(config).read_text());run=plan['run_name']
    install_archiver(worker,plan);original=worker.separation_check
    def measured(*args,**kwargs):
        started=time.perf_counter();report=original(*args,**kwargs)
        record=dict(report,wall_seconds=time.perf_counter()-started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),run_name=run)
        with resolve('results/'+run+'_separation_resources.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print('SEPARATION_RESOURCES '+json.dumps(record),flush=True)
        return report
    worker.separation_check=measured
    return worker.main()


if __name__=='__main__':sys.exit(main())
