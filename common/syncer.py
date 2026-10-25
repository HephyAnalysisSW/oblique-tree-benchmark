"""Figure publishing adapted from HEPHY-uncertainty/common/syncer.py.

Import common.syncer to intercept ROOT canvas output and Matplotlib savefig.
The benchmark uses the CERN CMS proxy at ~/.private/.proxy via X509_USER_PROXY;
Kerberos is not required. Figures are staged under the configured /www/ root
and uploaded with xrdfs/xrdcp on normal exit. Retry pending files with
python -m common.syncer.
"""

import os, uuid
import subprocess 

def has_kerberos_token():
    try:
        # Run the `klist` command
        result = subprocess.run(['klist'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if the output contains "No credentials cache found"
        if "No credentials cache found" in result.stderr:
            return False
        
        # If there's no error, and the command executed successfully, a token is present
        return result.returncode == 0
    except FileNotFoundError:
        # `klist` command is not available on the system
        print("Error: `klist` command not found. Ensure Kerberos is installed.")
        return False

# Keep the original sync queue and EOS mapping, with optional ROOT support and
# Figure.savefig interception for the benchmark's Matplotlib figures.
from pathlib import Path
from functools import wraps
import shutil
import common.user as user
from common.plot_layout import publication_path
os.environ.setdefault("CERN_USER", "schoef")
_proxy = Path.home() / ".private" / ".proxy"
if _proxy.is_file():
    os.environ.setdefault("X509_USER_PROXY", str(_proxy))
file_sync_storage = []
PENDING = user.ROOT / "results" / "file_sync_storage.txt"

def _register_file(filename):
    if not isinstance(filename, (str, bytes, os.PathLike)):
        return  # savefig also accepts BytesIO and other file objects
    source = Path(os.fsdecode(filename)).resolve()
    if not source.is_file():
        return
    if "/www/" in str(source):
        destination = source
    else:
        relative = publication_path(source, user.ROOT)
        destination = Path(user.plot_directory) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    indexes = []
    directory = destination.parent
    plot_root = Path(user.plot_directory).resolve()
    while directory == plot_root or plot_root in directory.parents:
        index = directory / 'index.php'
        if directory == plot_root / 'plots':
            shutil.copy2(Path(__file__).parent / 'scripts/php/plot_groups.php', index)
        elif not index.exists():
            shutil.copy2(Path(__file__).parent / 'scripts/php/index.php', index)
        indexes.append(index)
        if directory == plot_root:
            break
        directory = directory.parent
    for path in (destination, *indexes):
        if str(path) not in file_sync_storage:
            file_sync_storage.append(str(path))

def register_file(filename):
    try:
        return _register_file(filename)
    except OSError as error:
        import errno
        if error.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        fallback = user.ROOT / 'results/publish_cache/www/tree_benchmark'
        if Path(user.plot_directory) == fallback:
            raise
        user.plot_directory = str(fallback)
        print(f'Plot staging filesystem full; using {fallback}', flush=True)
        return _register_file(filename)

try:
    import ROOT
except ImportError:
    ROOT = None
if ROOT is not None:
    def _wrap_root(original):
        @wraps(original)
        def wrapped(self, *args, **kwargs):
            if args and isinstance(args[0], str):
                Path(args[0]).parent.mkdir(parents=True, exist_ok=True)
            result = original(self, *args, **kwargs)
            if args:
                register_file(args[0])
            return result
        return wrapped
    ROOT.TCanvas.Print = _wrap_root(ROOT.TCanvas.Print)
    ROOT.TCanvas.SaveAs = _wrap_root(ROOT.TCanvas.SaveAs)

from matplotlib.figure import Figure
_savefig = Figure.savefig
@wraps(_savefig)
def my_savefig(self, fname, *args, **kwargs):
    if isinstance(fname, (str, bytes, os.PathLike)):
        Path(os.fsdecode(fname)).parent.mkdir(parents=True, exist_ok=True)
    result = _savefig(self, fname, *args, **kwargs)
    register_file(fname)
    return result
Figure.savefig = my_savefig

## Wrap pickle dump
#import pickle
## that's the old dump method
#pickle._dump = pickle.dump
#def syncer_pickle_dump( *args ):
#    # second argument is file handle!
#    if len(args)>1:
#        file_sync_storage.append( args[1].name )
#    else:
#        logger.warning( "Pickle dump called with less than two arguments... shouldn't happen." )
#    pickle._dump(*args)
## that's the new dump method
#pickle.dump = syncer_pickle_dump

# What happens on exit 
def write_sync_files_txt(output_filename = 'file_sync_storage.txt'):
    # No logger here, since it is already unloaded!
    # write file_sync_storage.txt because we can't scp in the container with kerberos authentication
    
    # find php files and copy these also
    for filename in list(set(file_sync_storage)):
        dir_path = os.path.dirname(os.path.realpath(filename))
        phpfiles = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.php')]
        for f in phpfiles:
            if f not in file_sync_storage:
                file_sync_storage.append(f)
    n_files = 0
    if len(file_sync_storage)>0:
        with open( output_filename, 'w' ) as outfile:
            for filename in file_sync_storage:
                if 'www/' in filename:
                    # for rsync cmd with relative path
                    outfile.write('{filename}\n'.format(filename=os.path.expanduser(os.path.expandvars(filename)).replace('www/','www/./')))
                    n_files+=1
                else:
                    print(("Will not sync %s" % filename))
        print(("Analysis.Tools.syncer: Written %i files to %s for rsync." % (n_files, output_filename)))
    return n_files

gif_cmds = []
def makeRemoteGif(directory, pattern, name, delay=50):
    if "CERN_USER" not in os.environ: 
        print("To sync with CERN www directory, you need to set $CERN_USER")
        return
    if '/www/' not in directory:
        print("makeRemoteGif: /www/ not found. Do nothing.")
        return
    directory_ = '/'+(directory.split('/www/')[-1])
    cern_user = os.environ["CERN_USER"]
    remotedir = "/eos/user/{INITIAL}/{CERN_USER}/www/{directory}".format( CERN_USER=cern_user, INITIAL=cern_user[0],directory=directory_)
    cmd = "ssh {CERN_USER}@lxplus.cern.ch 'convert -delay {delay} -loop 0 {remotedir}/{pattern} {remotedir}/{name}.gif'".format(CERN_USER=cern_user, delay=delay, remotedir=remotedir, pattern=pattern, name=name)
    if cmd not in gif_cmds:
        gif_cmds.append( cmd )

def make_gifs( cmds=gif_cmds ):
    ret = []
    for cmd in cmds:
        print("make gif:", cmd )
        output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        if error:
            ret.append(cmd)
    return ret

#def sync(gifs=False):
#
#    if "CERN_USER" not in os.environ: 
#        print("To sync with CERN www directory, you need to set $CERN_USER")
#        return
#
#    #if not has_kerberos_token():
#    #    print("No kerberos token. Do nothing.")
#    #    return 
#    
#    global file_sync_storage
#    global gif_cmds 
#
#    if len(file_sync_storage)==0:
#        print("No files for syncing.")
#        return
#
#    filename = '/tmp/%s.txt'%uuid.uuid4()
#
#    if write_sync_files_txt(filename)==0: return 
#
#    cmd = "rsync -avR  `cat %s` ${CERN_USER}@lxplus.cern.ch:/eos/user/$(echo ${CERN_USER} | head -c 1)/${CERN_USER}/www/" % filename
#    #cmd = "xrdcp -f  `cat %s` root://eosuser.cern.ch//eos/user/$(echo ${CERN_USER} | head -c 1)/${CERN_USER}/www/" % filename
#    print(cmd)
#    output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
#    #os.remove(filename)
#    file_sync_storage = []
#    if gifs:
#        gif_cmds = make_gifs(gif_cmds) 
#
#    return #output, error
import os
import uuid
import shlex
import pathlib
import subprocess

# ----------------- CONFIG -----------------
EOS_HOST = "root://eosuser.cern.ch"
WWW_MARKER = "/www/"
# ------------------------------------------

def _run(cmd):
    """Run a shell command and print stderr on failure."""
    p = subprocess.run(cmd, shell=True, executable="/bin/bash",
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\n--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}"
        )
    return p.stdout

def _rel_under_www(src_abs):
    """
    Strip everything up to and including '/www/'.
    '/groups/.../www/t_sch/a.txt' -> 't_sch/a.txt'
    Fallback: if no '/www/' present, drop the leading slash to avoid mirroring root.
    """
    s = os.path.abspath(src_abs)
    if WWW_MARKER in s:
        return s.split(WWW_MARKER, 1)[1]
    return str(pathlib.Path(s).as_posix()).lstrip("/")

def _eos_user_base(user):
    return f"/eos/user/{user[0]}/{user}/www"

def _eos_url(abs_eos_path):
    # xrdcp requires the double slash before absolute EOS path
    return f"{EOS_HOST}/{abs_eos_path}"

def _sync_unlocked(gifs=False):
    """
    Sequential upload to EOS using xrdfs/xrdcp, driven by your file list.
    Path mapping: everything AFTER '/www/' in the source path is mirrored under .../www on EOS.
    """
    if "CERN_USER" not in os.environ:
        print("To sync with CERN www directory, you need to set $CERN_USER")
        return

    global file_sync_storage
    global gif_cmds

    if PENDING.exists():
        for line in PENDING.read_text().splitlines():
            normalized = os.path.abspath(line)
            if os.path.isfile(normalized) and normalized not in file_sync_storage:
                file_sync_storage.append(normalized)
    if len(file_sync_storage) == 0:
        return

    # Use your existing writer for the file list
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    filename = str(PENDING)
    if write_sync_files_txt(filename) == 0:
        return

    user = os.environ["CERN_USER"]
    eos_base = _eos_user_base(user)

    # Read file list
    with open(filename, "r") as f:
        files = list(dict.fromkeys(line.strip() for line in f if line.strip()))

    # Build copy plan and collect directories
    newest = {}          # One source per public URL, including fallback caches.
    dirs_needed = set()
    for src in files:
        src_abs = os.path.abspath(src)
        rel = _rel_under_www(src_abs)
        dest_abs = f"{eos_base}/{rel}"
        if not os.path.isfile(src_abs) or os.path.getsize(src_abs) == 0:
            continue
        prior = newest.get(dest_abs)
        if prior is None or os.path.getmtime(src_abs) > os.path.getmtime(prior):
            newest[dest_abs] = src_abs
        dirs_needed.add(os.path.dirname(dest_abs))
    plan = [(source, destination) for destination, source in newest.items()]

    # Create directories on EOS (sequential)
    for d in sorted(dirs_needed):
        cmd = f'xrdfs {EOS_HOST} mkdir -p {shlex.quote(d)}'
        try:
            _run(cmd)
        except RuntimeError as e:
            # If directory exists, xrdfs -p is usually fine; otherwise show the error and continue
            print(e)

    # Copy files (sequential)
    for src_abs, dest_abs in plan:
        dest_spec = _eos_url(dest_abs)
        cmd = f'xrdcp -f {shlex.quote(src_abs)} {shlex.quote(dest_spec)}'
        print(cmd)
        _run(cmd)

    # Cleanup / post
    os.remove(filename)
    file_sync_storage = []
    if gifs:
        gif_cmds = make_gifs(gif_cmds)

    return


def sync(gifs=False):
    # Multiple Slurm jobs can finish together. Serialize their pending-file queue.
    import fcntl
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    with open(str(PENDING) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _sync_unlocked(gifs=gifs)

import atexit
def _sync_at_exit():
    try:
        sync()
    except Exception as error:
        print(f"Plot upload failed; pending files retained in {PENDING}: {error}")
atexit.register(_sync_at_exit)
atexit.register( make_gifs )

if __name__ == "__main__":
    sync()  # also retries the persistent pending-file list
