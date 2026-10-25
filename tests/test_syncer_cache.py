import errno,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common.syncer as syncer


class SyncerCacheTests(unittest.TestCase):
    def test_full_shared_cache_falls_back_without_changing_web_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(syncer.user,'ROOT',Path(tmp)),patch.object(syncer.user,'plot_directory','/groups/example/www/tree_benchmark'):
            with patch.object(syncer,'_register_file',side_effect=[OSError(errno.ENOSPC,'full'),None]) as stage:
                syncer.register_file('plot.png')
                self.assertEqual(stage.call_count,2)
                self.assertEqual(syncer.user.plot_directory,str(Path(tmp)/'results/publish_cache/www/tree_benchmark'))
                self.assertEqual(syncer._rel_under_www(syncer.user.plot_directory+'/plots/a.png'),'tree_benchmark/plots/a.png')

    def test_unrelated_io_error_is_not_hidden(self):
        with patch.object(syncer,'_register_file',side_effect=OSError(errno.EIO,'disk error')):
            with self.assertRaises(OSError):syncer.register_file('plot.png')

    def test_pending_old_cache_cannot_overwrite_newer_public_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old/www/tree_benchmark/plot.png';new=root/'new/www/tree_benchmark/plot.png'
            old.parent.mkdir(parents=True);new.parent.mkdir(parents=True)
            old.write_bytes(b'old');new.write_bytes(b'new');os.utime(old,(10,10));os.utime(new,(20,20))
            pending=root/'pending.txt';pending.write_text(str(old)+'\n');commands=[]
            with patch.object(syncer,'PENDING',pending),patch.object(syncer,'file_sync_storage',[str(new)]),patch.object(syncer,'_run',side_effect=commands.append):
                syncer._sync_unlocked()
            copies=[c for c in commands if c.startswith('xrdcp')]
            self.assertEqual(len(copies),1);self.assertIn(str(new),copies[0]);self.assertNotIn(str(old),copies[0])
