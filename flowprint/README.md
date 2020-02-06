# pyshark
Pyshark cannot deal with incorrectly captured pcap files. In order to mitigate this, please add a `return` before the `raise TSharkCrashException` in the file `pyshark/capture/capture.py` at the definition of `_cleanup_subprocess`.

```
async def _cleanup_subprocess(self, process):
    """
    Kill the given process and properly closes any pipes connected to it.
    """
    if process.returncode is None:
        try:
            process.kill()
            return await asyncio.wait_for(process.wait(), 1)
        except concurrent.futures.TimeoutError:
            self._log.debug('Waiting for process to close failed, may have zombie process.')
        except ProcessLookupError:
            pass
        except OSError:
            if os.name != 'nt':
                raise
    elif process.returncode > 0:
        return # !!!!!!!!!!!!!!! This line is added !!!!!!!!!!!!!!!
        raise TSharkCrashException('TShark seems to have crashed (retcode: %d). '
                                   'Try rerunning in debug mode [ capture_obj.set_debug() ] or try updating tshark.'
                                   % process.returncode)
```
