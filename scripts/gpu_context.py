"""
GPU Context Manager for proper OpenCL resource cleanup.
This module provides context managers for OpenCL resources to prevent memory leaks.
"""

import pyopencl as cl
import contextlib
import logging

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def opencl_context():
    """
    Context manager for OpenCL context and command queue.
    Ensures proper cleanup of OpenCL resources.
    """
    ctx = None
    queue = None
    try:
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms available")
        
        # Try GPU first, fallback to CPU
        devices = []
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    break
            except:
                continue
        
        if not devices:
            # Fallback to CPU
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=cl.device_type.CPU)
                    if devices:
                        break
                except:
                    continue
        
        if not devices:
            raise RuntimeError("No OpenCL devices available")
        
        ctx = cl.Context(devices)
        queue = cl.CommandQueue(ctx)
        
        logger.info(f"Created OpenCL context with device: {devices[0].name}")
        yield ctx, queue
        
    except Exception as e:
        logger.error(f"Failed to create OpenCL context: {e}")
        raise
    finally:
        if queue is not None:
            try:
                queue.finish()
                queue = None
            except:
                pass
        if ctx is not None:
            try:
                # Force cleanup by deleting context
                del ctx
                ctx = None
            except:
                pass
        logger.info("OpenCL context cleaned up")

@contextlib.contextmanager
def opencl_buffers(ctx, host_arrays):
    """
    Context manager for OpenCL buffers.
    Automatically releases buffers when exiting context.
    
    Args:
        ctx: OpenCL context
        host_arrays: List of numpy arrays to create buffers for
    
    Yields:
        List of OpenCL buffers
    """
    buffers = []
    try:
        mf = cl.mem_flags
        for array in host_arrays:
            if array is not None:
                buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=array)
                buffers.append(buffer)
            else:
                buffers.append(None)
        yield buffers
    finally:
        # Explicitly release buffers
        for buffer in buffers:
            if buffer is not None:
                try:
                    buffer.release()
                except:
                    pass
        buffers.clear()

def cleanup_opencl_resources():
    """
    Force cleanup of any remaining OpenCL resources.
    Call this at application exit.
    """
    try:
        # This will force garbage collection of OpenCL objects
        import gc
        gc.collect()
        logger.info("OpenCL resources cleanup completed")
    except:
        pass
