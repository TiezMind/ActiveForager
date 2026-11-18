# Driver user-space libraries matching the kernel module; sourced by worker_env.sh and train_env.sh after
# DA_SCRATCH is set. Reads the version from the kernel module, not nvidia-smi, which is what breaks.
# Some images ship user-space driver libraries older than the kernel module (libnvidia-ml.so.1 points
# at the wrong version, so nvidia-smi and NVML fail), and the system links can be rewritten to empty
# placeholders under running jobs (GPU_POOL.md known issues; 4417015 and 4417014 at 12:28 on 2026-10-18,
# after their lanes had sourced this file). So the NVML of the kernel module always goes first, also when
# the links are intact: a process started later from this environment does not depend on them.
_kmod=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null)
if [ -n "$_kmod" ] && [ -s "/lib/x86_64-linux-gnu/libnvidia-ml.so.$_kmod" ]; then
    mkdir -p "$DA_SCRATCH/nvlib"
    ln -sfn "/lib/x86_64-linux-gnu/libnvidia-ml.so.$_kmod" "$DA_SCRATCH/nvlib/libnvidia-ml.so.1"
    export LD_LIBRARY_PATH="$DA_SCRATCH/nvlib:${LD_LIBRARY_PATH:-}"
fi
# Under a driver older than the CUDA 12.9 wheels, FlashInfer's JIT-built sampling kernels fail with
# "device kernel image is invalid". Images with such drivers ship CUDA forward-compat libraries, which
# must come first on LD_LIBRARY_PATH; on a newer driver they would downgrade it, so only then.
# The same applies when the image's libcuda does not match the kernel module.
if [ -e /usr/local/cuda/compat/libcuda.so.1 ] && [ -n "$_kmod" ]; then
    _compat=$(readlink /usr/local/cuda/compat/libcuda.so.1 | sed 's/^libcuda\.so\.//' | cut -d. -f1)
    if [ "${_kmod%%.*}" -lt "$_compat" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
    fi
    unset _compat
fi
unset _kmod
