## Troubleshooting

Check the proxy first:

```bash
curl -sS http://127.0.0.1:8123/status
```

If `backend_base_url` is `null`, the proxy is still waiting for Slurm or vLLM. Check the Slurm job:

```bash
squeue -u "$USER" -n pi-vllm-qwen36-27b
sacct -j <jobid> --format JobID,JobName%30,State,ExitCode,Elapsed,NodeList%30
tail -n 240 logs/pi-vllm-qwen36-27b-<jobid>.err
tail -n 180 logs/pi-vllm-qwen36-27b-<jobid>.out
```

If the log contains:

```text
unsupported GNU version! gcc versions later than 13 are not supported
```

then the worker did not receive `NVCC_APPEND_FLAGS`. Confirm the log has the vLLM Ray env-copy lines shown above. If it does not, the job was launched from an older `slurm/pi-vllm.sbatch` or the environment was changed before vLLM started.

If the log contains:

```text
Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

then the job was launched without the explicit `CUDA_HOME`/`CUDACXX` settings, or Ray did not copy them to workers. The current script sets both before Ray starts and includes them in `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`.

If the log contains:

```text
gcc: fatal error: cannot execute 'cc1plus'
```

then `nvcc` is seeing a mixed compiler environment. The current script pins `CC`, `CXX`, `CUDAHOSTCXX`, `COMPILER_PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` to the complete GCCcore 14.3.0 installation and copies those variables to Ray workers.

If the log contains FlashInfer GDN compile errors in files like:

```text
flashinfer/flat/hopper/collective
gdn_prefill_kernel
```

then vLLM is using the FlashInfer GDN prefill backend. The current Slurm script avoids that with `--additional-config '{"gdn_prefill_backend":"triton"}'`.

If the log contains:

```text
Kernel requires a runtime memory allocation, but no allocator was set
```

then the Triton/FLA GDN path was captured in a CUDA graph. The current Slurm script avoids that with `--enforce-eager`.

If Pi returns:

```text
400 Unexpected message role
```

then vLLM is receiving an OpenAI role that the Qwen chat template does not support. The proxy normalizes `/v1/chat/completions` JSON bodies before forwarding: all `system` and `developer` messages are merged into one leading `system` message, `function` is mapped to `tool`, and supported `user`/`assistant`/`tool` messages are preserved. Restart the local proxy or restart Pi after changing `pi_vllm_proxy.py`.

If the job reaches the Slurm time limit before `/v1/models` is available, keep `#SBATCH --time=04:00:00` or increase it. First launch can spend several minutes compiling and warming CUDA graphs.
