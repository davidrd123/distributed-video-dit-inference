---
title: "Distributed communication package - torch.distributed"
source_url: https://docs.pytorch.org/docs/stable/distributed.html
fetch_date: "2026-02-22"
source_type: docs
conversion_notes: >
  Converted from raw HTML via pandoc 2.9.2.1, then stripped site chrome
  (nav, sidebar, footer, rating widget, breadcrumbs, cookie banner).
  Pandoc span/class attributes removed via regex cleanup.
  Content extracted from line 3198 to 7569 of pandoc output.
  Some pandoc formatting artifacts may remain in complex API signatures.
---


Distributed communication package - torch.distributed
===================================================================================================================================================

Created On: Jul 12, 2017 \| Last Updated On: Jan 08, 2026


Note

Please refer to [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) for a brief introduction to all features related to distributed training.

[]{#module-torch.distributed}

Backends
-----------------------------------------------------------

`torch.distributed` supports four built-in backends, each with different capabilities. The table below shows which functions are available for use with a CPU or GPU for each backend. For NCCL, GPU refers to CUDA GPU while for XCCL to XPU GPU.

MPI supports CUDA only if the implementation used to build PyTorch supports it.


Backend


`gloo`

`mpi`

`nccl`

`xccl`

Device

CPU

GPU

CPU

GPU

CPU

GPU

CPU

GPU

send

✓

✘

✓

?

✘

✓

✘

✓

recv

✓

✘

✓

?

✘

✓

✘

✓

broadcast

✓

✓

✓

?

✘

✓

✘

✓

all\_reduce

✓

✓

✓

?

✘

✓

✘

✓

reduce

✓

✓

✓

?

✘

✓

✘

✓

all\_gather

✓

✓

✓

?

✘

✓

✘

✓

gather

✓

✓

✓

?

✘

✓

✘

✓

scatter

✓

✓

✓

?

✘

✓

✘

✓

reduce\_scatter

✓

✓

✘

✘

✘

✓

✘

✓

all\_to\_all

✘

✘

✓

?

✘

✓

✘

✓

barrier

✓

✘

✓

?

✘

✓

✘

✓


### Backends that come with PyTorch

PyTorch distributed package supports Linux (stable), macOS (stable), and Windows (prototype). By default for Linux, the Gloo and NCCL backends are built and included in PyTorch distributed (NCCL only when building with CUDA). MPI is an optional backend that can only be included if you build PyTorch from source. (e.g. building PyTorch on a host that has MPI installed.)


Note

As of PyTorch v1.8, Windows supports all collective communications backends but NCCL, If the `init_method` argument of [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") points to a file it must adhere to the following schema:

-   Local file system, `init_method="file:///d:/tmp/some_file"`

-   Shared file system, `init_method="file://////{machine_name}/{share_folder_name}/some_file"`

Same as on Linux platform, you can enable TcpStore by setting environment variables, MASTER\_ADDR and MASTER\_PORT.


### Which backend to use?

In the past, we were often asked: "which backend should I use?".

-   Rule of thumb

    -   Use the NCCL backend for distributed training with CUDA **GPU**.

    -   Use the XCCL backend for distributed training with XPU **GPU**.

    -   Use the Gloo backend for distributed training with **CPU**.

-   GPU hosts with InfiniBand interconnect

    -   Use NCCL, since it's the only backend that currently supports InfiniBand and GPUDirect.

-   GPU hosts with Ethernet interconnect

    -   Use NCCL, since it currently provides the best distributed GPU training performance, especially for multiprocess single-node or multi-node distributed training. If you encounter any problem with NCCL, use Gloo as the fallback option. (Note that Gloo currently runs slower than NCCL for GPUs.)

-   CPU hosts with InfiniBand interconnect

    -   If your InfiniBand has enabled IP over IB, use Gloo, otherwise, use MPI instead. We are planning on adding InfiniBand support for Gloo in the upcoming releases.

-   CPU hosts with Ethernet interconnect

    -   Use Gloo, unless you have specific reasons to use MPI.

### Common environment variables


#### Choosing the network interface to use

By default, both the NCCL and Gloo backends will try to find the right network interface to use. If the automatically detected interface is not correct, you can override it using the following environment variables (applicable to the respective backend):

-   **NCCL\_SOCKET\_IFNAME**, for example `export NCCL_SOCKET_IFNAME=eth0`

-   **GLOO\_SOCKET\_IFNAME**, for example `export GLOO_SOCKET_IFNAME=eth0`

If you're using the Gloo backend, you can specify multiple interfaces by separating them by a comma, like this: `export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3`. The backend will dispatch operations in a round-robin fashion across these interfaces. It is imperative that all processes specify the same number of interfaces in this variable.

#### Other NCCL environment variables

**Debugging** - in case of NCCL failure, you can set `NCCL_DEBUG=INFO` to print an explicit warning message as well as basic NCCL initialization information.

You may also use `NCCL_DEBUG_SUBSYS` to get more details about a specific aspect of NCCL. For example, `NCCL_DEBUG_SUBSYS=COLL` would print logs of collective calls, which may be helpful when debugging hangs, especially those caused by collective type or message size mismatch. In case of topology detection failure, it would be helpful to set `NCCL_DEBUG_SUBSYS=GRAPH` to inspect the detailed detection result and save as reference if further help from NCCL team is needed.

**Performance tuning** - NCCL performs automatic tuning based on its topology detection to save users' tuning effort. On some socket-based systems, users may still try tuning `NCCL_SOCKET_NTHREADS` and `NCCL_NSOCKS_PERTHREAD` to increase socket network bandwidth. These two environment variables have been pre-tuned by NCCL for some cloud providers, such as AWS or GCP.

For a full list of NCCL environment variables, please refer to [NVIDIA NCCL's official documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

You can tune NCCL communicators even further using `torch.distributed.ProcessGroupNCCL.NCCLConfig` and `torch.distributed.ProcessGroupNCCL.Options`. Learn more about them using `help` (e.g. `help(torch.distributed.ProcessGroupNCCL.NCCLConfig)`) in the interpreter.


### Copy Engine Collectives


Note

Copy Engine Collectives require NCCL 2.28 or later, and GPUs with peer-to-peer (P2P) access.

Copy Engine (CE) Collectives are an optimization for NCCL collective operations that offload data movement to the GPU's copy engines (DMA engines) instead of using CUDA streaming multiprocessors (SMs). This frees up SMs for compute work, enabling better overlap of communication and computation during distributed training.

To use CE collectives, you need to:

1.  Configure the NCCL process group with the zero-CTA policy

2.  Set up symmetric memory with the NCCL backend

3.  Allocate tensors using symmetric memory

4.  Register the tensors with symmetric memory via rendezvous

Once set up, standard collective functions like [`all_gather_into_tensor()`](#torch.distributed.all_gather_into_tensor "torch.distributed.all_gather_into_tensor") and [`all_to_all_single()`](#torch.distributed.all_to_all_single "torch.distributed.all_to_all_single") will automatically use the copy engines when operating on symmetric memory tensors.

**Example**


    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    # Initialize process group with zero-CTA policy for CE collectives
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    device = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)

    # Set up symmetric memory with NCCL backend
    symm_mem.set_backend("NCCL")
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    # Allocate tensors using symmetric memory
    numel = 1024 * 1024
    inp = symm_mem.empty(numel, device=device)
    out = symm_mem.empty(numel * world_size, device=device)

    # Register tensors for symmetric memory operations
    symm_mem.rendezvous(inp, group=group_name)
    symm_mem.rendezvous(out, group=group_name)

    # Perform collective operation using copy engines
    # This now runs on DMA engines instead of SMs
    work = dist.all_gather_into_tensor(out, inp, async_op=True)
    work.wait()


**Benefits**

-   **SM offloading**: Communication runs on copy engines, leaving SMs free for computation

-   **Better overlap**: Enables more efficient computation/communication overlap

-   **Transparent API**: Uses the same collective API, just with symmetric memory tensors

**Requirements and Limitations**

-   NCCL version 2.28 or later

-   GPUs must have peer-to-peer (P2P) access enabled

-   Tensors must be allocated using `torch.distributed._symmetric_memory()` and rendezvoused

-   The NCCL process group must be configured with `NCCL_CTA_POLICY_ZERO` or the environment variable `NCCL_CTA_POLICY` be set to 2

-   As of NCCL 2.28, CE collectives cannot run with the default stream, so you would need to use the `async_op=True` flag to activate the internal stream of `ProcessGroupNCCL` or create a side stream yourself

[]{#distributed-basics}

Basics
-------------------------------------------------------

The `torch.distributed` package provides PyTorch support and communication primitives for multiprocess parallelism across several computation nodes running on one or more machines. The class [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") builds on this functionality to provide synchronous distributed training as a wrapper around any PyTorch model. This differs from the kinds of parallelism provided by Multiprocessing package - torch.multiprocessing(multiprocessing.html) and [`torch.nn.DataParallel()`](generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") in that it supports multiple network-connected machines and in that the user must explicitly launch a separate copy of the main training script for each process.

In the single-machine synchronous case, `torch.distributed` or the [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") wrapper may still have advantages over other approaches to data-parallelism, including [`torch.nn.DataParallel()`](generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel"):

-   Each process maintains its own optimizer and performs a complete optimization step with each iteration. While this may appear redundant, since the gradients have already been gathered together and averaged across processes and are thus the same for every process, this means that no parameter broadcast step is needed, reducing time spent transferring tensors between nodes.

-   Each process contains an independent Python interpreter, eliminating the extra interpreter overhead and "GIL-thrashing" that comes from driving several execution threads, model replicas, or GPUs from a single Python process. This is especially important for models that make heavy use of the Python runtime, including models with recurrent layers or many small components.

Initialization
-----------------------------------------------------------------------

The package needs to be initialized using the [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") or [`torch.distributed.device_mesh.init_device_mesh()`](#torch.distributed.device_mesh.init_device_mesh "torch.distributed.device_mesh.init_device_mesh") function before calling any other methods. Both block until all processes have joined.


Warning

Initialization is not thread-safe. Process group creation should be performed from a single thread, to prevent inconsistent 'UUID' assignment across ranks, and to prevent races during initialization that can lead to hangs.

torch.distributed.is\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/__init__.py#L15)

:   Return `True` if the distributed package is available.

    Otherwise, `torch.distributed` does not expose any other APIs. Currently, `torch.distributed` is available on Linux, MacOS and Windows. Set `USE_DISTRIBUTED=1` to enable it when building PyTorch from source. Currently, the default value is `USE_DISTRIBUTED=1` for Linux and Windows, `USE_DISTRIBUTED=0` for MacOS.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.init\_process\_group[(]*backend=None*, *init\_method=None*, *timeout=None*, *world\_size=-1*, *rank=-1*, *store=None*, *group\_name=\'\'*, *pg\_options=None*, *device\_id=None*, *\_ranks=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1578)

:   Initialize the default distributed process group.

    This will also initialize the distributed package.

    There are 2 main ways to initialize a process group:

    :   1.  Specify `store`, `rank`, and `world_size` explicitly.

        2.  Specify `init_method` (a URL string) which indicates where/how to discover peers. Optionally specify `rank` and `world_size`, or encode all required parameters in the URL and omit them.

    If neither is specified, `init_method` is assumed to be "env://".

    Parameters[:]

    :   -   **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *or* [*Backend*](#torch.distributed.Backend "torch.distributed.Backend")*,* *optional*) -- The backend to use. Depending on build-time configurations, valid values include `mpi`, `gloo`, `nccl`, `ucc`, `xccl` or one that is registered by a third-party plugin. Since 2.6, if `backend` is not provided, c10d will use a backend registered for the device type indicated by the device\_id kwarg (if provided). The known default registrations today are: `nccl` for `cuda`, `gloo` for `cpu`, `xccl` for `xpu`. If neither `backend` nor `device_id` is provided, c10d will detect the accelerator on the run-time machine and use a backend registered for that detected accelerator (or `cpu`). This field can be given as a lowercase string (e.g., `"gloo"`), which can also be accessed via [`Backend`](#torch.distributed.Backend "torch.distributed.Backend") attributes (e.g., `Backend.GLOO`). If using multiple processes per machine with `nccl` backend, each process must have exclusive access to every GPU it uses, as sharing GPUs between processes can result in deadlock or NCCL invalid usage. `ucc` backend is experimental. Default backend for the device can be queried with [`get_default_backend_for_device()`](#torch.distributed.get_default_backend_for_device "torch.distributed.get_default_backend_for_device").

        -   **init\_method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *optional*) -- URL specifying how to initialize the process group. Default is "env://" if no `init_method` or `store` is specified. Mutually exclusive with `store`.

        -   **world\_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Number of processes participating in the job. Required if `store` is specified.

        -   **rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Rank of the current process (it should be a number between 0 and `world_size`-1). Required if `store` is specified.

        -   **store** ([*Store*](#torch.distributed.Store "torch.distributed.Store")*,* *optional*) -- Key/value store accessible to all workers, used to exchange connection/address information. Mutually exclusive with `init_method`.

        -   **timeout** (*timedelta,* *optional*) -- Timeout for operations executed against the process group. Default value is 10 minutes for NCCL and 30 minutes for other backends. This is the duration after which collectives will be aborted asynchronously and the process will crash. This is done since CUDA execution is async and it is no longer safe to continue executing user code since failed async NCCL operations might result in subsequent CUDA operations running on corrupted data. When TORCH\_NCCL\_BLOCKING\_WAIT is set, the process will block and wait for this timeout.

        -   **group\_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *optional,* *deprecated*) -- Group name. This argument is ignored

        -   **pg\_options** (*ProcessGroupOptions,* *optional*) -- process group options specifying what additional options need to be passed in during the construction of specific process groups. As of now, the only options we support is `ProcessGroupNCCL.Options` for the `nccl` backend, `is_high_priority_stream` can be specified so that the nccl backend can pick up high priority cuda streams when there're compute kernels waiting. For other available options to config nccl, See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t>

        -   **device\_id** ([*torch.device*](tensor_attributes.html#torch.device "torch.device") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- a single, specific device this process will work on, allowing for backend-specific optimizations. Currently this has two effects, only under NCCL: the communicator is immediately formed (calling `ncclCommInit*` immediately rather than the normal lazy call) and sub-groups will use `ncclCommSplit` when possible to avoid unnecessary overhead of group creation. If you want to know NCCL initialization error early, you can also use this field. If an int is provided, the API assumes that the accelerator type at compile time will be used.

        -   **\_ranks** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) -- The ranks in the process group. If provided, the process group name will be the hash of all the ranks in the group.

    :::
    Note

    To enable `backend == Backend.MPI`, PyTorch needs to be built from source on a system that supports MPI.
    :::

    :::
    Note

    Support for multiple backends is experimental. Currently when no backend is specified, both `gloo` and `nccl` backends will be created. The `gloo` backend will be used for collectives with CPU tensors and the `nccl` backend will be used for collectives with CUDA tensors. A custom backend can be specified by passing in a string with format "\<device\_type\>:\<backend\_name\>,\<device\_type\>:\<backend\_name\>", e.g. "cpu:gloo,cuda:custom\_backend".
    :::

```{=html}
<!-- -->
```

torch.distributed.device\_mesh.init\_device\_mesh[(]*device\_type*, *mesh\_shape*, *\**, *mesh\_dim\_names=None*, *backend\_override=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L1277)

:   Initializes a DeviceMesh based on device\_type, mesh\_shape, and mesh\_dim\_names parameters.

    This creates a DeviceMesh with an n-dimensional array layout, where n is the length of mesh\_shape. If mesh\_dim\_names is provided, each dimension is labeled as mesh\_dim\_names\[i\].

    :::
    Note

    init\_device\_mesh follows SPMD programming model, meaning the same PyTorch Python program runs on all processes/ranks in the cluster. Ensure mesh\_shape (the dimensions of the nD array describing device layout) is identical across all ranks. Inconsistent mesh\_shape may lead to hanging.
    :::

    :::
    Note

    If no process group is found, init\_device\_mesh will initialize distributed process group/groups required for distributed communications behind the scene.
    :::

    Parameters[:]

    :   -   **device\_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like", "xpu". Passing in a device type with a GPU index, such as "cuda:0", is not allowed.

        -   **mesh\_shape** (*Tuple\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) -- A tuple defining the dimensions of the multi-dimensional array describing the layout of devices.

        -   **mesh\_dim\_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *\...\],* *optional*) -- A tuple of mesh dimension names to assign to each dimension of the multi-dimensional array describing the layout of devices. Its length must match the length of mesh\_shape. Each string in mesh\_dim\_names must be unique.

        -   **backend\_override** (*Dict\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Options\]* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Options\],* *optional*) -- Overrides for some or all of the ProcessGroups that will be created for each mesh dimension. Each key can be either the index of a dimension or its name (if mesh\_dim\_names is provided). Each value can be a tuple containing the name of the backend and its options, or just one of these two components (in which case the other will be set to its default value).

    Returns[:]

    :   A [`DeviceMesh`](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh") object representing the device layout.

    Return type[:]

    :   [DeviceMesh](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh")

    Example:

    :::
    :::
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>>
        >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
        >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.is\_initialized[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1310)

:   Check if the default process group has been initialized.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.is\_mpi\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1248)

:   Check if the MPI backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.is\_nccl\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1253)

:   Check if the NCCL backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.is\_gloo\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1258)

:   Check if the Gloo backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_xccl\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1268)

:   Check if the XCCL backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.batch\_isend\_irecv[(]*p2p\_op\_list*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2786)

:   Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in `p2p_op_list` and return the corresponding requests. NCCL, Gloo, and UCC backend are currently supported.

    Parameters[:]

    :   **p2p\_op\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*P2POp*](#torch.distributed.P2POp "torch.distributed.distributed_c10d.P2POp")*\]*) -- A list of point-to-point operations(type of each operator is `torch.distributed.P2POp`). The order of the isend/irecv in the list matters and it needs to match with corresponding isend/irecv on the remote end.

    Returns[:]

    :   A list of distributed request objects returned by calling the corresponding op in the op\_list.

    Return type[:]

    :   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[*Work*](#torch.distributed.Work "torch.distributed.distributed_c10d.Work")\]

    Examples

    :::
    :::
        >>> send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
        >>> recv_tensor = torch.randn(2, dtype=torch.float32)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
        >>> recv_op = dist.P2POp(
        ...     dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
        ... )
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1
    :::
    :::

    :::
    Note

    Note that when this API is used with the NCCL PG backend, users must set the current GPU device with torch.cuda.set\_device, otherwise it will lead to unexpected hang issues.

    In addition, if this API is the first collective call in the `group` passed to `dist.P2POp`, all ranks of the `group` must participate in this API call; otherwise, the behavior is undefined. If this API call is not the first collective call in the `group`, batched P2P operations involving only a subset of ranks of the `group` are allowed.
    :::

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.destroy\_process\_group[(]*group=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2222)

:   Destroy a given process group, and deinitialize the distributed package.

    Parameters[:]

    :   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup")*,* *optional*) -- The process group to be destroyed, if group.WORLD is given, all process groups including the default one will be destroyed.

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_backend\_available[(]*backend*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1285)

:   Check backend availability.

    Checks if the given backend is available and supports the built-in backends or third-party backends through function `Backend.register_backend`.

    Parameters[:]

    :   **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- Backend name.

    Returns[:]

    :   Returns true if the backend is available otherwise false.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.irecv[(]*tensor*, *src=None*, *group=None*, *tag=0*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2494)

:   Receives a tensor asynchronously.

    :::
    Warning

    `tag` is not supported with the NCCL backend.
    :::

    Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to fill with received data.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Source rank on global process group (regardless of `group` argument). Will receive from any process if unspecified.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **tag** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Tag to match recv with remote send

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `src` and `group_src`.

    Returns[:]

    :   A distributed request object. None, if not part of the group

    Return type[:]

    :   [*Work*](#torch.distributed.Work "torch.distributed.distributed_c10d.Work") \| None

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_gloo\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1258)

:   Check if the Gloo backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_initialized[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1310)

:   Check if the default process group has been initialized.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_mpi\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1248)

:   Check if the MPI backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_nccl\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1253)

:   Check if the NCCL backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_torchelastic\_launched[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1315)

:   Check whether this process was launched with `torch.distributed.elastic` (aka torchelastic).

    The existence of `TORCHELASTIC_RUN_ID` environment variable is used as a proxy to determine whether the current process was launched with torchelastic. This is a reasonable proxy since `TORCHELASTIC_RUN_ID` maps to the rendezvous id which is always a non-null value indicating the job id for peer discovery purposes..

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.is\_ucc\_available[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1263)

:   Check if the UCC backend is available.

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.is\_torchelastic\_launched[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1315)

:   Check whether this process was launched with `torch.distributed.elastic` (aka torchelastic).

    The existence of `TORCHELASTIC_RUN_ID` environment variable is used as a proxy to determine whether the current process was launched with torchelastic. This is a reasonable proxy since `TORCHELASTIC_RUN_ID` maps to the rendezvous id which is always a non-null value indicating the job id for peer discovery purposes..

    Return type[:]

    :   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.get\_default\_backend\_for\_device[(]*device*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1413)

:   Return the default backend for the given device.

    Parameters[:]

    :   **device** (*Union\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*torch.device*](tensor_attributes.html#torch.device "torch.device")*\]*) -- The device to get the default backend for.

    Returns[:]

    :   The default backend for the given device as a lower case string.

    Return type[:]

    :   [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

------------------------------------------------------------------------

Currently three initialization methods are supported:


### TCP initialization

There are two ways to initialize using TCP, both requiring a network address reachable from all processes and a desired `world_size`. The first way requires specifying an address that belongs to the rank 0 process. This initialization method requires that all processes have manually specified ranks.

Note that multicast address is not supported anymore in the latest distributed package. `group_name` is deprecated as well.


    import torch.distributed as dist

    # Use address of one of the machines
    dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                            rank=args.rank, world_size=4)


### Shared file-system initialization

Another initialization method makes use of a file system that is shared and visible from all machines in a group, along with a desired `world_size`. The URL should start with `file://` and contain a path to a non-existent file (in an existing directory) on a shared file system. File-system initialization will automatically create that file if it doesn't exist, but will not delete the file. Therefore, it is your responsibility to make sure that the file is cleaned up before the next [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") call on the same file path/name.

Note that automatic rank assignment is not supported anymore in the latest distributed package and `group_name` is deprecated as well.


Warning

This method assumes that the file system supports locking using `fcntl` - most local systems and NFS support it.

Warning

This method will always create the file and try its best to clean up and remove the file at the end of the program. In other words, each initialization with the file init method will need a brand new empty file in order for the initialization to succeed. If the same file used by the previous initialization (which happens not to get cleaned up) is used again, this is unexpected behavior and can often cause deadlocks and failures. Therefore, even though this method will try its best to clean up the file, if the auto-delete happens to be unsuccessful, it is your responsibility to ensure that the file is removed at the end of the training to prevent the same file to be reused again during the next time. This is especially important if you plan to call [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") multiple times on the same file name. In other words, if the file is not removed/cleaned up and you call [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") again on that file, failures are expected. The rule of thumb here is that, make sure that the file is non-existent or empty every time [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") is called.

    import torch.distributed as dist

    # rank should always be specified
    dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                            world_size=4, rank=args.rank)


### Environment variable initialization

This method will read the configuration from environment variables, allowing one to fully customize how the information is obtained. The variables to be set are:

-   `MASTER_PORT` - required; has to be a free port on machine with rank 0

-   `MASTER_ADDR` - required (except for rank 0); address of rank 0 node

-   `WORLD_SIZE` - required; can be set either here, or in a call to init function

-   `RANK` - required; can be set either here, or in a call to init function

The machine with rank 0 will be used to set up all connections.

This is the default method, meaning that `init_method` does not have to be specified (or can be `env://`).

### Improving initialization time

-   `TORCH_GLOO_LAZY_INIT` - establishes connections on demand rather than using a full mesh which can greatly improve initialization time for non all2all operations.


Post-Initialization
---------------------------------------------------------------------------------

Once [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") was run, the following functions can be used. To check whether the process group has already been initialized use [`torch.distributed.is_initialized()`](#torch.distributed.is_initialized "torch.distributed.is_initialized").

*[class][ ]*torch.distributed.Backend[(]*name*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L242)

:   An enum-like class for backends.

    Available backends: GLOO, NCCL, UCC, MPI, XCCL, and other registered backends.

    The values of this class are lowercase strings, e.g., `"gloo"`. They can be accessed as attributes, e.g., `Backend.NCCL`.

    This class can be directly called to parse the string, e.g., `Backend(backend_str)` will check if `backend_str` is valid, and return the parsed lowercase string if so. It also accepts uppercase strings, e.g., `Backend("GLOO")` returns `"gloo"`.

    :::
    Note

    The entry `Backend.UNDEFINED` is present but only used as initial value of some fields. Users should neither use it directly nor assume its existence.
    :::

```{=html}
<!-- -->
```

*[classmethod][ ]*register\_backend[(]*name*, *func*, *extended\_api=False*, *devices=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L309)

:   Register a new backend with the given name and instantiating function.

    This class method is used by 3rd party `ProcessGroup` extension to register new backends.

    Parameters[:]

    :   -   **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- Backend name of the `ProcessGroup` extension. It should match the one in `init_process_group()`.

        -   **func** (*function*) -- Function handler that instantiates the backend. The function should be implemented in the backend extension and takes four arguments, including `store`, `rank`, `world_size`, and `timeout`.

        -   **extended\_api** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether the backend supports extended argument structure. Default: `False`. If set to `True`, the backend will get an instance of `c10d::DistributedBackendOptions`, and a process group options object as defined by the backend implementation.

        -   **device** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *optional*) -- device type this backend supports, e.g. "cpu", "cuda", etc. If None, assuming both "cpu" and "cuda"

    :::
    Note

    This support of 3rd party backend is experimental and subject to change.
    :::

```{=html}
<!-- -->
```

torch.distributed.get\_backend[(]*group=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1387)

:   Return the backend of the given process group.

    Parameters[:]

    :   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. The default is the general main process group. If another specific group is specified, the calling process must be part of `group`.

    Returns[:]

    :   The backend of the given process group as a lower case string.

    Return type[:]

    :   [*Backend*](#torch.distributed.Backend "torch.distributed.distributed_c10d.Backend")

```{=html}
<!-- -->
```

torch.distributed.get\_rank[(]*group=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2403)

:   Return the rank of the current process in the provided `group`, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed process group. They are always consecutive integers ranging from 0 to `world_size`.

    Parameters[:]

    :   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

    Returns[:]

    :   The rank of the process group -1, if not part of the group

    Return type[:]

    :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

```{=html}
<!-- -->
```

torch.distributed.get\_world\_size[(]*group=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2430)

:   Return the number of processes in the current process group.

    Parameters[:]

    :   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

    Returns[:]

    :   The world size of the process group -1, if not part of the group

    Return type[:]

    :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

Shutdown
-----------------------------------------------------------

It is important to clean up resources on exit by calling `destroy_process_group()`.

The simplest pattern to follow is to destroy every process group and backend by calling `destroy_process_group()` with the default value of None for the `group` argument, at a point in the training script where communications are no longer needed, usually near the end of main(). The call should be made once per trainer-process, not at the outer process-launcher level.

if `destroy_process_group()` is not called by all ranks in a pg within the timeout duration, especially when there are multiple process-groups in the application e.g. for N-D parallelism, hangs on exit are possible. This is because the destructor for ProcessGroupNCCL calls ncclCommAbort, which must be called collectively, but the order of calling ProcessGroupNCCL's destructor if called by python's GC is not deterministic. Calling `destroy_process_group()` helps by ensuring ncclCommAbort is called in a consistent order across ranks, and avoids calling ncclCommAbort during ProcessGroupNCCL's destructor.


### Reinitialization

`destroy_process_group` can also be used to destroy individual process groups. One use case could be fault tolerant training, where a process group may be destroyed and then a new one initialized during runtime. In this case, it's critical to synchronize the trainer processes using some means other than torch.distributed primitives \_after\_ calling destroy and before subsequently initializing. This behavior is currently unsupported/untested, due to the difficulty of achieving this synchronization, and is considered a known issue. Please file a github issue or RFC if this is a use case that's blocking you.


------------------------------------------------------------------------


Groups
-------------------------------------------------------

By default collectives operate on the default group (also called the world) and require all processes to enter the distributed function call. However, some workloads can benefit from more fine-grained communication. This is where distributed groups come into play. [`new_group()`](#torch.distributed.new_group "torch.distributed.new_group") function can be used to create new groups, with arbitrary subsets of all processes. It returns an opaque group handle that can be given as a `group` argument to all collectives (collectives are distributed functions to exchange information in certain well-known programming patterns).

torch.distributed.new\_group[(]*ranks=None*, *timeout=None*, *backend=None*, *pg\_options=None*, *use\_local\_synchronization=False*, *group\_desc=None*, *device\_id=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L5296)

:   Create a new distributed group.

    This function requires that all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group. Additionally, groups should be created in the same order in all processes.

    :::
    Warning

    Safe concurrent usage: When using multiple process groups with the `NCCL` backend, the user must ensure a globally consistent execution order of collectives across ranks.

    If multiple threads within a process issue collectives, explicit synchronization is necessary to ensure consistent ordering.

    When using async variants of torch.distributed communication APIs, a work object is returned and the communication kernel is enqueued on a separate CUDA stream, allowing overlap of communication and computation. Once one or more async ops have been issued on one process group, they must be synchronized with other cuda streams by calling work.wait() before using another process group.

    See Using multiple NCCL communicators concurrently \<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html\#using-multiple-nccl-communicators-concurrently\> for more details.
    :::

    Parameters[:]

    :   -   **ranks** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) -- List of ranks of group members. If `None`, will be set to all ranks. Default is `None`.

        -   **timeout** (*timedelta,* *optional*) -- see init\_process\_group for details and default value.

        -   **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *or* [*Backend*](#torch.distributed.Backend "torch.distributed.Backend")*,* *optional*) -- The backend to use. Depending on build-time configurations, valid values are `gloo` and `nccl`. By default uses the same backend as the global group. This field should be given as a lowercase string (e.g., `"gloo"`), which can also be accessed via [`Backend`](#torch.distributed.Backend "torch.distributed.Backend") attributes (e.g., `Backend.GLOO`). If `None` is passed in, the backend corresponding to the default process group will be used. Default is `None`.

        -   **pg\_options** (*ProcessGroupOptions,* *optional*) -- process group options specifying what additional options need to be passed in during the construction of specific process groups. i.e. for the `nccl` backend, `is_high_priority_stream` can be specified so that process group can pick up high priority cuda streams. For other available options to config nccl, See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-tuse_local_synchronization> (bool, optional): perform a group-local barrier at the end of the process group creation. This is different in that non-member ranks don't need to call into API and don't join the barrier.

        -   **group\_desc** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *optional*) -- a string to describe the process group.

        -   **device\_id** ([*torch.device*](tensor_attributes.html#torch.device "torch.device")*,* *optional*) -- a single, specific device to "bind" this process to, The new\_group call will try to initialize a communication backend immediately for the device if this field is given.

    Returns[:]

    :   A handle of distributed group that can be given to collective calls or GroupMember.NON\_GROUP\_MEMBER if the rank is not part of `ranks`.

    N.B. use\_local\_synchronization doesn't work with MPI.

    N.B. While use\_local\_synchronization=True can be significantly faster with larger clusters and small process groups, care must be taken since it changes cluster behavior as non-member ranks don't join the group barrier().

    N.B. use\_local\_synchronization=True can lead to deadlocks when each rank creates multiple overlapping process groups. To avoid that, make sure all ranks follow the same global creation order.

```{=html}
<!-- -->
```

torch.distributed.distributed\_c10d.shrink\_group[(]*ranks\_to\_exclude*, *group=None*, *shrink\_flags=0*, *pg\_options=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L5776)

:   Shrinks a process group by excluding specified ranks.

    Creates and returns a new, smaller process group comprising only the ranks from the original group that were not in the `ranks_to_exclude` list.

    Parameters[:]

    :   -   **ranks\_to\_exclude** (*List\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) -- A list of ranks from the original `group` to exclude from the new group.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup")*,* *optional*) -- The process group to shrink. If `None`, the default process group is used. Defaults to `None`.

        -   **shrink\_flags** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Flags to control the shrinking behavior. Can be `SHRINK_DEFAULT` (default) or `SHRINK_ABORT`. `SHRINK_ABORT` will attempt to terminate ongoing operations in the parent communicator before shrinking. Defaults to `SHRINK_DEFAULT`.

        -   **pg\_options** (*ProcessGroupOptions,* *optional*) -- Backend-specific options to apply to the shrunken process group. If provided, the backend will use these options when creating the new group. If omitted, the new group inherits defaults from the parent.

    Returns[:]

    :   a new group comprised of the remaining ranks. If the default group was shrunk, the returned group becomes the new default group.

    Return type[:]

    :   [ProcessGroup](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup")

    Raises[:]

    :   -   [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") -- if the group's backend does not support shrinking.

        -   [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") -- if `ranks_to_exclude` is invalid (empty, out of bounds,

        -   **duplicates, or** **excludes all ranks).** --

        -   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.14)") -- if an excluded rank calls this function or the backend

        -   **fails the operation.** --

    Notes

    -   Only non-excluded ranks should call this function; excluded ranks must not participate in the shrink operation.

    -   Shrinking the default group destroys all other process groups since rank reassignment makes them inconsistent.

```{=html}
<!-- -->
```

torch.distributed.get\_group\_rank[(]*group*, *global\_rank*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1029)

:   Translate a global rank into a group rank.

    `global_rank` must be part of `group` otherwise this raises RuntimeError.

    Parameters[:]

    :   -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")) -- ProcessGroup to find the relative rank.

        -   **global\_rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Global rank to query.

    Returns[:]

    :   Group rank of `global_rank` relative to `group`

    Return type[:]

    :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

    N.B. calling this function on the default process group returns identity

```{=html}
<!-- -->
```

torch.distributed.get\_global\_rank[(]*group*, *group\_rank*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1057)

:   Translate a group rank into a global rank.

    `group_rank` must be part of group otherwise this raises RuntimeError.

    Parameters[:]

    :   -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")) -- ProcessGroup to find the global rank from.

        -   **group\_rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Group rank to query.

    Returns[:]

    :   Global rank of `group_rank` relative to `group`

    Return type[:]

    :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

    N.B. calling this function on the default process group returns identity

```{=html}
<!-- -->
```

torch.distributed.get\_process\_group\_ranks[(]*group*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1095)

:   Get all ranks associated with `group`.

    Parameters[:]

    :   **group** (*Optional\[*[*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*\]*) -- ProcessGroup to get all ranks from. If None, the default process group will be used.

    Returns[:]

    :   List of global ranks ordered by group rank.

    Return type[:]

    :   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]

DeviceMesh
---------------------------------------------------------------

DeviceMesh is a higher level abstraction that manages process groups (or NCCL communicators). It allows user to easily create inter node and intra node process groups without worrying about how to set up the ranks correctly for different sub process groups, and it helps manage those distributed process group easily. [`init_device_mesh()`](#torch.distributed.device_mesh.init_device_mesh "torch.distributed.device_mesh.init_device_mesh") function can be used to create new DeviceMesh, with a mesh shape describing the device topology.

*[class][ ]*torch.distributed.device\_mesh.DeviceMesh[(]*device\_type*, *mesh=None*, *\**, *mesh\_dim\_names=None*, *backend\_override=None*, *\_init\_backend=True*, *\_rank=None*, *\_layout=None*, *\_rank\_map=None*, *\_root\_mesh=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L129)

:   DeviceMesh represents a mesh of devices, where layout of devices could be represented as a n-d dimension array, and each value of the n-d dimensional array is the global id of the default process group ranks.

    DeviceMesh could be used to setup the N dimensional device connections across the cluster, and manage the ProcessGroups for N dimensional parallelisms. Communications could happen on each dimension of the DeviceMesh separately. DeviceMesh respects the device that user selects already (i.e. if user call torch.cuda.set\_device before the DeviceMesh initialization), and will select/set the device for the current process if user does not set the device beforehand. Note that manual device selection should happen BEFORE the DeviceMesh initialization.

    DeviceMesh can also be used as a context manager when using together with DTensor APIs.

    :::
    Note

    DeviceMesh follows SPMD programming model, which means the same PyTorch Python program is running on all processes/ranks in the cluster. Therefore, users need to make sure the mesh array (which describes the layout of devices) should be identical across all ranks. Inconsistent mesh will lead to silent hang.
    :::

    Parameters[:]

    :   -   **device\_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".

        -   **mesh** (*ndarray*) -- A multi-dimensional array or an integer tensor describing the layout of devices, where the IDs are global IDs of the default process group.

        -   **\_rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- (experimental/internal) The global rank of the current process. If not provided, it will be inferred from the default process group.

    Returns[:]

    :   A [`DeviceMesh`](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh") object representing the device layout.

    Return type[:]

    :   [DeviceMesh](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh")

    The following program runs on each process/rank in an SPMD manner. In this example, we have 2 hosts with 4 GPUs each. A reduction over the first dimension of mesh will reduce across columns (0, 4), .. and (3, 7), a reduction over the second dimension of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7).

    Example:

    :::
    :::
        >>> from torch.distributed.device_mesh import DeviceMesh
        >>>
        >>> # Initialize device mesh as (2, 4) to represent the topology
        >>> # of cross-host(dim 0), and within-host (dim 1).
        >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
    :::
    :::

    *[property][ ]*device\_type*:[ ]str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*

    :   Returns the device type of the mesh.

    *[static][ ]*from\_group[(]*group*, *device\_type*, *mesh=None*, *\**, *mesh\_dim\_names=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L895)

    :   Constructs a [`DeviceMesh`](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh") with `device_type` from an existing `ProcessGroup` or a list of existing `ProcessGroup`.

        The constructed device mesh has number of dimensions equal to the number of groups passed. For example, if a single process group is passed in, the resulted DeviceMesh is a 1D mesh. If a list of 2 process groups is passed in, the resulted DeviceMesh is a 2D mesh.

        If more than one group is passed, then the `mesh` and `mesh_dim_names` arguments are required. The order of the process groups passed in determines the topology of the mesh. For example, the first process group will be the 0th dimension of the DeviceMesh. The mesh tensor passed in must have the same number of dimensions as the number of process groups passed in, and the order of the dimensions in the mesh tensor must match the order in the process groups passed in.

        Parameters[:]

        :   -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*\]*) -- the existing ProcessGroup or a list of existing ProcessGroups.

            -   **device\_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like". Passing in a device type with a GPU index, such as "cuda:0", is not allowed.

            -   **mesh** ([*torch.Tensor*](tensors.html#torch.Tensor "torch.Tensor") *or* *ArrayLike,* *optional*) -- A multi-dimensional array or an integer tensor describing the layout of devices, where the IDs are global IDs of the default process group. Default is None.

            -   **mesh\_dim\_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *\...\],* *optional*) -- A tuple of mesh dimension names to assign to each dimension of the multi-dimensional array describing the layout of devices. Its length must match the length of mesh\_shape. Each string in mesh\_dim\_names must be unique. Default is None.

        Returns[:]

        :   A [`DeviceMesh`](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh") object representing the device layout.

        Return type[:]

        :   [DeviceMesh](#torch.distributed.device_mesh.DeviceMesh "torch.distributed.device_mesh.DeviceMesh")

    get\_all\_groups[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L663)

    :   Returns a list of ProcessGroups for all mesh dimensions.

        Returns[:]

        :   A list of `ProcessGroup` object.

        Return type[:]

        :   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup")\]

    get\_coordinate[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L1053)

    :   Return the relative indices of this rank relative to all dimensions of the mesh. If this rank is not part of the mesh, return None.

        Return type[:]

        :   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| None

    get\_group[(]*mesh\_dim=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L617)

    :   Returns the single ProcessGroup specified by mesh\_dim, or, if mesh\_dim is not specified and the DeviceMesh is 1-dimensional, returns the only ProcessGroup in the mesh.

        Parameters[:]

        :   -   **mesh\_dim** (*str/python:int,* *optional*) -- it can be the name of the mesh dimension or the index

            -   **None.** (*of the mesh dimension. Default is*) --

        Returns[:]

        :   A `ProcessGroup` object.

        Return type[:]

        :   [*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup")

    get\_local\_rank[(]*mesh\_dim=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L1009)

    :   Returns the local rank of the given mesh\_dim of the DeviceMesh.

        Parameters[:]

        :   -   **mesh\_dim** (*str/python:int,* *optional*) -- it can be the name of the mesh dimension or the index

            -   **None.** (*of the mesh dimension. Default is*) --

        Returns[:]

        :   An integer denotes the local rank.

        Return type[:]

        :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2 hosts with 4 GPUs each. Calling mesh\_2d.get\_local\_rank(mesh\_dim=0) on rank 0, 1, 2, 3 would return 0. Calling mesh\_2d.get\_local\_rank(mesh\_dim=0) on rank 4, 5, 6, 7 would return 1. Calling mesh\_2d.get\_local\_rank(mesh\_dim=1) on rank 0, 4 would return 0. Calling mesh\_2d.get\_local\_rank(mesh\_dim=1) on rank 1, 5 would return 1. Calling mesh\_2d.get\_local\_rank(mesh\_dim=1) on rank 2, 6 would return 2. Calling mesh\_2d.get\_local\_rank(mesh\_dim=1) on rank 3, 7 would return 3.

        Example:

        :::
        :::
            >>> from torch.distributed.device_mesh import DeviceMesh
            >>>
            >>> # Initialize device mesh as (2, 4) to represent the topology
            >>> # of cross-host(dim 0), and within-host (dim 1).
            >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
        :::
        :::

    get\_rank[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/device_mesh.py#L1003)

    :   Returns the current global rank.

        Return type[:]

        :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

    *[property][ ]*mesh*:[ ]Tensor(tensors.html#torch.Tensor "torch.Tensor")*

    :   Returns the tensor representing the layout of devices.

    *[property][ ]*mesh\_dim\_names*:[ ]tuple(https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"),[ ]\...\][ ]\|[ ]None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*

    :   Returns the names of mesh dimensions.

Point-to-point communication
---------------------------------------------------------------------------------------------------

torch.distributed.send[(]*tensor*, *dst=None*, *group=None*, *tag=0*, *group\_dst=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2539)

:   Send a tensor synchronously.

    :::
    Warning

    `tag` is not supported with the NCCL backend.
    :::

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to send.

        -   **dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Destination rank on global process group (regardless of `group` argument). Destination rank should not be the same as the rank of the current process.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **tag** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Tag to match send with remote recv

        -   **group\_dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `dst` and `group_dst`.

```{=html}
<!-- -->
```

torch.distributed.recv[(]*tensor*, *src=None*, *group=None*, *tag=0*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2571)

:   Receives a tensor synchronously.

    :::
    Warning

    `tag` is not supported with the NCCL backend.
    :::

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to fill with received data.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Source rank on global process group (regardless of `group` argument). Will receive from any process if unspecified.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **tag** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Tag to match recv with remote send

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `src` and `group_src`.

    Returns[:]

    :   Sender rank -1, if not part of the group

    Return type[:]

    :   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

[`isend()`](#torch.distributed.isend "torch.distributed.isend") and [`irecv()`](#torch.distributed.irecv "torch.distributed.irecv") return distributed request objects when used. In general, the type of this object is unspecified as they should never be created manually, but they are guaranteed to support two methods:

-   `is_completed()` - returns True if the operation has finished

-   `wait()` - will block the process until the operation is finished. `is_completed()` is guaranteed to return True once it returns.

torch.distributed.isend[(]*tensor*, *dst=None*, *group=None*, *tag=0*, *group\_dst=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2449)

:   Send a tensor asynchronously.

    :::
    Warning

    Modifying `tensor` before the request completes causes undefined behavior.
    :::

    :::
    Warning

    `tag` is not supported with the NCCL backend.
    :::

    Unlike send, which is blocking, isend allows src == dst rank, i.e. send to self.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to send.

        -   **dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Destination rank on global process group (regardless of `group` argument)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **tag** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Tag to match send with remote recv

        -   **group\_dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `dst` and `group_dst`

    Returns[:]

    :   A distributed request object. None, if not part of the group

    Return type[:]

    :   [*Work*](#torch.distributed.Work "torch.distributed.distributed_c10d.Work") \| None

```{=html}
<!-- -->
```

torch.distributed.irecv[(]*tensor*, *src=None*, *group=None*, *tag=0*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2494)

:   Receives a tensor asynchronously.

    :::
    Warning

    `tag` is not supported with the NCCL backend.
    :::

    Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to fill with received data.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Source rank on global process group (regardless of `group` argument). Will receive from any process if unspecified.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **tag** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Tag to match recv with remote send

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `src` and `group_src`.

    Returns[:]

    :   A distributed request object. None, if not part of the group

    Return type[:]

    :   [*Work*](#torch.distributed.Work "torch.distributed.distributed_c10d.Work") \| None

```{=html}
<!-- -->
```

torch.distributed.send\_object\_list[(]*object\_list*, *dst=None*, *group=None*, *device=None*, *group\_dst=None*, *use\_batch=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3402)

:   Sends picklable objects in `object_list` synchronously.

    Similar to [`send()`](#torch.distributed.send "torch.distributed.send"), but Python objects can be passed in. Note that all objects in `object_list` must be picklable in order to be sent.

    Parameters[:]

    :   -   **object\_list** (*List\[Any\]*) -- List of input objects to sent. Each object must be picklable. Receiver must provide lists of equal sizes.

        -   **dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Destination rank to send `object_list` to. Destination rank is based on global process group (regardless of `group` argument)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup") *\|* *None*) -- (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is `None`.

        -   **device** (`torch.device`, optional) -- If not None, the objects are serialized and converted to tensors which are moved to the `device` before sending. Default is `None`.

        -   **group\_dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Must specify one of `dst` and `group_dst` but not both

        -   **use\_batch** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- If True, use batch p2p operations instead of regular send operations. This avoids initializing 2-rank communicators and uses existing entire group communicators. See batch\_isend\_irecv for usage and assumptions. Default is `False`.

    Returns[:]

    :   `None`.

    :::
    Note

    For NCCL-based process groups, internal tensor representations of objects must be moved to the GPU device before communication takes place. In this case, the device used is given by `torch.cuda.current_device()` and it is the user's responsibility to ensure that this is set so that each rank has an individual GPU, via `torch.cuda.set_device()`.
    :::

    :::
    Warning

    Object collectives have a number of serious performance and scalability limitations. See Object collectives(#object-collectives) for details.
    :::

    :::
    Warning

    [`send_object_list()`](#torch.distributed.send_object_list "torch.distributed.send_object_list") uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Only call this function with data you trust.
    :::

    :::
    Warning

    Calling [`send_object_list()`](#torch.distributed.send_object_list "torch.distributed.send_object_list") with GPU tensors is not well supported and inefficient as it incurs GPU -\> CPU transfer since tensors would be pickled. Please consider using [`send()`](#torch.distributed.send "torch.distributed.send") instead.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> # Assumes backend is not NCCL
            >>> device = torch.device("cpu")
            >>> if dist.get_rank() == 0:
            >>>     # Assumes world_size of 2.
            >>>     objects = ["foo", 12, {1: 2}] # any picklable object
            >>>     dist.send_object_list(objects, dst=1, device=device)
            >>> else:
            >>>     objects = [None, None, None]
            >>>     dist.recv_object_list(objects, src=0, device=device)
            >>> objects
            ['foo', 12, {1: 2}]
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.recv\_object\_list[(]*object\_list*, *src=None*, *group=None*, *device=None*, *group\_src=None*, *use\_batch=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3520)

:   Receives picklable objects in `object_list` synchronously.

    Similar to [`recv()`](#torch.distributed.recv "torch.distributed.recv"), but can receive Python objects.

    Parameters[:]

    :   -   **object\_list** (*List\[Any\]*) -- List of objects to receive into. Must provide a list of sizes equal to the size of the list being sent.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Source rank from which to recv `object_list`. Source rank is based on global process group (regardless of `group` argument) Will receive from any rank if set to None. Default is `None`.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup") *\|* *None*) -- (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is `None`.

        -   **device** (`torch.device`, optional) -- If not None, receives on this device. Default is `None`.

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `src` and `group_src`.

        -   **use\_batch** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- If True, use batch p2p operations instead of regular send operations. This avoids initializing 2-rank communicators and uses existing entire group communicators. See batch\_isend\_irecv for usage and assumptions. Default is `False`.

    Returns[:]

    :   Sender rank. -1 if rank is not part of the group. If rank is part of the group, `object_list` will contain the sent objects from `src` rank.

    :::
    Note

    For NCCL-based process groups, internal tensor representations of objects must be moved to the GPU device before communication takes place. In this case, the device used is given by `torch.cuda.current_device()` and it is the user's responsibility to ensure that this is set so that each rank has an individual GPU, via `torch.cuda.set_device()`.
    :::

    :::
    Warning

    Object collectives have a number of serious performance and scalability limitations. See Object collectives(#object-collectives) for details.
    :::

    :::
    Warning

    [`recv_object_list()`](#torch.distributed.recv_object_list "torch.distributed.recv_object_list") uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Only call this function with data you trust.
    :::

    :::
    Warning

    Calling [`recv_object_list()`](#torch.distributed.recv_object_list "torch.distributed.recv_object_list") with GPU tensors is not well supported and inefficient as it incurs GPU -\> CPU transfer since tensors would be pickled. Please consider using [`recv()`](#torch.distributed.recv "torch.distributed.recv") instead.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> # Assumes backend is not NCCL
            >>> device = torch.device("cpu")
            >>> if dist.get_rank() == 0:
            >>>     # Assumes world_size of 2.
            >>>     objects = ["foo", 12, {1: 2}] # any picklable object
            >>>     dist.send_object_list(objects, dst=1, device=device)
            >>> else:
            >>>     objects = [None, None, None]
            >>>     dist.recv_object_list(objects, src=0, device=device)
            >>> objects
            ['foo', 12, {1: 2}]
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.batch\_isend\_irecv[(]*p2p\_op\_list*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2786)

:   Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in `p2p_op_list` and return the corresponding requests. NCCL, Gloo, and UCC backend are currently supported.

    Parameters[:]

    :   **p2p\_op\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*P2POp*](#torch.distributed.P2POp "torch.distributed.distributed_c10d.P2POp")*\]*) -- A list of point-to-point operations(type of each operator is `torch.distributed.P2POp`). The order of the isend/irecv in the list matters and it needs to match with corresponding isend/irecv on the remote end.

    Returns[:]

    :   A list of distributed request objects returned by calling the corresponding op in the op\_list.

    Return type[:]

    :   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[*Work*](#torch.distributed.Work "torch.distributed.distributed_c10d.Work")\]

    Examples

    :::
    :::
        >>> send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
        >>> recv_tensor = torch.randn(2, dtype=torch.float32)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
        >>> recv_op = dist.P2POp(
        ...     dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
        ... )
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1
    :::
    :::

    :::
    Note

    Note that when this API is used with the NCCL PG backend, users must set the current GPU device with torch.cuda.set\_device, otherwise it will lead to unexpected hang issues.

    In addition, if this API is the first collective call in the `group` passed to `dist.P2POp`, all ranks of the `group` must participate in this API call; otherwise, the behavior is undefined. If this API call is not the first collective call in the `group`, batched P2P operations involving only a subset of ranks of the `group` are allowed.
    :::

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.P2POp[(]*op*, *tensor*, *peer=None*, *group=None*, *tag=0*, *group\_peer=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L486)

:   A class to build point-to-point operations for `batch_isend_irecv`.

    This class builds the type of P2P operation, communication buffer, peer rank, Process Group, and tag. Instances of this class will be passed to `batch_isend_irecv` for point-to-point communications.

    Parameters[:]

    :   -   **op** (*Callable*) -- A function to send data to or receive data from a peer process. The type of `op` is either `torch.distributed.isend` or `torch.distributed.irecv`.

        -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to send or receive.

        -   **peer** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination or source rank.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **tag** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Tag to match send with recv.

        -   **group\_peer** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination or source rank.

Synchronous and asynchronous collective operations
-----------------------------------------------------------------------------------------------------------------------------------------------

Every collective operation function supports the following two kinds of operations, depending on the setting of the `async_op` flag passed into the collective:

**Synchronous operation** - the default mode, when `async_op` is set to `False`. When the function returns, it is guaranteed that the collective operation is performed. In the case of CUDA operations, it is not guaranteed that the CUDA operation is completed, since CUDA operations are asynchronous. For CPU collectives, any further function calls utilizing the output of the collective call will behave as expected. For CUDA collectives, function calls utilizing the output on the same CUDA stream will behave as expected. Users must take care of synchronization under the scenario of running under different streams. For details on CUDA semantics such as stream synchronization, see [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html). See the below script to see examples of differences in these semantics for CPU and CUDA operations.

**Asynchronous operation** - when `async_op` is set to True. The collective operation function returns a distributed request object. In general, you don't need to create it manually and it is guaranteed to support two methods:

-   `is_completed()` - in the case of CPU collectives, returns `True` if completed. In the case of CUDA operations, returns `True` if the operation has been successfully enqueued onto a CUDA stream and the output can be utilized on the default stream without further synchronization.

-   `wait()` - in the case of CPU collectives, will block the process until the operation is completed. In the case of CUDA collectives, will block the currently active CUDA stream until the operation is completed (but will not block the CPU).

-   `get_future()` - returns `torch._C.Future` object. Supported for NCCL, also supported for most operations on GLOO and MPI, except for peer to peer operations. Note: as we continue adopting Futures and merging APIs, `get_future()` call might become redundant.

**Example**

The following code can serve as a reference regarding semantics for CUDA operations when using distributed collectives. It shows the explicit need to synchronize when using collective outputs on different CUDA streams:


    # Code runs on each rank.
    dist.init_process_group("nccl", rank=rank, world_size=2)
    output = torch.tensor([rank]).cuda(rank)
    s = torch.cuda.Stream()
    handle = dist.all_reduce(output, async_op=True)
    # Wait ensures the operation is enqueued, but not necessarily complete.
    handle.wait()
    # Using result on non-default stream.
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        output.add_(100)
    if rank == 0:
        # if the explicit call to wait_stream was omitted, the output below will be
        # non-deterministically 1 or 101, depending on whether the allreduce overwrote
        # the value after the add completed.
        print(output)


Collective functions
-----------------------------------------------------------------------------------

torch.distributed.broadcast[(]*tensor*, *src=None*, *group=None*, *async\_op=False*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2865)

:   Broadcasts the tensor to the whole group.

    `tensor` must have the same number of elements in all processes participating in the collective.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Data to be sent if `src` is the rank of current process, and tensor to be used to save received data otherwise.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Source rank on global process group (regardless of `group` argument).

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Source rank on `group`. Must specify one of `group_src` and `src` but not both.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

```{=html}
<!-- -->
```

torch.distributed.broadcast\_object\_list[(]*object\_list*, *src=None*, *group=None*, *device=None*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3662)

:   Broadcasts picklable objects in `object_list` to the whole group.

    Similar to [`broadcast()`](#torch.distributed.broadcast "torch.distributed.broadcast"), but Python objects can be passed in. Note that all objects in `object_list` must be picklable in order to be broadcasted.

    Parameters[:]

    :   -   **object\_list** (*List\[Any\]*) -- List of input objects to broadcast. Each object must be picklable. Only objects on the `src` rank will be broadcast, but each rank must provide lists of equal sizes.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Source rank from which to broadcast `object_list`. Source rank is based on global process group (regardless of `group` argument)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup") *\|* *None*) -- (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is `None`.

        -   **device** (`torch.device`, optional) -- If not None, the objects are serialized and converted to tensors which are moved to the `device` before broadcasting. Default is `None`.

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Source rank on `group`. Must not specify one of `group_src` and `src` but not both.

    Returns[:]

    :   `None`. If rank is part of the group, `object_list` will contain the broadcasted objects from `src` rank.

    :::
    Note

    For NCCL-based process groups, internal tensor representations of objects must be moved to the GPU device before communication takes place. In this case, the device used is given by `torch.cuda.current_device()` and it is the user's responsibility to ensure that this is set so that each rank has an individual GPU, via `torch.cuda.set_device()`.
    :::

    :::
    Note

    Note that this API differs slightly from the [`broadcast()`](#torch.distributed.broadcast "torch.distributed.broadcast") collective since it does not provide an `async_op` handle and thus will be a blocking call.
    :::

    :::
    Warning

    Object collectives have a number of serious performance and scalability limitations. See Object collectives(#object-collectives) for details.
    :::

    :::
    Warning

    [`broadcast_object_list()`](#torch.distributed.broadcast_object_list "torch.distributed.broadcast_object_list") uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Only call this function with data you trust.
    :::

    :::
    Warning

    Calling [`broadcast_object_list()`](#torch.distributed.broadcast_object_list "torch.distributed.broadcast_object_list") with GPU tensors is not well supported and inefficient as it incurs GPU -\> CPU transfer since tensors would be pickled. Please consider using [`broadcast()`](#torch.distributed.broadcast "torch.distributed.broadcast") instead.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> if dist.get_rank() == 0:
            >>>     # Assumes world_size of 3.
            >>>     objects = ["foo", 12, {1: 2}] # any picklable object
            >>> else:
            >>>     objects = [None, None, None]
            >>> # Assumes backend is not NCCL
            >>> device = torch.device("cpu")
            >>> dist.broadcast_object_list(objects, src=0, device=device)
            >>> objects
            ['foo', 12, {1: 2}]
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.all\_reduce[(]*tensor*, *[[op=\<RedOpType.SUM:] [0\>]]*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L2917)

:   Reduces the tensor data across all machines in a way that all get the final result.

    After the call `tensor` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Input and output of the collective. The function operates in-place.

        -   **op** (*optional*) -- One of the values from `torch.distributed.ReduceOp` enum. Specifies an operation used for element-wise reductions.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

    Examples

    :::
    :::
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1
    :::
    :::

    :::
    :::
        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor(
        ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
        ... ) + 2 * rank * (1 + 1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j], device='cuda:0') # Rank 0
        tensor([4.+4.j, 6.+6.j], device='cuda:1') # Rank 1
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.reduce[(]*tensor*, *dst=None*, *[[op=\<RedOpType.SUM:] [0\>]]*, *group=None*, *async\_op=False*, *group\_dst=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3087)

:   Reduces the tensor data across all machines.

    Only the process with rank `dst` is going to receive the final result.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Input and output of the collective. The function operates in-place.

        -   **dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Destination rank on global process group (regardless of `group` argument)

        -   **op** (*optional*) -- One of the values from `torch.distributed.ReduceOp` enum. Specifies an operation used for element-wise reductions.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

        -   **group\_dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Destination rank on `group`. Must specify one of `group_dst` and `dst` but not both.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

```{=html}
<!-- -->
```

torch.distributed.all\_gather[(]*tensor\_list*, *tensor*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3929)

:   Gathers tensors from the whole group in a list.

    Complex and uneven sized tensors are supported.

    Parameters[:]

    :   -   **tensor\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Tensor*](tensors.html#torch.Tensor "torch.Tensor")*\]*) -- Output list. It should contain correctly-sized tensors to be used for output of the collective. Uneven sized tensors are supported.

        -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to be broadcast from current process.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

    Examples

    :::
    :::
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_list = [
        ...     torch.zeros(2, dtype=torch.int64, device=device) for _ in range(2)
        ... ]
        >>> tensor_list
        [tensor([0, 0], device='cuda:0'), tensor([0, 0], device='cuda:0')] # Rank 0
        [tensor([0, 0], device='cuda:1'), tensor([0, 0], device='cuda:1')] # Rank 1
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2], device='cuda:0'), tensor([3, 4], device='cuda:0')] # Rank 0
        [tensor([1, 2], device='cuda:1'), tensor([3, 4], device='cuda:1')] # Rank 1
    :::
    :::

    :::
    :::
        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [
        ...     torch.zeros(2, dtype=torch.cfloat, device=device) for _ in range(2)
        ... ]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j], device='cuda:0'), tensor([0.+0.j, 0.+0.j], device='cuda:0')] # Rank 0
        [tensor([0.+0.j, 0.+0.j], device='cuda:1'), tensor([0.+0.j, 0.+0.j], device='cuda:1')] # Rank 1
        >>> tensor = torch.tensor(
        ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
        ... ) + 2 * rank * (1 + 1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device='cuda:0') # Rank 0
        tensor([3.+3.j, 4.+4.j], device='cuda:1') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j], device='cuda:0'), tensor([3.+3.j, 4.+4.j], device='cuda:0')] # Rank 0
        [tensor([1.+1.j, 2.+2.j], device='cuda:1'), tensor([3.+3.j, 4.+4.j], device='cuda:1')] # Rank 1
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.all\_gather\_into\_tensor[(]*output\_tensor*, *input\_tensor*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4029)

:   Gather tensors from all ranks and put them in a single output tensor.

    This function requires all tensors to be the same size on each process.

    Parameters[:]

    :   -   **output\_tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Output tensor to accommodate tensor elements from all ranks. It must be correctly sized to have one of the following forms: (i) a concatenation of all the input tensors along the primary dimension; for definition of "concatenation", see `torch.cat()`; (ii) a stack of all the input tensors along the primary dimension; for definition of "stack", see `torch.stack()`. Examples below may better explain the supported output forms.

        -   **input\_tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Tensor to be gathered from current rank. Different from the `all_gather` API, the input tensors in this API must have the same size across all ranks.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

    Examples

    :::
    :::
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor_in
        tensor([1, 2], device='cuda:0') # Rank 0
        tensor([3, 4], device='cuda:1') # Rank 1
        >>> # Output in concatenation form
        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([1, 2, 3, 4], device='cuda:0') # Rank 0
        tensor([1, 2, 3, 4], device='cuda:1') # Rank 1
        >>> # Output in stack form
        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> tensor_out2
        tensor([[1, 2],
                [3, 4]], device='cuda:0') # Rank 0
        tensor([[1, 2],
                [3, 4]], device='cuda:1') # Rank 1
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.all\_gather\_object[(]*object\_list*, *obj*, *group=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3176)

:   Gathers picklable objects from the whole group into a list.

    Similar to [`all_gather()`](#torch.distributed.all_gather "torch.distributed.all_gather"), but Python objects can be passed in. Note that the object must be picklable in order to be gathered.

    Parameters[:]

    :   -   **object\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[Any\]*) -- Output list. It should be correctly sized as the size of the group for this collective and will contain the output.

        -   **obj** (*Any*) -- Pickable Python object to be broadcast from current process.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used. Default is `None`.

    Returns[:]

    :   None. If the calling rank is part of this group, the output of the collective will be populated into the input `object_list`. If the calling rank is not part of the group, the passed in `object_list` will be unmodified.

    :::
    Note

    Note that this API differs slightly from the [`all_gather()`](#torch.distributed.all_gather "torch.distributed.all_gather") collective since it does not provide an `async_op` handle and thus will be a blocking call.
    :::

    :::
    Note

    For NCCL-based processed groups, internal tensor representations of objects must be moved to the GPU device before communication takes place. In this case, the device used is given by `torch.cuda.current_device()` and it is the user's responsibility to ensure that this is set so that each rank has an individual GPU, via `torch.cuda.set_device()`.
    :::

    :::
    Warning

    Object collectives have a number of serious performance and scalability limitations. See Object collectives(#object-collectives) for details.
    :::

    :::
    Warning

    [`all_gather_object()`](#torch.distributed.all_gather_object "torch.distributed.all_gather_object") uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Only call this function with data you trust.
    :::

    :::
    Warning

    Calling [`all_gather_object()`](#torch.distributed.all_gather_object "torch.distributed.all_gather_object") with GPU tensors is not well supported and inefficient as it incurs GPU -\> CPU transfer since tensors would be pickled. Please consider using [`all_gather()`](#torch.distributed.all_gather "torch.distributed.all_gather") instead.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> # Assumes world_size of 3.
            >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
            >>> output = [None for _ in gather_objects]
            >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
            >>> output
            ['foo', 12, {1: 2}]
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.gather[(]*tensor*, *gather\_list=None*, *dst=None*, *group=None*, *async\_op=False*, *group\_dst=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4268)

:   Gathers a list of tensors in a single process.

    This function requires all tensors to be the same size on each process.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Input tensor.

        -   **gather\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Tensor*](tensors.html#torch.Tensor "torch.Tensor")*\],* *optional*) -- List of appropriately, same-sized tensors to use for gathered data (default is None, must be specified on the destination rank)

        -   **dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on global process group (regardless of `group` argument). (If both `dst` and `group_dst` are None, default is global rank 0)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

        -   **group\_dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `dst` and `group_dst`

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

    :::
    Note

    Note that all Tensors in gather\_list must have the same size.
    :::

    Example::

    :   :::
        :::
            >>> # We have 2 process groups, 2 ranks.
            >>> tensor_size = 2
            >>> device = torch.device(f'cuda:{rank}')
            >>> tensor = torch.ones(tensor_size, device=device) + rank
            >>> if dist.get_rank() == 0:
            >>>     gather_list = [torch.zeros_like(tensor, device=device) for i in range(2)]
            >>> else:
            >>>     gather_list = None
            >>> dist.gather(tensor, gather_list, dst=0)
            >>> # Rank 0 gets gathered data.
            >>> gather_list
            [tensor([1., 1.], device='cuda:0'), tensor([2., 2.], device='cuda:0')] # Rank 0
            None                                                                   # Rank 1
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.gather\_object[(]*obj*, *object\_gather\_list=None*, *dst=None*, *group=None*, *group\_dst=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3271)

:   Gathers picklable objects from the whole group in a single process.

    Similar to [`gather()`](#torch.distributed.gather "torch.distributed.gather"), but Python objects can be passed in. Note that the object must be picklable in order to be gathered.

    Parameters[:]

    :   -   **obj** (*Any*) -- Input object. Must be picklable.

        -   **object\_gather\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[Any\]*) -- Output list. On the `dst` rank, it should be correctly sized as the size of the group for this collective and will contain the output. Must be `None` on non-dst ranks. (default is `None`)

        -   **dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on global process group (regardless of `group` argument). (If both `dst` and `group_dst` are None, default is global rank 0)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup") *\|* *None*) -- (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is `None`.

        -   **group\_dst** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Destination rank on `group`. Invalid to specify both `dst` and `group_dst`

    Returns[:]

    :   None. On the `dst` rank, `object_gather_list` will contain the output of the collective.

    :::
    Note

    Note that this API differs slightly from the gather collective since it does not provide an async\_op handle and thus will be a blocking call.
    :::

    :::
    Note

    For NCCL-based processed groups, internal tensor representations of objects must be moved to the GPU device before communication takes place. In this case, the device used is given by `torch.cuda.current_device()` and it is the user's responsibility to ensure that this is set so that each rank has an individual GPU, via `torch.cuda.set_device()`.
    :::

    :::
    Warning

    Object collectives have a number of serious performance and scalability limitations. See Object collectives(#object-collectives) for details.
    :::

    :::
    Warning

    [`gather_object()`](#torch.distributed.gather_object "torch.distributed.gather_object") uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Only call this function with data you trust.
    :::

    :::
    Warning

    Calling [`gather_object()`](#torch.distributed.gather_object "torch.distributed.gather_object") with GPU tensors is not well supported and inefficient as it incurs GPU -\> CPU transfer since tensors would be pickled. Please consider using [`gather()`](#torch.distributed.gather "torch.distributed.gather") instead.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> # Assumes world_size of 3.
            >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
            >>> output = [None for _ in gather_objects]
            >>> dist.gather_object(
            ...     gather_objects[dist.get_rank()],
            ...     output if dist.get_rank() == 0 else None,
            ...     dst=0
            ... )
            >>> # On rank 0
            >>> output
            ['foo', 12, {1: 2}]
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.scatter[(]*tensor*, *scatter\_list=None*, *src=None*, *group=None*, *async\_op=False*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4351)

:   Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the `tensor` argument.

    Complex tensors are supported.

    Parameters[:]

    :   -   **tensor** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Output tensor.

        -   **scatter\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Tensor*](tensors.html#torch.Tensor "torch.Tensor")*\]*) -- List of tensors to scatter (default is None, must be specified on the source rank)

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Source rank on global process group (regardless of `group` argument). (If both `src` and `group_src` are None, default is global rank 0)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Source rank on `group`. Invalid to specify both `src` and `group_src`

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

    :::
    Note

    Note that all Tensors in scatter\_list must have the same size.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> tensor_size = 2
            >>> device = torch.device(f'cuda:{rank}')
            >>> output_tensor = torch.zeros(tensor_size, device=device)
            >>> if dist.get_rank() == 0:
            >>>     # Assumes world_size of 2.
            >>>     # Only tensors, all of which must be the same size.
            >>>     t_ones = torch.ones(tensor_size, device=device)
            >>>     t_fives = torch.ones(tensor_size, device=device) * 5
            >>>     scatter_list = [t_ones, t_fives]
            >>> else:
            >>>     scatter_list = None
            >>> dist.scatter(output_tensor, scatter_list, src=0)
            >>> # Rank i gets scatter_list[i].
            >>> output_tensor
            tensor([1., 1.], device='cuda:0') # Rank 0
            tensor([5., 5.], device='cuda:1') # Rank 1
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.scatter\_object\_list[(]*scatter\_object\_output\_list*, *scatter\_object\_input\_list=None*, *src=None*, *group=None*, *group\_src=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L3794)

:   Scatters picklable objects in `scatter_object_input_list` to the whole group.

    Similar to [`scatter()`](#torch.distributed.scatter "torch.distributed.scatter"), but Python objects can be passed in. On each rank, the scattered object will be stored as the first element of `scatter_object_output_list`. Note that all objects in `scatter_object_input_list` must be picklable in order to be scattered.

    Parameters[:]

    :   -   **scatter\_object\_output\_list** (*List\[Any\]*) -- Non-empty list whose first element will store the object scattered to this rank.

        -   **scatter\_object\_input\_list** (*List\[Any\],* *optional*) -- List of input objects to scatter. Each object must be picklable. Only objects on the `src` rank will be scattered, and the argument can be `None` for non-src ranks.

        -   **src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Source rank from which to scatter `scatter_object_input_list`. Source rank is based on global process group (regardless of `group` argument). (If both `src` and `group_src` are None, default is global rank 0)

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed.distributed_c10d.ProcessGroup") *\|* *None*) -- (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is `None`.

        -   **group\_src** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- Source rank on `group`. Invalid to specify both `src` and `group_src`

    Returns[:]

    :   `None`. If rank is part of the group, `scatter_object_output_list` will have its first element set to the scattered object for this rank.

    :::
    Note

    Note that this API differs slightly from the scatter collective since it does not provide an `async_op` handle and thus will be a blocking call.
    :::

    :::
    Warning

    Object collectives have a number of serious performance and scalability limitations. See Object collectives(#object-collectives) for details.
    :::

    :::
    Warning

    [`scatter_object_list()`](#torch.distributed.scatter_object_list "torch.distributed.scatter_object_list") uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Only call this function with data you trust.
    :::

    :::
    Warning

    Calling [`scatter_object_list()`](#torch.distributed.scatter_object_list "torch.distributed.scatter_object_list") with GPU tensors is not well supported and inefficient as it incurs GPU -\> CPU transfer since tensors would be pickled. Please consider using [`scatter()`](#torch.distributed.scatter "torch.distributed.scatter") instead.
    :::

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> if dist.get_rank() == 0:
            >>>     # Assumes world_size of 3.
            >>>     objects = ["foo", 12, {1: 2}] # any picklable object
            >>> else:
            >>>     # Can be any list on non-src ranks, elements are not used.
            >>>     objects = [None, None, None]
            >>> output_list = [None]
            >>> dist.scatter_object_list(output_list, objects, src=0)
            >>> # Rank i gets objects[i]. For example, on rank 2:
            >>> output_list
            [{1: 2}]
        :::
        :::

```{=html}
<!-- -->
```

torch.distributed.reduce\_scatter[(]*output*, *input\_list*, *[[op=\<RedOpType.SUM:] [0\>]]*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4456)

:   Reduces, then scatters a list of tensors to all processes in a group.

    Parameters[:]

    :   -   **output** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Output tensor.

        -   **input\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Tensor*](tensors.html#torch.Tensor "torch.Tensor")*\]*) -- List of tensors to reduce and scatter.

        -   **op** (*optional*) -- One of the values from `torch.distributed.ReduceOp` enum. Specifies an operation used for element-wise reductions.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group.

```{=html}
<!-- -->
```

torch.distributed.reduce\_scatter\_tensor[(]*output*, *input*, *[[op=\<RedOpType.SUM:] [0\>]]*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4501)

:   Reduces, then scatters a tensor to all ranks in a group.

    Parameters[:]

    :   -   **output** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Output tensor. It should have the same size across all ranks.

        -   **input** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Input tensor to be reduced and scattered. Its size should be output tensor size times the world size. The input tensor can have one of the following shapes: (i) a concatenation of the output tensors along the primary dimension, or (ii) a stack of the output tensors along the primary dimension. For definition of "concatenation", see `torch.cat()`. For definition of "stack", see `torch.stack()`.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group.

    Examples

    :::
    :::
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_out = torch.zeros(2, dtype=torch.int64, device=device)
        >>> # Input in concatenation form
        >>> tensor_in = torch.arange(world_size * 2, dtype=torch.int64, device=device)
        >>> tensor_in
        tensor([0, 1, 2, 3], device='cuda:0') # Rank 0
        tensor([0, 1, 2, 3], device='cuda:1') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1
        >>> # Input in stack form
        >>> tensor_in = torch.reshape(tensor_in, (world_size, 2))
        >>> tensor_in
        tensor([[0, 1],
                [2, 3]], device='cuda:0') # Rank 0
        tensor([[0, 1],
                [2, 3]], device='cuda:1') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device='cuda:0') # Rank 0
        tensor([4, 6], device='cuda:1') # Rank 1
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.all\_to\_all\_single[(]*output*, *input*, *output\_split\_sizes=None*, *input\_split\_sizes=None*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4632)

:   Split input tensor and then scatter the split list to all processes in a group.

    Later the received tensors are concatenated from all the processes in the group and returned as a single output tensor.

    Complex tensors are supported.

    Parameters[:]

    :   -   **output** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Gathered concatenated output tensor.

        -   **input** ([*Tensor*](tensors.html#torch.Tensor "torch.Tensor")) -- Input tensor to scatter.

        -   **output\_split\_sizes** -- (list\[Int\], optional): Output split sizes for dim 0 if specified None or empty, dim 0 of `output` tensor must divide equally by `world_size`.

        -   **input\_split\_sizes** -- (list\[Int\], optional): Input split sizes for dim 0 if specified None or empty, dim 0 of `input` tensor must divide equally by `world_size`.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group.

    :::
    Warning

    all\_to\_all\_single is experimental and subject to change.
    :::

    Examples

    :::
    :::
        >>> input = torch.arange(4) + rank * 4
        >>> input
        tensor([0, 1, 2, 3])     # Rank 0
        tensor([4, 5, 6, 7])     # Rank 1
        tensor([8, 9, 10, 11])   # Rank 2
        tensor([12, 13, 14, 15]) # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([0, 4, 8, 12])    # Rank 0
        tensor([1, 5, 9, 13])    # Rank 1
        tensor([2, 6, 10, 14])   # Rank 2
        tensor([3, 7, 11, 15])   # Rank 3
    :::
    :::

    :::
    :::
        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = list(input.chunk(world_size))
        >>> gather_list = list(output.chunk(world_size))
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)
    :::
    :::

    :::
    :::
        >>> # Another example with uneven split
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> output = ...
        >>> dist.all_to_all_single(output, input, output_splits, input_splits)
        >>> output
        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3
    :::
    :::

    :::
    :::
        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor(
        ...     [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.cfloat
        ... ) + 4 * rank * (1 + 1j)
        >>> input
        tensor([1+1j, 2+2j, 3+3j, 4+4j])                                # Rank 0
        tensor([5+5j, 6+6j, 7+7j, 8+8j])                                # Rank 1
        tensor([9+9j, 10+10j, 11+11j, 12+12j])                          # Rank 2
        tensor([13+13j, 14+14j, 15+15j, 16+16j])                        # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([1+1j, 5+5j, 9+9j, 13+13j])                              # Rank 0
        tensor([2+2j, 6+6j, 10+10j, 14+14j])                            # Rank 1
        tensor([3+3j, 7+7j, 11+11j, 15+15j])                            # Rank 2
        tensor([4+4j, 8+8j, 12+12j, 16+16j])                            # Rank 3
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.all\_to\_all[(]*output\_tensor\_list*, *input\_tensor\_list*, *group=None*, *async\_op=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4781)

:   Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Complex tensors are supported.

    Parameters[:]

    :   -   **output\_tensor\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Tensor*](tensors.html#torch.Tensor "torch.Tensor")*\]*) -- List of tensors to be gathered one per rank.

        -   **input\_tensor\_list** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Tensor*](tensors.html#torch.Tensor "torch.Tensor")*\]*) -- List of tensors to scatter one per rank.

        -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group.

    :::
    Warning

    all\_to\_all is experimental and subject to change.
    :::

    Examples

    :::
    :::
        >>> input = torch.arange(4) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3
    :::
    :::

    :::
    :::
        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = input
        >>> gather_list = output
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)
    :::
    :::

    :::
    :::
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> input = list(input.split(input_splits))
        >>> input
        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0
        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1
        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2
        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3
        >>> output = ...
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0
        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1
        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2
        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3
    :::
    :::

    :::
    :::
        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor(
        ...     [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.cfloat
        ... ) + 4 * rank * (1 + 1j)
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([1+1j]), tensor([2+2j]), tensor([3+3j]), tensor([4+4j])]            # Rank 0
        [tensor([5+5j]), tensor([6+6j]), tensor([7+7j]), tensor([8+8j])]            # Rank 1
        [tensor([9+9j]), tensor([10+10j]), tensor([11+11j]), tensor([12+12j])]      # Rank 2
        [tensor([13+13j]), tensor([14+14j]), tensor([15+15j]), tensor([16+16j])]    # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([1+1j]), tensor([5+5j]), tensor([9+9j]), tensor([13+13j])]          # Rank 0
        [tensor([2+2j]), tensor([6+6j]), tensor([10+10j]), tensor([14+14j])]        # Rank 1
        [tensor([3+3j]), tensor([7+7j]), tensor([11+11j]), tensor([15+15j])]        # Rank 2
        [tensor([4+4j]), tensor([8+8j]), tensor([12+12j]), tensor([16+16j])]        # Rank 3
    :::
    :::

```{=html}
<!-- -->
```

torch.distributed.barrier[(]*group=None*, *async\_op=False*, *device\_ids=None*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4905)

:   Synchronize all processes.

    This collective blocks processes until the whole group enters this function, if async\_op is False, or if async work handle is called on wait().

    Parameters[:]

    :   -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If None, the default process group will be used.

        -   **async\_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether this op should be an async op

        -   **device\_ids** (*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* *optional*) -- List of device/GPU ids. Only one id is expected.

    Returns[:]

    :   Async work handle, if async\_op is set to True. None, if not async\_op or if not part of the group

    :::
    Note

    ProcessGroupNCCL now blocks the cpu thread till the completion of the barrier collective.
    :::

    :::
    Note

    ProcessGroupNCCL implements barrier as an all\_reduce of a 1-element tensor. A device must be chosen for allocating this tensor. The device choice is made by checking in this order (1) the first device passed to device\_ids arg of barrier if not None, (2) the device passed to init\_process\_group if not None, (3) the device that was first used with this process group, if another collective with tensor inputs has been performed, (4) the device index indicated by the global rank mod local device count.
    :::

```{=html}
<!-- -->
```

torch.distributed.monitored\_barrier[(]*group=None*, *timeout=None*, *wait\_all\_ranks=False*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L4980)

:   Synchronize processes similar to `torch.distributed.barrier`, but consider a configurable timeout.

    It is able to report ranks that did not pass this barrier within the provided timeout. Specifically, for non-zero ranks, will block until a send/recv is processed from rank 0. Rank 0 will block until all send /recv from other ranks are processed, and will report failures for ranks that failed to respond in time. Note that if one rank does not reach the monitored\_barrier (for example due to a hang), all other ranks would fail in monitored\_barrier.

    This collective will block all processes/ranks in the group, until the whole group exits the function successfully, making it useful for debugging and synchronizing. However, it can have a performance impact and should only be used for debugging or scenarios that require full synchronization points on the host-side. For debugging purposes, this barrier can be inserted before the application's collective calls to check if any ranks are desynchronized.

    :::
    Note

    Note that this collective is only supported with the GLOO backend.
    :::

    Parameters[:]

    :   -   **group** ([*ProcessGroup*](distributed._dist2.html#torch.distributed._dist2.ProcessGroup "torch.distributed._dist2.ProcessGroup")*,* *optional*) -- The process group to work on. If `None`, the default process group will be used.

        -   **timeout** ([*datetime.timedelta*](https://docs.python.org/3/library/datetime.html#datetime.timedelta "(in Python v3.14)")*,* *optional*) -- Timeout for monitored\_barrier. If `None`, the default process group timeout will be used.

        -   **wait\_all\_ranks** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether to collect all failed ranks or not. By default, this is `False` and `monitored_barrier` on rank 0 will throw on the first failed rank it encounters in order to fail fast. By setting `wait_all_ranks=True` `monitored_barrier` will collect all failed ranks and throw an error containing information about all failed ranks.

    Returns[:]

    :   `None`.

    Example::

    :   :::
        :::
            >>> # Note: Process group initialization omitted on each rank.
            >>> import torch.distributed as dist
            >>> if dist.get_rank() != 1:
            >>>     dist.monitored_barrier() # Raises exception indicating that
            >>> # rank 1 did not call into monitored_barrier.
            >>> # Example with wait_all_ranks=True
            >>> if dist.get_rank() == 0:
            >>>     dist.monitored_barrier(wait_all_ranks=True) # Raises exception
            >>> # indicating that ranks 1, 2, ... world_size - 1 did not call into
            >>> # monitored_barrier.
        :::
        :::

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.Work

:   A Work object represents the handle to a pending asynchronous operation in PyTorch's distributed package. It is returned by non-blocking collective operations, such as dist.all\_reduce(tensor, async\_op=True).

    block\_current\_stream[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Blocks the currently active GPU stream on the operation to complete. For GPU based collectives this is equivalent to synchronize. For CPU initiated collectives such as with Gloo this will block the CUDA stream until the operation is complete.

        This returns immediately in all cases.

        To check whether an operation was successful you should check the Work object result asynchronously.

    boxed[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [object(https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")]]

    :

    exception[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] std::\_\_exception\_ptr::exception\_ptr]

    :

    get\_future[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] torch.Future]

    :

        Returns[:]

        :   A `torch.futures.Future` object which is associated with the completion of the `Work`. As an example, a future object can be retrieved by `fut = process_group.allreduce(tensors).get_future()`.

        Example::

        :   Below is an example of a simple allreduce DDP communication hook that uses `get_future` API to retrieve a Future associated with the completion of `allreduce`.

            :::
            :::
                >>> def allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket): -> torch.futures.Future
                >>>     group_to_use = process_group if process_group is not None else torch.distributed.group.WORLD
                >>>     tensor = bucket.buffer().div_(group_to_use.size())
                >>>     return torch.distributed.all_reduce(tensor, group=group_to_use, async_op=True).get_future()
                >>> ddp_model.register_comm_hook(state=None, hook=allreduce)
            :::
            :::

        :::
        Warning

        `get_future` API supports NCCL, and partially GLOO and MPI backends (no support for peer-to-peer operations like send/recv) and will return a `torch.futures.Future`.

        In the example above, `allreduce` work will be done on GPU using NCCL backend, `fut.wait()` will return after synchronizing the appropriate NCCL streams with PyTorch's current device streams to ensure we can have asynchronous CUDA execution and it does not wait for the entire operation to complete on GPU. Note that `CUDAFuture` does not support `TORCH_NCCL_BLOCKING_WAIT` flag or NCCL's `barrier()`. In addition, if a callback function was added by `fut.then()`, it will wait until `WorkNCCL`'s NCCL streams synchronize with `ProcessGroupNCCL`'s dedicated callback stream and invoke the callback inline after running the callback on the callback stream. `fut.then()` will return another `CUDAFuture` that holds the return value of the callback and a `CUDAEvent` that recorded the callback stream.

        > <div>
        >
        > 1.  For CPU work, `fut.done()` returns true when work has been completed and value() tensors are ready.
        >
        > 2.  For GPU work, `fut.done()` returns true only whether the operation has been enqueued.
        >
        > 3.  For mixed CPU-GPU work (e.g. sending GPU tensors with GLOO), `fut.done()` returns true when tensors have arrived on respective nodes, but not yet necessarily synched on respective GPUs (similarly to GPU work).
        >
        > </div>
        :::

    get\_future\_result[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] torch.Future]

    :

        Returns[:]

        :   A `torch.futures.Future` object of int type which maps to the enum type of WorkResult As an example, a future object can be retrieved by `fut = process_group.allreduce(tensor).get_future_result()`.

        Example::

        :   users can use `fut.wait()` to blocking wait for the completion of the work and get the WorkResult by `fut.value()`. Also, users can use `fut.then(call_back_func)` to register a callback function to be called when the work is completed, without blocking the current thread.

        :::
        Warning

        `get_future_result` API supports NCCL
        :::

    is\_completed[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")]]

    :

    is\_success[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")]]

    :

    result[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [list(https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[torch.Tensor(tensors.html#torch.Tensor "torch.Tensor")\]]]

    :

    source\_rank[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [int(https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")]]

    :

    synchronize[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :

    *[static][ ]*unbox[(]*arg0:[ ][object(https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")]*[)] [[→] torch.\_C.\_distributed\_c10d.Work]

    :

    wait[(]*self:[ ]torch.\_C.\_distributed\_c10d.Work*, *timeout:[ ][datetime.timedelta(https://docs.python.org/3/library/datetime.html#datetime.timedelta "(in Python v3.14)")][ ]=[ ]datetime.timedelta(0)*[)] [[→] [bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")]]

    :

        Returns[:]

        :   true/false.

        Example::

        :

            try:

            :   work.wait(timeout)

            except:

            :   \# some handling

        :::
        Warning

        In normal cases, users do not need to set the timeout. calling wait() is the same as calling synchronize(): Letting the current stream block on the completion of the NCCL work. However, if timeout is set, it will block the CPU thread until the NCCL work is completed or timed out. If timeout, exception will be thrown.
        :::

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.ReduceOp

:   An enum-like class for available reduction operations: `SUM`, `PRODUCT`, `MIN`, `MAX`, `BAND`, `BOR`, `BXOR`, and `PREMUL_SUM`.

    `BAND`, `BOR`, and `BXOR` reductions are not available when using the `NCCL` backend.

    `AVG` divides values by the world size before summing across ranks. `AVG` is only available with the `NCCL` backend, and only for NCCL versions 2.10 or later.

    `PREMUL_SUM` multiplies inputs by a given scalar locally before reduction. `PREMUL_SUM` is only available with the `NCCL` backend, and only available for NCCL versions 2.11 or later. Users are supposed to use `torch.distributed._make_nccl_premul_sum`.

    Additionally, `MAX`, `MIN` and `PRODUCT` are not supported for complex tensors.

    The values of this class can be accessed as attributes, e.g., `ReduceOp.SUM`. They are used in specifying strategies for reduction collectives, e.g., [`reduce()`](#torch.distributed.reduce "torch.distributed.reduce").

    This class does not support `__members__` property.

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.reduce\_op

:   Deprecated enum-like class for reduction operations: `SUM`, `PRODUCT`, `MIN`, and `MAX`.

    [`ReduceOp`](#torch.distributed.ReduceOp "torch.distributed.ReduceOp") is recommended to use instead.

Distributed Key-Value Store
-------------------------------------------------------------------------------------------------

The distributed package comes with a distributed key-value store, which can be used to share information between processes in the group as well as to initialize the distributed package in [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") (by explicitly creating the store as an alternative to specifying `init_method`.) There are 3 choices for Key-Value Stores: [`TCPStore`](#torch.distributed.TCPStore "torch.distributed.TCPStore"), [`FileStore`](#torch.distributed.FileStore "torch.distributed.FileStore"), and [`HashStore`](#torch.distributed.HashStore "torch.distributed.HashStore").

*[class][ ]*torch.distributed.Store

:   Base class for all store implementations, such as the 3 provided by PyTorch distributed: ([`TCPStore`](#torch.distributed.TCPStore "torch.distributed.TCPStore"), [`FileStore`](#torch.distributed.FileStore "torch.distributed.FileStore"), and [`HashStore`](#torch.distributed.HashStore "torch.distributed.HashStore")).

    \_\_init\_\_[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :

    add[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *arg1:[ ][SupportsInt(https://docs.python.org/3/library/typing.html#typing.SupportsInt "(in Python v3.14)")]*[)] [[→] [int(https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")]]

    :   The first call to add for a given `key` creates a counter associated with `key` in the store, initialized to `amount`. Subsequent calls to add with the same `key` increment the counter by the specified `amount`. Calling `add()` with a key that has already been set in the store by `set()` will result in an exception.

        Parameters[:]

        :   -   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key in the store whose counter will be incremented.

            -   **amount** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- The quantity by which the counter will be incremented.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, other store types can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.add("first_key", 1)
                >>> store.add("first_key", 6)
                >>> # Should return 7
                >>> store.get("first_key")
            :::
            :::

    append[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *arg1:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Append the key-value pair into the store based on the supplied `key` and `value`. If `key` does not exists in the store, it will be created.

        Parameters[:]

        :   -   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key to be appended to the store.

            -   **value** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The value associated with `key` to be added to the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.append("first_key", "po")
                >>> store.append("first_key", "tato")
                >>> # Should return "potato"
                >>> store.get("first_key")
            :::
            :::

    check[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][collections.abc.Sequence(https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\]]*[)] [[→] [bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")]]

    :   The call to check whether a given list of `keys` have value stored in the store. This call immediately returns in normal cases but still suffers from some edge deadlock cases, e.g, calling check after TCPStore has been destroyed. Calling `check()` with a list of keys that one wants to check whether stored in the store or not.

        Parameters[:]

        :   **keys** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) -- The keys to query whether stored in the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, other store types can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.add("first_key", 1)
                >>> # Should return 7
                >>> store.check(["first_key"])
            :::
            :::

    clone[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*[)] [[→] torch.\_C.\_distributed\_c10d.Store]

    :   Clones the store and returns a new object that points to the same underlying store. The returned store can be used concurrently with the original object. This is intended to provide a safe way to use a store from multiple threads by cloning one store per thread.

    compare\_set[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *arg1:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *arg2:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [bytes(https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")]]

    :   Inserts the key-value pair into the store based on the supplied `key` and performs comparison between `expected_value` and `desired_value` before inserting. `desired_value` will only be set if `expected_value` for the `key` already exists in the store or if `expected_value` is an empty string.

        Parameters[:]

        :   -   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key to be checked in the store.

            -   **expected\_value** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The value associated with `key` to be checked before insertion.

            -   **desired\_value** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The value associated with `key` to be added to the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set("key", "first_value")
                >>> store.compare_set("key", "first_value", "second_value")
                >>> # Should return "second_value"
                >>> store.get("key")
            :::
            :::

    delete\_key[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")]]

    :   Deletes the key-value pair associated with `key` from the store. Returns true if the key was successfully deleted, and false if it was not.

        :::
        Warning

        The `delete_key` API is only supported by the [`TCPStore`](#torch.distributed.TCPStore "torch.distributed.TCPStore") and [`HashStore`](#torch.distributed.HashStore "torch.distributed.HashStore"). Using this API with the [`FileStore`](#torch.distributed.FileStore "torch.distributed.FileStore") will result in an exception.
        :::

        Parameters[:]

        :   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key to be deleted from the store

        Returns[:]

        :   True if `key` was deleted, otherwise False.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, HashStore can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set("first_key")
                >>> # This should return true
                >>> store.delete_key("first_key")
                >>> # This should return false
                >>> store.delete_key("bad_key")
            :::
            :::

    get[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [bytes(https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")]]

    :   Retrieves the value associated with the given `key` in the store. If `key` is not present in the store, the function will wait for `timeout`, which is defined when initializing the store, before throwing an exception.

        Parameters[:]

        :   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The function will return the value associated with this key.

        Returns[:]

        :   Value associated with `key` if `key` is in the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set("first_key", "first_value")
                >>> # Should return "first_value"
                >>> store.get("first_key")
            :::
            :::

    has\_extended\_api[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*[)] [[→] [bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")]]

    :   Returns true if the store supports extended operations.

    list\_keys[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*[)] [[→] [list(https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\]]]

    :   Returns a list of all keys in the store.

    multi\_get[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][collections.abc.Sequence(https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\]]*[)] [[→] [list(https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[bytes(https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")\]]]

    :   Retrieve all values in `keys`. If any key in `keys` is not present in the store, the function will wait for `timeout`

        Parameters[:]

        :   **keys** (*List\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) -- The keys to be retrieved from the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set("first_key", "po")
                >>> store.set("second_key", "tato")
                >>> # Should return [b"po", b"tato"]
                >>> store.multi_get(["first_key", "second_key"])
            :::
            :::

    multi\_set[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][collections.abc.Sequence(https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\]]*, *arg1:[ ][collections.abc.Sequence(https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\]]*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Inserts a list key-value pair into the store based on the supplied `keys` and `values`

        Parameters[:]

        :   -   **keys** (*List\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) -- The keys to insert.

            -   **values** (*List\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) -- The values to insert.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.multi_set(["first_key", "second_key"], ["po", "tato"])
                >>> # Should return b"po"
                >>> store.get("first_key")
            :::
            :::

    num\_keys[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*[)] [[→] [int(https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")]]

    :   Returns the number of keys set in the store. Note that this number will typically be one greater than the number of keys added by `set()` and `add()` since one key is used to coordinate all the workers using the store.

        :::
        Warning

        When used with the [`TCPStore`](#torch.distributed.TCPStore "torch.distributed.TCPStore"), `num_keys` returns the number of keys written to the underlying file. If the store is destructed and another store is created with the same file, the original keys will be retained.
        :::

        Returns[:]

        :   The number of keys present in the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, other store types can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set("first_key", "first_value")
                >>> # This should return 2
                >>> store.num_keys()
            :::
            :::

    queue\_len[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [int(https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")]]

    :   Returns the length of the specified queue.

        If the queue doesn't exist it returns 0.

        See queue\_push for more details.

        Parameters[:]

        :   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key of the queue to get the length.

    queue\_pop[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *key:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *block:[ ][bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")][ ]=[ ]True*[)] [[→] [bytes(https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")]]

    :   Pops a value from the specified queue or waits until timeout if the queue is empty.

        See queue\_push for more details.

        If block is False, a dist.QueueEmptyError will be raised if the queue is empty.

        Parameters[:]

        :   -   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key of the queue to pop from.

            -   **block** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) -- Whether to block waiting for the key or immediately return.

    queue\_push[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *arg1:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Pushes a value into the specified queue.

        Using the same key for queues and set/get operations may result in unexpected behavior.

        wait/check operations are supported for queues.

        wait with queues will only wake one waiting worker rather than all.

        Parameters[:]

        :   -   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key of the queue to push to.

            -   **value** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The value to push into the queue.

    set[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *arg1:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Inserts the key-value pair into the store based on the supplied `key` and `value`. If `key` already exists in the store, it will overwrite the old value with the new supplied `value`.

        Parameters[:]

        :   -   **key** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The key to be added to the store.

            -   **value** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The value associated with `key` to be added to the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set("first_key", "first_value")
                >>> # Should return "first_value"
                >>> store.get("first_key")
            :::
            :::

    set\_timeout[(]*self:[ ]torch.\_C.\_distributed\_c10d.Store*, *arg0:[ ][datetime.timedelta(https://docs.python.org/3/library/datetime.html#datetime.timedelta "(in Python v3.14)")]*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Sets the store's default timeout. This timeout is used during initialization and in `wait()` and `get()`.

        Parameters[:]

        :   **timeout** (*timedelta*) -- timeout to be set in the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, other store types can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> store.set_timeout(timedelta(seconds=10))
                >>> # This will throw an exception after 10 seconds
                >>> store.wait(["bad_key"])
            :::
            :::

    *[property][ ]*timeout

    :   Gets the timeout of the store.

    wait[(]*\*args*, *\*\*kwargs*[)]

    :   Overloaded function.

        1.  wait(self: torch.\_C.\_distributed\_c10d.Store, arg0: collections.abc.Sequence\[str\]) -\> None

        Waits for each key in `keys` to be added to the store. If not all keys are set before the `timeout` (set during store initialization), then `wait` will throw an exception.

        Parameters[:]

        :   **keys** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")) -- List of keys on which to wait until they are set in the store.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, other store types can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> # This will throw an exception after 30 seconds
                >>> store.wait(["bad_key"])
            :::
            :::

        2.  wait(self: torch.\_C.\_distributed\_c10d.Store, arg0: collections.abc.Sequence\[str\], arg1: datetime.timedelta) -\> None

        Waits for each key in `keys` to be added to the store, and throws an exception if the keys have not been set by the supplied `timeout`.

        Parameters[:]

        :   -   **keys** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")) -- List of keys on which to wait until they are set in the store.

            -   **timeout** (*timedelta*) -- Time to wait for the keys to be added before throwing an exception.

        Example::

        :   :::
            :::
                >>> import torch.distributed as dist
                >>> from datetime import timedelta
                >>> # Using TCPStore as an example, other store types can also be used
                >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
                >>> # This will throw an exception after 10 seconds
                >>> store.wait(["bad_key"], timedelta(seconds=10))
            :::
            :::

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.TCPStore

:   A TCP-based distributed key-value store implementation. The server store holds the data, while the client stores can connect to the server store over TCP and perform actions such as `set()` to insert a key-value pair, `get()` to retrieve a key-value pair, etc. There should always be one server store initialized because the client store(s) will wait for the server to establish a connection.

    Parameters[:]

    :   -   **host\_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The hostname or IP Address the server store should run on.

        -   **port** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- The port on which the server store should listen for incoming requests.

        -   **world\_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- The total number of store users (number of clients + 1 for the server). Default is None (None indicates a non-fixed number of store users).

        -   **is\_master** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- True when initializing the server store and False for client stores. Default is False.

        -   **timeout** (*timedelta,* *optional*) -- Timeout used by the store during initialization and for methods such as `get()` and `wait()`. Default is timedelta(seconds=300)

        -   **wait\_for\_workers** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- Whether to wait for all the workers to connect with the server store. This is only applicable when world\_size is a fixed value. Default is True.

        -   **multi\_tenant** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- If True, all `TCPStore` instances in the current process with the same host/port will use the same underlying `TCPServer`. Default is False.

        -   **master\_listen\_fd** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- If specified, the underlying `TCPServer` will listen on this file descriptor, which must be a socket already bound to `port`. To bind an ephemeral port we recommend setting the port to 0 and reading `.port`. Default is None (meaning the server creates a new socket and attempts to bind it to `port`).

        -   **use\_libuv** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) -- If True, use libuv for `TCPServer` backend. Default is True.

    Example::

    :   :::
        :::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Run on process 1 (server)
            >>> server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
            >>> # Run on process 2 (client)
            >>> client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
            >>> # Use any of the store methods from either the client or server after initialization
            >>> server_store.set("first_key", "first_value")
            >>> client_store.get("first_key")
        :::
        :::

    \_\_init\_\_[(]*self:[ ][torch.\_C.\_distributed\_c10d.TCPStore(#torch.distributed.TCPStore "torch._C._distributed_c10d.TCPStore")]*, *host\_name:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *port:[ ][SupportsInt(https://docs.python.org/3/library/typing.html#typing.SupportsInt "(in Python v3.14)")]*, *world\_size:[ ][SupportsInt(https://docs.python.org/3/library/typing.html#typing.SupportsInt "(in Python v3.14)")[ ]\|[ ]None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")][ ]=[ ]None*, *is\_master:[ ][bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")][ ]=[ ]False*, *timeout:[ ][datetime.timedelta(https://docs.python.org/3/library/datetime.html#datetime.timedelta "(in Python v3.14)")][ ]=[ ]datetime.timedelta(seconds=300)*, *wait\_for\_workers:[ ][bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")][ ]=[ ]True*, *multi\_tenant:[ ][bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")][ ]=[ ]False*, *master\_listen\_fd:[ ][SupportsInt(https://docs.python.org/3/library/typing.html#typing.SupportsInt "(in Python v3.14)")[ ]\|[ ]None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")][ ]=[ ]None*, *use\_libuv:[ ][bool(https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")][ ]=[ ]True*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Creates a new TCPStore.

    *[property][ ]*host

    :   Gets the hostname on which the store listens for requests.

    *[property][ ]*libuvBackend

    :   Returns True if it's using the libuv backend.

    *[property][ ]*port

    :   Gets the port number on which the store listens for requests.

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.HashStore

:   A thread-safe store implementation based on an underlying hashmap. This store can be used within the same process (for example, by other threads), but cannot be used across processes.

    Example::

    :   :::
        :::
            >>> import torch.distributed as dist
            >>> store = dist.HashStore()
            >>> # store can be used from other threads
            >>> # Use any of the store methods after initialization
            >>> store.set("first_key", "first_value")
        :::
        :::

    \_\_init\_\_[(]*self:[ ][torch.\_C.\_distributed\_c10d.HashStore(#torch.distributed.HashStore "torch._C._distributed_c10d.HashStore")]*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Creates a new HashStore.

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.FileStore

:   A store implementation that uses a file to store the underlying key-value pairs.

    Parameters[:]

    :   -   **file\_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- path of the file in which to store the key-value pairs

        -   **world\_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) -- The total number of processes using the store. Default is -1 (a negative value indicates a non-fixed number of store users).

    Example::

    :   :::
        :::
            >>> import torch.distributed as dist
            >>> store1 = dist.FileStore("/tmp/filestore", 2)
            >>> store2 = dist.FileStore("/tmp/filestore", 2)
            >>> # Use any of the store methods from either the client or server after initialization
            >>> store1.set("first_key", "first_value")
            >>> store2.get("first_key")
        :::
        :::

    \_\_init\_\_[(]*self:[ ][torch.\_C.\_distributed\_c10d.FileStore(#torch.distributed.FileStore "torch._C._distributed_c10d.FileStore")]*, *file\_name:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *world\_size:[ ][SupportsInt(https://docs.python.org/3/library/typing.html#typing.SupportsInt "(in Python v3.14)")][ ]=[ ]-1*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Creates a new FileStore.

    *[property][ ]*path

    :   Gets the path of the file used by FileStore to store key-value pairs.

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.PrefixStore

:   A wrapper around any of the 3 key-value stores ([`TCPStore`](#torch.distributed.TCPStore "torch.distributed.TCPStore"), [`FileStore`](#torch.distributed.FileStore "torch.distributed.FileStore"), and [`HashStore`](#torch.distributed.HashStore "torch.distributed.HashStore")) that adds a prefix to each key inserted to the store.

    Parameters[:]

    :   -   **prefix** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) -- The prefix string that is prepended to each key before being inserted into the store.

        -   **store** (*torch.distributed.store*) -- A store object that forms the underlying key-value store.

    \_\_init\_\_[(]*self:[ ]torch.\_C.\_distributed\_c10d.PrefixStore*, *prefix:[ ][str(https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")]*, *store:[ ]torch.\_C.\_distributed\_c10d.Store*[)] [[→] [None(https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")]]

    :   Creates a new PrefixStore.

    *[property][ ]*underlying\_store

    :   Gets the underlying store object that PrefixStore wraps around.

Profiling Collective Communication
---------------------------------------------------------------------------------------------------------------

Note that you can use `torch.profiler` (recommended, only available after 1.8.1) or `torch.autograd.profiler` to profile collective communication and point-to-point communication APIs mentioned here. All out-of-the-box backends (`gloo`, `nccl`, `mpi`) are supported and collective communication usage will be rendered as expected in profiling output/traces. Profiling your code is the same as any regular torch operator:


    import torch
    import torch.distributed as dist
    with torch.profiler():
        tensor = torch.randn(20, 10)
        dist.all_reduce(tensor)


Please refer to the [profiler documentation](https://pytorch.org/docs/main/profiler.html) for a full overview of profiler features.

Multi-GPU collective functions
-------------------------------------------------------------------------------------------------------


Warning

The multi-GPU functions (which stand for multiple GPUs per CPU thread) are deprecated. As of today, PyTorch Distributed's preferred programming model is one device per thread, as exemplified by the APIs in this document. If you are a backend developer and want to support multiple devices per thread, please contact PyTorch Distributed's maintainers.


[]{#id1}

Object collectives
-------------------------------------------------------------------------------


Warning

Object collectives have a number of serious limitations. Read further to determine if they are safe to use for your use case.

Object collectives are a set of collective-like operations that work on arbitrary Python objects, as long as they can be pickled. There are various collective patterns implemented (e.g. broadcast, all\_gather, ...) but they each roughly follow this pattern:

1.  convert the input object into a pickle (raw bytes), then shove it into a byte tensor

2.  communicate the size of this byte tensor to peers (first collective operation)

3.  allocate appropriately sized tensor to perform the real collective

4.  communicate the object data (second collective operation)

5.  convert raw data back into Python (unpickle)

Object collectives sometimes have surprising performance or memory characteristics that lead to long runtimes or OOMs, and thus they should be used with caution. Here are some common issues.

**Asymmetric pickle/unpickle time** - Pickling objects can be slow, depending on the number, type and size of the objects. When the collective has a fan-in (e.g. gather\_object), the receiving rank(s) must unpickle N times more objects than the sending rank(s) had to pickle, which can cause other ranks to time out on their next collective.

**Inefficient tensor communication** - Tensors should be sent via regular collective APIs, not object collective APIs. It is possible to send Tensors via object collective APIs, but they will be serialized and deserialized (including a CPU-sync and device-to-host copy in the case of non-CPU tensors), and in almost every case other than debugging or troubleshooting code, it would be worth the trouble to refactor the code to use non-object collectives instead.

**Unexpected tensor devices** - If you still want to send tensors via object collectives, there is another aspect specific to cuda (and possibly other accelerators) tensors. If you pickle a tensor that is currently on `cuda:3`, and then unpickle it, you will get another tensor on `cuda:3` *regardless of which process you are on, or which CUDA device is the 'default' device for that process*. With regular tensor collective APIs, 'output tensors' will always be on the same, local device, which is generally what you'd expect.

Unpickling a tensor will implicitly activate a CUDA context if it is the first time a GPU is used by the process, which can waste significant amounts of GPU memory. This issue can be avoided by moving tensors to CPU before passing them as inputs to an object collective.

Third-party backends
-----------------------------------------------------------------------------------

Besides the builtin GLOO/MPI/NCCL backends, PyTorch distributed supports third-party backends through a run-time register mechanism. For references on how to develop a third-party backend through C++ Extension, please refer to [Tutorials - Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) and `test/cpp_extensions/cpp_c10d_extension.cpp`. The capability of third-party backends are decided by their own implementations.

The new backend derives from `c10d::ProcessGroup` and registers the backend name and the instantiating interface through [`torch.distributed.Backend.register_backend()`](#torch.distributed.Backend.register_backend "torch.distributed.Backend.register_backend") when imported.

When manually importing this backend and invoking [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") with the corresponding backend name, the `torch.distributed` package runs on the new backend.


Warning

The support of third-party backend is experimental and subject to change.


[]{#distributed-launch}

Launch utility
-----------------------------------------------------------------------

The `torch.distributed` package also provides a launch utility in `torch.distributed.launch`. This helper utility can be used to launch multiple processes per node for distributed training.

Module `torch.distributed.launch`.

`torch.distributed.launch` is a module that spawns up multiple distributed training processes on each of the training nodes.


Warning

This module is going to be deprecated in favor of torchrun(elastic/run.html#launcher-api).

The utility can be used for single-node distributed training, in which one or more processes per node will be spawned. The utility can be used for either CPU training or GPU training. If the utility is used for GPU training, each distributed process will be operating on a single GPU. This can achieve well-improved single-node training performance. It can also be used in multi-node distributed training, by spawning up multiple processes on each node for well-improved multi-node distributed training performance as well. This will especially be beneficial for systems with multiple Infiniband interfaces that have direct-GPU support, since all of them can be utilized for aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed training, this utility will launch the given number of processes per node (`--nproc-per-node`). If used for GPU training, this number needs to be less or equal to the number of GPUs on the current system (`nproc_per_node`), and each process will be operating on a single GPU from *GPU 0 to GPU (nproc\_per\_node - 1)*.

**How to use this module:**

1.  Single-Node multi-process distributed training


    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)


2.  Multi-Node multi-process distributed training: (e.g. two nodes)

Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*


    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)


Node 2:


    python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node-rank=1 --master-addr="192.168.1.1"
               --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)


3.  To look up what optional arguments this module offers:


    python -m torch.distributed.launch --help


**Important Notices:**

1\. This utility and multi-process distributed (single-node or multi-node) GPU training currently only achieves the best performance using the NCCL distributed backend. Thus NCCL backend is the recommended backend to use for GPU training.

2\. In your training program, you must parse the command-line argument: `--local-rank=LOCAL_PROCESS_RANK`, which will be provided by this module. If your training program uses GPUs, you should ensure that your code only runs on the GPU device of LOCAL\_PROCESS\_RANK. This can be done by:

Parsing the local\_rank argument


    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local-rank", "--local_rank", type=int)
    >>> args = parser.parse_args()


Set your device to local rank using either


    >>> torch.cuda.set_device(args.local_rank)  # before your code runs


or


    >>> with torch.cuda.device(args.local_rank):
    >>>    # your code to run
    >>>    ...


[Changed in version 2.0.0: ]The launcher will passes the `--local-rank=<rank>` argument to your script. From PyTorch 2.0.0 onwards, the dashed `--local-rank` is preferred over the previously used underscored `--local_rank`.

For backward compatibility, it may be necessary for users to handle both cases in their argument parsing code. This means including both `"--local-rank"` and `"--local_rank"` in the argument parser. If only `"--local_rank"` is provided, the launcher will trigger an error: "error: unrecognized arguments: --local-rank=\<rank\>". For training code that only supports PyTorch 2.0.0+, including `"--local-rank"` should be sufficient.

3\. In your training program, you are supposed to call the following function at the beginning to start the distributed backend. It is strongly recommended that `init_method=env://`. Other init methods (e.g. `tcp://`) may work, but `env://` is the one that is officially supported by this module.


    >>> torch.distributed.init_process_group(backend='YOUR BACKEND',
    >>>                                      init_method='env://')


4\. In your training program, you can either use regular distributed functions or use [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") module. If your training program uses GPUs for training and you would like to use [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") module, here is how to configure it.


    >>> model = torch.nn.parallel.DistributedDataParallel(model,
    >>>                                                   device_ids=[args.local_rank],
    >>>                                                   output_device=args.local_rank)


Please ensure that `device_ids` argument is set to be the only GPU device id that your code will be operating on. This is generally the local rank of the process. In other words, the `device_ids` needs to be `[args.local_rank]`, and `output_device` needs to be `args.local_rank` in order to use this utility

5\. Another way to pass `local_rank` to the subprocesses via environment variable `LOCAL_RANK`. This behavior is enabled when you launch the script with `--use-env=True`. You must adjust the subprocess example above to replace `args.local_rank` with `os.environ['LOCAL_RANK']`; the launcher will not pass `--local-rank` when you specify this flag.


Warning

`local_rank` is NOT globally unique: it is only unique per process on a machine. Thus, don't use it to decide if you should, e.g., write to a networked filesystem. See [pytorch/pytorch\#12042](https://github.com/pytorch/pytorch/issues/12042) for an example of how things can go wrong if you don't do this correctly.


Spawn utility
---------------------------------------------------------------------

The Multiprocessing package - torch.multiprocessing(multiprocessing.html#multiprocessing-doc) package also provides a `spawn` function in [`torch.multiprocessing.spawn()`](multiprocessing.html#module-torch.multiprocessing.spawn "torch.multiprocessing.spawn"). This helper function can be used to spawn multiple processes. It works by passing in the function that you want to run and spawns N processes to run it. This can be used for multiprocess distributed training as well.

For references on how to use it, please refer to [PyTorch example - ImageNet implementation](https://github.com/pytorch/examples/tree/master/imagenet)

Note that this function requires Python 3.4 or higher.

Debugging `torch.distributed` applications
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Debugging distributed applications can be challenging due to hard to understand hangs, crashes, or inconsistent behavior across ranks. `torch.distributed` provides a suite of tools to help debug training applications in a self-serve fashion:


### Python Breakpoint

It is extremely convenient to use python's debugger in a distributed environment, but because it does not work out of the box many people do not use it at all. PyTorch offers a customized wrapper around pdb that streamlines the process.

`torch.distributed.breakpoint` makes this process easy. Internally, it customizes `pdb`'s breakpoint behavior in two ways but otherwise behaves as normal `pdb`.

1.  Attaches the debugger only on one rank (specified by the user).

2.  Ensures all other ranks stop, by using a `torch.distributed.barrier()` that will release once the debugged rank issues a `continue`

3.  Reroutes stdin from the child process such that it connects to your terminal.

To use it, simply issue `torch.distributed.breakpoint(rank)` on all ranks, using the same value for `rank` in each case.

### Monitored Barrier

As of v1.10, [`torch.distributed.monitored_barrier()`](#torch.distributed.monitored_barrier "torch.distributed.monitored_barrier") exists as an alternative to [`torch.distributed.barrier()`](#torch.distributed.barrier "torch.distributed.barrier") which fails with helpful information about which rank may be faulty when crashing, i.e. not all ranks calling into [`torch.distributed.monitored_barrier()`](#torch.distributed.monitored_barrier "torch.distributed.monitored_barrier") within the provided timeout. [`torch.distributed.monitored_barrier()`](#torch.distributed.monitored_barrier "torch.distributed.monitored_barrier") implements a host-side barrier using `send`/`recv` communication primitives in a process similar to acknowledgements, allowing rank 0 to report which rank(s) failed to acknowledge the barrier in time. As an example, consider the following function where rank 1 fails to call into [`torch.distributed.monitored_barrier()`](#torch.distributed.monitored_barrier "torch.distributed.monitored_barrier") (in practice this could be due to an application bug or hang in a previous collective):


    import os
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def worker(rank):
        dist.init_process_group("nccl", rank=rank, world_size=2)
        # monitored barrier requires gloo process group to perform host-side sync.
        group_gloo = dist.new_group(backend="gloo")
        if rank not in [1]:
            dist.monitored_barrier(group=group_gloo, timeout=timedelta(seconds=2))


    if __name__ == "__main__":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        mp.spawn(worker, nprocs=2, args=())


The following error message is produced on rank 0, allowing the user to determine which rank(s) may be faulty and investigate further:


    RuntimeError: Rank 1 failed to pass monitoredBarrier in 2000 ms
     Original exception:
    [gloo/transport/tcp/pair.cc:598] Connection closed by peer [2401:db00:eef0:1100:3560:0:1c05:25d]:8594


### `TORCH_DISTRIBUTED_DEBUG`

With `TORCH_CPP_LOG_LEVEL=INFO`, the environment variable `TORCH_DISTRIBUTED_DEBUG` can be used to trigger additional useful logging and collective synchronization checks to ensure all ranks are synchronized appropriately. `TORCH_DISTRIBUTED_DEBUG` can be set to either `OFF` (default), `INFO`, or `DETAIL` depending on the debugging level required. Please note that the most verbose option, `DETAIL` may impact the application performance and thus should only be used when debugging issues.

Setting `TORCH_DISTRIBUTED_DEBUG=INFO` will result in additional debug logging when models trained with [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") are initialized, and `TORCH_DISTRIBUTED_DEBUG=DETAIL` will additionally log runtime performance statistics a select number of iterations. These runtime statistics include data such as forward time, backward time, gradient communication time, etc. As an example, given the following application:


    import os

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    class TwoLinLayerNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(10, 10, bias=False)
            self.b = torch.nn.Linear(10, 1, bias=False)

        def forward(self, x):
            a = self.a(x)
            b = self.b(x)
            return (a, b)


    def worker(rank):
        dist.init_process_group("nccl", rank=rank, world_size=2)
        torch.cuda.set_device(rank)
        print("init model")
        model = TwoLinLayerNet().cuda()
        print("init ddp")
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        inp = torch.randn(10, 10).cuda()
        print("train")

        for _ in range(20):
            output = ddp_model(inp)
            loss = output[0] + output[1]
            loss.sum().backward()


    if __name__ == "__main__":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ[
            "TORCH_DISTRIBUTED_DEBUG"
        ] = "DETAIL"  # set to DETAIL for runtime logging.
        mp.spawn(worker, nprocs=2, args=())


The following logs are rendered at initialization time:


    I0607 16:10:35.739390 515217 logger.cpp:173] [Rank 0]: DDP Initialized with:
    broadcast_buffers: 1
    bucket_cap_bytes: 26214400
    find_unused_parameters: 0
    gradient_as_bucket_view: 0
    is_multi_device_module: 0
    iteration: 0
    num_parameter_tensors: 2
    output_device: 0
    rank: 0
    total_parameter_size_bytes: 440
    world_size: 2
    backend_name: nccl
    bucket_sizes: 440
    cuda_visible_devices: N/A
    device_ids: 0
    dtypes: float
    master_addr: localhost
    master_port: 29501
    module_name: TwoLinLayerNet
    nccl_async_error_handling: N/A
    nccl_blocking_wait: N/A
    nccl_debug: WARN
    nccl_ib_timeout: N/A
    nccl_nthreads: N/A
    nccl_socket_ifname: N/A
    torch_distributed_debug: INFO


The following logs are rendered during runtime (when `TORCH_DISTRIBUTED_DEBUG=DETAIL` is set):


    I0607 16:18:58.085681 544067 logger.cpp:344] [Rank 1 / 2] Training TwoLinLayerNet unused_parameter_size=0
     Avg forward compute time: 40838608
     Avg backward compute time: 5983335
    Avg backward comm. time: 4326421
     Avg backward comm/comp overlap time: 4207652
    I0607 16:18:58.085693 544066 logger.cpp:344] [Rank 0 / 2] Training TwoLinLayerNet unused_parameter_size=0
     Avg forward compute time: 42850427
     Avg backward compute time: 3885553
    Avg backward comm. time: 2357981
     Avg backward comm/comp overlap time: 2234674


In addition, `TORCH_DISTRIBUTED_DEBUG=INFO` enhances crash logging in [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") due to unused parameters in the model. Currently, `find_unused_parameters=True` must be passed into [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") initialization if there are parameters that may be unused in the forward pass, and as of v1.10, all model outputs are required to be used in loss computation as [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") does not support unused parameters in the backwards pass. These constraints are challenging especially for larger models, thus when crashing with an error, [`torch.nn.parallel.DistributedDataParallel()`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") will log the fully qualified name of all parameters that went unused. For example, in the above application, if we modify `loss` to be instead computed as `loss = output[1]`, then `TwoLinLayerNet.a` does not receive a gradient in the backwards pass, and thus results in `DDP` failing. On a crash, the user is passed information about parameters which went unused, which may be challenging to manually find for large models:


    RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing
     the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
    making sure all `forward` function outputs participate in calculating loss.
    If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return va
    lue of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
    Parameters which did not receive grad for rank 0: a.weight
    Parameter indices which did not receive grad for rank 0: 0


Setting `TORCH_DISTRIBUTED_DEBUG=DETAIL` will trigger additional consistency and synchronization checks on every collective call issued by the user either directly or indirectly (such as DDP `allreduce`). This is done by creating a wrapper process group that wraps all process groups returned by [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") and [`torch.distributed.new_group()`](#torch.distributed.new_group "torch.distributed.new_group") APIs. As a result, these APIs will return a wrapper process group that can be used exactly like a regular process group, but performs consistency checks before dispatching the collective to an underlying process group. Currently, these checks include a [`torch.distributed.monitored_barrier()`](#torch.distributed.monitored_barrier "torch.distributed.monitored_barrier"), which ensures all ranks complete their outstanding collective calls and reports ranks which are stuck. Next, the collective itself is checked for consistency by ensuring all collective functions match and are called with consistent tensor shapes. If this is not the case, a detailed error report is included when the application crashes, rather than a hang or uninformative error message. As an example, consider the following function which has mismatched input shapes into [`torch.distributed.all_reduce()`](#torch.distributed.all_reduce "torch.distributed.all_reduce"):


    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def worker(rank):
        dist.init_process_group("nccl", rank=rank, world_size=2)
        torch.cuda.set_device(rank)
        tensor = torch.randn(10 if rank == 0 else 20).cuda()
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device=rank)


    if __name__ == "__main__":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        mp.spawn(worker, nprocs=2, args=())


With the `NCCL` backend, such an application would likely result in a hang which can be challenging to root-cause in nontrivial scenarios. If the user enables `TORCH_DISTRIBUTED_DEBUG=DETAIL` and reruns the application, the following error message reveals the root cause:


    work = default_pg.allreduce([tensor], opts)
    RuntimeError: Error when verifying shape tensors for collective ALLREDUCE on rank 0. This likely indicates that input shapes into the collective are mismatched across ranks. Got shapes:  10
    [ torch.LongTensor{1} ]


Note

For fine-grained control of the debug level during runtime the functions `torch.distributed.set_debug_level()`, `torch.distributed.set_debug_level_from_env()`, and `torch.distributed.get_debug_level()` can also be used.

In addition, `TORCH_DISTRIBUTED_DEBUG=DETAIL` can be used in conjunction with `TORCH_SHOW_CPP_STACKTRACES=1` to log the entire callstack when a collective desynchronization is detected. These collective desynchronization checks will work for all applications that use `c10d` collective calls backed by process groups created with the [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") and [`torch.distributed.new_group()`](#torch.distributed.new_group "torch.distributed.new_group") APIs.

### torch.distributed.debug HTTP Server

The `torch.distributed.debug` module provides a HTTP server that can be used to debug distributed applications. The server can be started by calling [`torch.distributed.debug.start_debug_server()`](#torch.distributed.debug.start_debug_server "torch.distributed.debug.start_debug_server"). This allows users to collect data across all workers at runtime.

torch.distributed.debug.start\_debug\_server[(]*port=25999*, *worker\_port=0*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/debug/__init__.py#L22)

:   Start the debug server stack on all workers. The frontend debug server is only started on rank0 while the per rank worker servers are started on all ranks.

    This server provides an HTTP frontend that allows for debugging slow and deadlocked distributed jobs across all ranks simultaneously. This collects data such as stack traces, FlightRecorder events, and performance profiles.

    This depends on dependencies which are not installed by default.

    Dependencies: - Jinja2 - aiohttp

    WARNING: This is intended to only be used in trusted network environments. The debug server is not designed to be secure and should not be exposed to the public internet. See SECURITY.md for more details.

    WARNING: This is an experimental feature and may change at any time.

    Parameters[:]

    :   -   **port** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- The port to start the frontend debug server on.

        -   **worker\_port** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- The port to start the worker server on. Defaults to 0, which will cause the worker server to bind to an ephemeral port.

```{=html}
<!-- -->
```

torch.distributed.debug.stop\_debug\_server[(][)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/debug/__init__.py#L72)

:   Shutdown the debug server and stop the frontend debug server process.


Logging
---------------------------------------------------------

In addition to explicit debugging support via [`torch.distributed.monitored_barrier()`](#torch.distributed.monitored_barrier "torch.distributed.monitored_barrier") and `TORCH_DISTRIBUTED_DEBUG`, the underlying C++ library of `torch.distributed` also outputs log messages at various levels. These messages can be helpful to understand the execution state of a distributed training job and to troubleshoot problems such as network connection failures. The following matrix shows how the log level can be adjusted via the combination of `TORCH_CPP_LOG_LEVEL` and `TORCH_DISTRIBUTED_DEBUG` environment variables.


  `TORCH_CPP_LOG_LEVEL`   `TORCH_DISTRIBUTED_DEBUG`   Effective Log Level
  -------------------------------------------------------- ------------------------------------------------------------ ---------------------
  `ERROR`                 ignored                                                      Error
  `WARNING`               ignored                                                      Warning
  `INFO`                  ignored                                                      Info
  `INFO`                  `INFO`                      Debug
  `INFO`                  `DETAIL`                    Trace (a.k.a. All)

Distributed components raise custom Exception types derived from `RuntimeError`:

-   `torch.distributed.DistError`: This is the base type of all distributed exceptions.

-   `torch.distributed.DistBackendError`: This exception is thrown when a backend-specific error occurs. For example, if the `NCCL` backend is used and the user attempts to use a GPU that is not available to the `NCCL` library.

-   `torch.distributed.DistNetworkError`: This exception is thrown when networking libraries encounter errors (ex: Connection reset by peer)

-   `torch.distributed.DistStoreError`: This exception is thrown when the Store encounters an error (ex: TCPStore timeout)

*[class][ ]*torch.distributed.DistError

:   Exception raised when an error occurs in the distributed library

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.DistBackendError

:   Exception raised when a backend error occurs in distributed

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.DistNetworkError

:   Exception raised when a network error occurs in distributed

```{=html}
<!-- -->
```

*[class][ ]*torch.distributed.DistStoreError

:   Exception raised when an error occurs in the distributed store

If you are running single node training, it may be convenient to interactively breakpoint your script. We offer a way to conveniently breakpoint a single rank:

torch.distributed.breakpoint[(]*rank=0*, *skip=0*, *timeout\_s=3600*[)][\[source\]](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/__init__.py#L86)

:   Set a breakpoint, but only on a single rank. All other ranks will wait for you to be done with the breakpoint before continuing.

    Parameters[:]

    :   -   **rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Which rank to break on. Default: `0`

        -   **skip** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) -- Skip the first `skip` calls to this breakpoint. Default: `0`.


[]{#module-torch.distributed.utils}[]{#module-torch.distributed.tensor.parallel.style}[]{#module-torch.distributed.tensor.parallel.loss}[]{#module-torch.distributed.tensor.parallel.input_reshard}[]{#module-torch.distributed.tensor.parallel.fsdp}[]{#module-torch.distributed.tensor.parallel.ddp}[]{#module-torch.distributed.tensor.parallel.api}[]{#module-torch.distributed.rpc.server_process_global_profiler}[]{#module-torch.distributed.rpc.rref_proxy}[]{#module-torch.distributed.rpc.options}[]{#module-torch.distributed.rpc.internal}[]{#module-torch.distributed.rpc.functions}[]{#module-torch.distributed.rpc.constants}[]{#module-torch.distributed.rpc.backend_registry}[]{#module-torch.distributed.rpc.api}[]{#module-torch.distributed.rendezvous}[]{#module-torch.distributed.remote_device}[]{#module-torch.distributed.optim.zero_redundancy_optimizer}[]{#module-torch.distributed.optim.utils}[]{#module-torch.distributed.optim.post_localSGD_optimizer}[]{#module-torch.distributed.optim.optimizer}[]{#module-torch.distributed.optim.named_optimizer}[]{#module-torch.distributed.optim.functional_sgd}[]{#module-torch.distributed.optim.functional_rprop}[]{#module-torch.distributed.optim.functional_rmsprop}[]{#module-torch.distributed.optim.functional_adamw}[]{#module-torch.distributed.optim.functional_adamax}[]{#module-torch.distributed.optim.functional_adam}[]{#module-torch.distributed.optim.functional_adagrad}[]{#module-torch.distributed.optim.functional_adadelta}[]{#module-torch.distributed.optim.apply_optimizer_in_backward}[]{#module-torch.distributed.nn.jit.templates.remote_module_template}[]{#module-torch.distributed.nn.jit.instantiator}[]{#module-torch.distributed.nn.functional}[]{#module-torch.distributed.nn.api.remote_module}[]{#module-torch.distributed.logging_handlers}[]{#module-torch.distributed.launcher.api}[]{#module-torch.distributed.fsdp.wrap}[]{#module-torch.distributed.fsdp.sharded_grad_scaler}[]{#module-torch.distributed.fsdp.fully_sharded_data_parallel}[]{#module-torch.distributed.fsdp.api}[]{#module-torch.distributed.elastic.utils.store}[]{#module-torch.distributed.elastic.utils.logging}[]{#module-torch.distributed.elastic.utils.log_level}[]{#module-torch.distributed.elastic.utils.distributed}[]{#module-torch.distributed.elastic.utils.data.elastic_distributed_sampler}[]{#module-torch.distributed.elastic.utils.data.cycling_iterator}[]{#module-torch.distributed.elastic.utils.api}[]{#module-torch.distributed.elastic.timer.local_timer}[]{#module-torch.distributed.elastic.timer.file_based_local_timer}[]{#module-torch.distributed.elastic.timer.api}[]{#module-torch.distributed.elastic.rendezvous.utils}[]{#module-torch.distributed.elastic.rendezvous.static_tcp_rendezvous}[]{#module-torch.distributed.elastic.rendezvous.etcd_store}[]{#module-torch.distributed.elastic.rendezvous.etcd_server}[]{#module-torch.distributed.elastic.rendezvous.etcd_rendezvous_backend}[]{#module-torch.distributed.elastic.rendezvous.etcd_rendezvous}[]{#module-torch.distributed.elastic.rendezvous.dynamic_rendezvous}[]{#module-torch.distributed.elastic.rendezvous.c10d_rendezvous_backend}[]{#module-torch.distributed.elastic.rendezvous.api}[]{#module-torch.distributed.elastic.multiprocessing.tail_log}[]{#module-torch.distributed.elastic.multiprocessing.redirects}[]{#module-torch.distributed.elastic.multiprocessing.errors.handlers}[]{#module-torch.distributed.elastic.multiprocessing.errors.error_handler}[]{#module-torch.distributed.elastic.multiprocessing.api}[]{#module-torch.distributed.elastic.metrics.api}[]{#module-torch.distributed.elastic.events.handlers}[]{#module-torch.distributed.elastic.events.api}[]{#module-torch.distributed.elastic.agent.server.local_elastic_agent}[]{#module-torch.distributed.elastic.agent.server.api}[]{#module-torch.distributed.distributed_c10d}[]{#module-torch.distributed.device_mesh}[]{#module-torch.distributed.constants}[]{#module-torch.distributed.collective_utils}[]{#module-torch.distributed.checkpoint.utils}[]{#module-torch.distributed.checkpoint.storage}[]{#module-torch.distributed.checkpoint.stateful}[]{#module-torch.distributed.checkpoint.state_dict_saver}[]{#module-torch.distributed.checkpoint.state_dict_loader}[]{#module-torch.distributed.checkpoint.resharding}[]{#module-torch.distributed.checkpoint.planner_helpers}[]{#module-torch.distributed.checkpoint.planner}[]{#module-torch.distributed.checkpoint.optimizer}[]{#module-torch.distributed.checkpoint.metadata}[]{#module-torch.distributed.checkpoint.quantized_hf_storage}[]{#module-torch.distributed.checkpoint.hf_storage}[]{#module-torch.distributed.checkpoint.filesystem}[]{#module-torch.distributed.checkpoint.default_planner}[]{#module-torch.distributed.checkpoint.api}[]{#module-torch.distributed.c10d_logger}[]{#module-torch.distributed.argparse_util}[]{#module-torch.distributed.algorithms.model_averaging.utils}[]{#module-torch.distributed.algorithms.model_averaging.hierarchical_model_averager}[]{#module-torch.distributed.algorithms.model_averaging.averagers}[]{#module-torch.distributed.algorithms.join}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.mixed_precision_hooks}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.default_hooks}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks}[]{#module-torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook}[]{#module-torch.distributed.nn.jit.templates}[]{#module-torch.distributed.nn.jit}[]{#module-torch.distributed.nn.api}[]{#module-torch.distributed.nn}[]{#module-torch.distributed.launcher}[]{#module-torch.distributed.elastic.utils.data}[]{#module-torch.distributed.elastic.utils}[]{#module-torch.distributed.elastic}[]{#module-torch.distributed.algorithms.model_averaging}[]{#module-torch.distributed.algorithms.ddp_comm_hooks}[]{#module-torch.distributed.algorithms}


Rate this Page


[★]{.star behavior="tutorial-rating" count="1" data-value="1"} [★]{.star behavior="tutorial-rating" count="2" data-value="2"} [★]{.star behavior="tutorial-rating" count="3" data-value="3"} [★]{.star behavior="tutorial-rating" count="4" data-value="4"} [★]{.star behavior="tutorial-rating" count="5" data-value="5"}


Send Feedback


previous

torch.backends


next

Experimental Object Oriented Distributed API


Built with the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) 0.15.4.


previous

torch.backends


next

Experimental Object Oriented Distributed API


On this page

-   [Backends](#backends)
    -   [Backends that come with PyTorch](#backends-that-come-with-pytorch)
    -   [Which backend to use?](#which-backend-to-use)
    -   [Common environment variables](#common-environment-variables)
        -   [Choosing the network interface to use](#choosing-the-network-interface-to-use)
        -   [Other NCCL environment variables](#other-nccl-environment-variables)
    -   [Copy Engine Collectives](#copy-engine-collectives)
-   [Basics](#basics)
-   [Initialization](#initialization)
    -   [`is_available()`](#torch.distributed.is_available)
    -   [`init_process_group()`](#torch.distributed.init_process_group)
    -   [`init_device_mesh()`](#torch.distributed.device_mesh.init_device_mesh)
    -   [`is_initialized()`](#torch.distributed.is_initialized)
    -   [`is_mpi_available()`](#torch.distributed.is_mpi_available)
    -   [`is_nccl_available()`](#torch.distributed.is_nccl_available)
    -   [`is_gloo_available()`](#torch.distributed.is_gloo_available)
    -   [`is_xccl_available()`](#torch.distributed.distributed_c10d.is_xccl_available)
    -   [`batch_isend_irecv()`](#torch.distributed.distributed_c10d.batch_isend_irecv)
    -   [`destroy_process_group()`](#torch.distributed.distributed_c10d.destroy_process_group)
    -   [`is_backend_available()`](#torch.distributed.distributed_c10d.is_backend_available)
    -   [`irecv()`](#torch.distributed.distributed_c10d.irecv)
    -   [`is_gloo_available()`](#torch.distributed.distributed_c10d.is_gloo_available)
    -   [`is_initialized()`](#torch.distributed.distributed_c10d.is_initialized)
    -   [`is_mpi_available()`](#torch.distributed.distributed_c10d.is_mpi_available)
    -   [`is_nccl_available()`](#torch.distributed.distributed_c10d.is_nccl_available)
    -   [`is_torchelastic_launched()`](#torch.distributed.distributed_c10d.is_torchelastic_launched)
    -   [`is_ucc_available()`](#torch.distributed.distributed_c10d.is_ucc_available)
    -   [`is_torchelastic_launched()`](#torch.distributed.is_torchelastic_launched)
    -   [`get_default_backend_for_device()`](#torch.distributed.get_default_backend_for_device)
    -   [TCP initialization](#tcp-initialization)
    -   [Shared file-system initialization](#shared-file-system-initialization)
    -   [Environment variable initialization](#environment-variable-initialization)
    -   [Improving initialization time](#improving-initialization-time)
-   [Post-Initialization](#post-initialization)
    -   [`Backend`](#torch.distributed.Backend)
        -   [`Backend.register_backend()`](#torch.distributed.Backend.register_backend)
    -   [`get_backend()`](#torch.distributed.get_backend)
    -   [`get_rank()`](#torch.distributed.get_rank)
    -   [`get_world_size()`](#torch.distributed.get_world_size)
-   [Shutdown](#shutdown)
    -   [Reinitialization](#reinitialization)
-   [Groups](#groups)
    -   [`new_group()`](#torch.distributed.new_group)
    -   [`shrink_group()`](#torch.distributed.distributed_c10d.shrink_group)
    -   [`get_group_rank()`](#torch.distributed.get_group_rank)
    -   [`get_global_rank()`](#torch.distributed.get_global_rank)
    -   [`get_process_group_ranks()`](#torch.distributed.get_process_group_ranks)
-   [DeviceMesh](#devicemesh)
    -   [`DeviceMesh`](#torch.distributed.device_mesh.DeviceMesh)
        -   [`DeviceMesh.device_type`](#torch.distributed.device_mesh.DeviceMesh.device_type)
        -   [`DeviceMesh.from_group()`](#torch.distributed.device_mesh.DeviceMesh.from_group)
        -   [`DeviceMesh.get_all_groups()`](#torch.distributed.device_mesh.DeviceMesh.get_all_groups)
        -   [`DeviceMesh.get_coordinate()`](#torch.distributed.device_mesh.DeviceMesh.get_coordinate)
        -   [`DeviceMesh.get_group()`](#torch.distributed.device_mesh.DeviceMesh.get_group)
        -   [`DeviceMesh.get_local_rank()`](#torch.distributed.device_mesh.DeviceMesh.get_local_rank)
        -   [`DeviceMesh.get_rank()`](#torch.distributed.device_mesh.DeviceMesh.get_rank)
        -   [`DeviceMesh.mesh`](#torch.distributed.device_mesh.DeviceMesh.mesh)
        -   [`DeviceMesh.mesh_dim_names`](#torch.distributed.device_mesh.DeviceMesh.mesh_dim_names)
-   [Point-to-point communication](#point-to-point-communication)
    -   [`send()`](#torch.distributed.send)
    -   [`recv()`](#torch.distributed.recv)
    -   [`isend()`](#torch.distributed.isend)
    -   [`irecv()`](#torch.distributed.irecv)
    -   [`send_object_list()`](#torch.distributed.send_object_list)
    -   [`recv_object_list()`](#torch.distributed.recv_object_list)
    -   [`batch_isend_irecv()`](#torch.distributed.batch_isend_irecv)
    -   [`P2POp`](#torch.distributed.P2POp)
-   [Synchronous and asynchronous collective operations](#synchronous-and-asynchronous-collective-operations)
-   [Collective functions](#collective-functions)
    -   [`broadcast()`](#torch.distributed.broadcast)
    -   [`broadcast_object_list()`](#torch.distributed.broadcast_object_list)
    -   [`all_reduce()`](#torch.distributed.all_reduce)
    -   [`reduce()`](#torch.distributed.reduce)
    -   [`all_gather()`](#torch.distributed.all_gather)
    -   [`all_gather_into_tensor()`](#torch.distributed.all_gather_into_tensor)
    -   [`all_gather_object()`](#torch.distributed.all_gather_object)
    -   [`gather()`](#torch.distributed.gather)
    -   [`gather_object()`](#torch.distributed.gather_object)
    -   [`scatter()`](#torch.distributed.scatter)
    -   [`scatter_object_list()`](#torch.distributed.scatter_object_list)
    -   [`reduce_scatter()`](#torch.distributed.reduce_scatter)
    -   [`reduce_scatter_tensor()`](#torch.distributed.reduce_scatter_tensor)
    -   [`all_to_all_single()`](#torch.distributed.all_to_all_single)
    -   [`all_to_all()`](#torch.distributed.all_to_all)
    -   [`barrier()`](#torch.distributed.barrier)
    -   [`monitored_barrier()`](#torch.distributed.monitored_barrier)
    -   [`Work`](#torch.distributed.Work)
        -   [`Work.block_current_stream()`](#torch.distributed.Work.block_current_stream)
        -   [`Work.boxed()`](#torch.distributed.Work.boxed)
        -   [`Work.exception()`](#torch.distributed.Work.exception)
        -   [`Work.get_future()`](#torch.distributed.Work.get_future)
        -   [`Work.get_future_result()`](#torch.distributed.Work.get_future_result)
        -   [`Work.is_completed()`](#torch.distributed.Work.is_completed)
        -   [`Work.is_success()`](#torch.distributed.Work.is_success)
        -   [`Work.result()`](#torch.distributed.Work.result)
        -   [`Work.source_rank()`](#torch.distributed.Work.source_rank)
        -   [`Work.synchronize()`](#torch.distributed.Work.synchronize)
        -   [`Work.unbox()`](#torch.distributed.Work.unbox)
        -   [`Work.wait()`](#torch.distributed.Work.wait)
    -   [`ReduceOp`](#torch.distributed.ReduceOp)
    -   [`reduce_op`](#torch.distributed.reduce_op)
-   [Distributed Key-Value Store](#distributed-key-value-store)
    -   [`Store`](#torch.distributed.Store)
        -   [`Store.__init__()`](#torch.distributed.Store.__init__)
        -   [`Store.add()`](#torch.distributed.Store.add)
        -   [`Store.append()`](#torch.distributed.Store.append)
        -   [`Store.check()`](#torch.distributed.Store.check)
        -   [`Store.clone()`](#torch.distributed.Store.clone)
        -   [`Store.compare_set()`](#torch.distributed.Store.compare_set)
        -   [`Store.delete_key()`](#torch.distributed.Store.delete_key)
        -   [`Store.get()`](#torch.distributed.Store.get)
        -   [`Store.has_extended_api()`](#torch.distributed.Store.has_extended_api)
        -   [`Store.list_keys()`](#torch.distributed.Store.list_keys)
        -   [`Store.multi_get()`](#torch.distributed.Store.multi_get)
        -   [`Store.multi_set()`](#torch.distributed.Store.multi_set)
        -   [`Store.num_keys()`](#torch.distributed.Store.num_keys)
        -   [`Store.queue_len()`](#torch.distributed.Store.queue_len)
        -   [`Store.queue_pop()`](#torch.distributed.Store.queue_pop)
        -   [`Store.queue_push()`](#torch.distributed.Store.queue_push)
        -   [`Store.set()`](#torch.distributed.Store.set)
        -   [`Store.set_timeout()`](#torch.distributed.Store.set_timeout)
        -   [`Store.timeout`](#torch.distributed.Store.timeout)
        -   [`Store.wait()`](#torch.distributed.Store.wait)
    -   [`TCPStore`](#torch.distributed.TCPStore)
        -   [`TCPStore.__init__()`](#torch.distributed.TCPStore.__init__)
        -   [`TCPStore.host`](#torch.distributed.TCPStore.host)
        -   [`TCPStore.libuvBackend`](#torch.distributed.TCPStore.libuvBackend)
        -   [`TCPStore.port`](#torch.distributed.TCPStore.port)
    -   [`HashStore`](#torch.distributed.HashStore)
        -   [`HashStore.__init__()`](#torch.distributed.HashStore.__init__)
    -   [`FileStore`](#torch.distributed.FileStore)
        -   [`FileStore.__init__()`](#torch.distributed.FileStore.__init__)
        -   [`FileStore.path`](#torch.distributed.FileStore.path)
    -   [`PrefixStore`](#torch.distributed.PrefixStore)
        -   [`PrefixStore.__init__()`](#torch.distributed.PrefixStore.__init__)
        -   [`PrefixStore.underlying_store`](#torch.distributed.PrefixStore.underlying_store)
-   [Profiling Collective Communication](#profiling-collective-communication)
-   [Multi-GPU collective functions](#multi-gpu-collective-functions)
-   [Object collectives](#object-collectives)
-   [Third-party backends](#third-party-backends)
-   [Launch utility](#launch-utility)
-   [Spawn utility](#spawn-utility)
-   [Debugging `torch.distributed` applications](#debugging-torch-distributed-applications)
    -   [Python Breakpoint](#python-breakpoint)
    -   [Monitored Barrier](#monitored-barrier)
    -   [`TORCH_DISTRIBUTED_DEBUG`](#torch-distributed-debug)
    -   [torch.distributed.debug HTTP Server](#torch-distributed-debug-http-server)
        -   [`start_debug_server()`](#torch.distributed.debug.start_debug_server)
        -   [`stop_debug_server()`](#torch.distributed.debug.stop_debug_server)
-   [Logging](#logging)
    -   [`DistError`](#torch.distributed.DistError)
    -   [`DistBackendError`](#torch.distributed.DistBackendError)
    -   [`DistNetworkError`](#torch.distributed.DistNetworkError)
    -   [`DistStoreError`](#torch.distributed.DistStoreError)
    -   [`breakpoint()`](#torch.distributed.breakpoint)

[ Edit on GitHub](https://github.com/pytorch/pytorch/edit/main/docs/source/distributed.md)


[ Show Source](_sources/distributed.md.txt)


PyTorch Libraries

-   [torchao](https://docs.pytorch.org/ao)
-   [torchrec](https://docs.pytorch.org/torchrec)
-   [torchft](https://docs.pytorch.org/torchft)
-   [TorchCodec](https://docs.pytorch.org/torchcodec)
-   [torchvision](https://docs.pytorch.org/vision)
-   [ExecuTorch](https://docs.pytorch.org/executorch)
-   [PyTorch on XLA Devices](https://docs.pytorch.org/xla)


Docs
----

Access comprehensive developer documentation for PyTorch

[View Docs](https://docs.pytorch.org/docs/stable/index.html)

Tutorials
---------

Get in-depth tutorials for beginners and advanced developers

[View Tutorials](https://docs.pytorch.org/tutorials)

Resources
---------

Find development resources and get your questions answered

[View Resources](https://pytorch.org/resources)


**Stay in touch** for updates, event info, and the latest news

By submitting this form, I consent to receive marketing emails from the LF and its projects regarding their events, training, research, developments, and related announcements. I understand that I can unsubscribe at any time using the links in the footers of the emails I receive. [Privacy Policy](https://www.linuxfoundation.org/privacy/).

-
-
-
-
