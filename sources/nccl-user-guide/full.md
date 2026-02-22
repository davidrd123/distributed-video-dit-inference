---
title: "NCCL User Guide"
source_url: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
fetch_date: 2026-02-22
source_type: docs
author: NVIDIA
conversion_notes: |
  Assembled from the NCCL User Guide v2.29 multi-page Sphinx site.
  Raw index: sources/nccl-user-guide/raw/page.html
  Raw subpages: sources/nccl-user-guide/raw/subpages/*.html

  Policy: strip site chrome only; preserve all page body content.
  Conversion: for each HTML page, extract the Sphinx article body, remove permalink header anchors,
  unwrap Sphinx span/section/div wrappers, replace figures/images with bracketed descriptions, then
  convert via pandoc (HTML → GitHub-flavored Markdown) and concatenate pages with '---' separators.

  Note: the index TOC links to py-modindex.html, but it returned 404 at fetch time.
  Included pages (in order):
    - index (table of contents)
    - overview.html
    - setup.html
    - usage.html
    - usage/communicators.html
    - usage/collectives.html
    - usage/data.html
    - usage/streams.html
    - usage/groups.html
    - usage/p2p.html
    - usage/threadsafety.html
    - usage/inplace.html
    - usage/cudagraph.html
    - usage/bufferreg.html
    - usage/deviceapi.html
    - api.html
    - api/comms.html
    - api/colls.html
    - api/group.html
    - api/p2p.html
    - api/types.html
    - api/ops.html
    - api/flags.html
    - api/device.html
    - nccl1.html
    - examples.html
    - mpi.html
    - env.html
    - troubleshooting.html
    - troubleshooting/ras.html
    - genindex.html
    - search.html
---

# NVIDIA Collective Communication Library (NCCL) Documentation

Contents:

  - [Overview of NCCL](overview.html)
  - [Setup](setup.html)
  - [Using NCCL](usage.html)
      - [Creating a Communicator](usage/communicators.html)
          - [Creating a communicator with options](usage/communicators.html#creating-a-communicator-with-options)
          - [Creating a communicator using multiple ncclUniqueIds](usage/communicators.html#creating-a-communicator-using-multiple-nccluniqueids)
          - [Shrinking a communicator](usage/communicators.html#shrinking-a-communicator)
          - [Growing a communicator](usage/communicators.html#growing-a-communicator)
          - [Creating more communicators](usage/communicators.html#creating-more-communicators)
          - [Using multiple NCCL communicators concurrently](usage/communicators.html#using-multiple-nccl-communicators-concurrently)
          - [Finalizing a communicator](usage/communicators.html#finalizing-a-communicator)
          - [Destroying a communicator](usage/communicators.html#destroying-a-communicator)
      - [Error handling and communicator abort](usage/communicators.html#error-handling-and-communicator-abort)
          - [Asynchronous errors and error handling](usage/communicators.html#asynchronous-errors-and-error-handling)
      - [Fault Tolerance](usage/communicators.html#fault-tolerance)
      - [Quality of Service](usage/communicators.html#quality-of-service)
      - [Collective Operations](usage/collectives.html)
          - [AllReduce](usage/collectives.html#allreduce)
          - [Broadcast](usage/collectives.html#broadcast)
          - [Reduce](usage/collectives.html#reduce)
          - [AllGather](usage/collectives.html#allgather)
          - [ReduceScatter](usage/collectives.html#reducescatter)
          - [AlltoAll](usage/collectives.html#alltoall)
          - [Gather](usage/collectives.html#gather)
          - [Scatter](usage/collectives.html#scatter)
      - [Data Pointers](usage/data.html)
      - [CUDA Stream Semantics](usage/streams.html)
          - [Mixing Multiple Streams within the same ncclGroupStart/End() group](usage/streams.html#mixing-multiple-streams-within-the-same-ncclgroupstart-end-group)
      - [Group Calls](usage/groups.html)
          - [Management Of Multiple GPUs From One Thread](usage/groups.html#management-of-multiple-gpus-from-one-thread)
          - [Aggregated Operations (2.2 and later)](usage/groups.html#aggregated-operations-2-2-and-later)
          - [Group Operation Ordering Semantics](usage/groups.html#group-operation-ordering-semantics)
          - [Nonblocking Group Operation](usage/groups.html#nonblocking-group-operation)
      - [Point-to-point communication](usage/p2p.html)
          - [Two-sided communication](usage/p2p.html#two-sided-communication)
              - [Sendrecv](usage/p2p.html#sendrecv)
              - [One-to-all (scatter)](usage/p2p.html#one-to-all-scatter)
              - [All-to-one (gather)](usage/p2p.html#all-to-one-gather)
              - [All-to-all](usage/p2p.html#all-to-all)
              - [Neighbor exchange](usage/p2p.html#neighbor-exchange)
          - [One-sided communication](usage/p2p.html#one-sided-communication)
              - [PutSignal and WaitSignal](usage/p2p.html#putsignal-and-waitsignal)
              - [Barrier](usage/p2p.html#barrier)
              - [All-to-all](usage/p2p.html#id1)
      - [Thread Safety](usage/threadsafety.html)
      - [In-place Operations](usage/inplace.html)
      - [Using NCCL with CUDA Graphs](usage/cudagraph.html)
      - [User Buffer Registration](usage/bufferreg.html)
          - [NVLink Sharp Buffer Registration](usage/bufferreg.html#nvlink-sharp-buffer-registration)
          - [IB Sharp Buffer Registration](usage/bufferreg.html#ib-sharp-buffer-registration)
          - [General Buffer Registration](usage/bufferreg.html#general-buffer-registration)
          - [Buffer Registration and PXN](usage/bufferreg.html#buffer-registration-and-pxn)
          - [Memory Allocator](usage/bufferreg.html#memory-allocator)
          - [Window Registration](usage/bufferreg.html#window-registration)
          - [Zero-CTA Optimization](usage/bufferreg.html#zero-cta-optimization)
      - [Device-Initiated Communication](usage/deviceapi.html)
          - [Device API](usage/deviceapi.html#device-api)
          - [Requirements](usage/deviceapi.html#requirements)
          - [Host-Side Setup](usage/deviceapi.html#host-side-setup)
          - [Simple LSA Kernel](usage/deviceapi.html#simple-lsa-kernel)
          - [Multimem Device Kernel](usage/deviceapi.html#multimem-device-kernel)
          - [Thread Groups](usage/deviceapi.html#thread-groups)
          - [Teams](usage/deviceapi.html#teams)
          - [Host-Accessible Device Pointer Functions](usage/deviceapi.html#host-accessible-device-pointer-functions)
          - [GIN Device Kernel](usage/deviceapi.html#gin-device-kernel)
  - [NCCL API](api.html)
      - [Communicator Creation and Management Functions](api/comms.html)
          - [ncclGetLastError](api/comms.html#ncclgetlasterror)
          - [ncclGetErrorString](api/comms.html#ncclgeterrorstring)
          - [ncclGetVersion](api/comms.html#ncclgetversion)
          - [ncclGetUniqueId](api/comms.html#ncclgetuniqueid)
          - [ncclCommInitRank](api/comms.html#ncclcomminitrank)
          - [ncclCommInitAll](api/comms.html#ncclcomminitall)
          - [ncclCommInitRankConfig](api/comms.html#ncclcomminitrankconfig)
          - [ncclCommInitRankScalable](api/comms.html#ncclcomminitrankscalable)
          - [ncclCommSplit](api/comms.html#ncclcommsplit)
          - [ncclCommShrink](api/comms.html#ncclcommshrink)
          - [ncclCommGetUniqueId](api/comms.html#ncclcommgetuniqueid)
          - [ncclCommGrow](api/comms.html#ncclcommgrow)
          - [ncclCommFinalize](api/comms.html#ncclcommfinalize)
          - [ncclCommRevoke](api/comms.html#ncclcommrevoke)
          - [ncclCommDestroy](api/comms.html#ncclcommdestroy)
          - [ncclCommAbort](api/comms.html#ncclcommabort)
          - [ncclCommGetAsyncError](api/comms.html#ncclcommgetasyncerror)
          - [ncclCommCount](api/comms.html#ncclcommcount)
          - [ncclCommCuDevice](api/comms.html#ncclcommcudevice)
          - [ncclCommUserRank](api/comms.html#ncclcommuserrank)
          - [ncclCommRegister](api/comms.html#ncclcommregister)
          - [ncclCommDeregister](api/comms.html#ncclcommderegister)
          - [ncclCommWindowRegister](api/comms.html#ncclcommwindowregister)
          - [ncclCommWindowDeregister](api/comms.html#ncclcommwindowderegister)
          - [ncclMemAlloc](api/comms.html#ncclmemalloc)
          - [ncclMemFree](api/comms.html#ncclmemfree)
      - [Collective Communication Functions](api/colls.html)
          - [ncclAllReduce](api/colls.html#ncclallreduce)
          - [ncclBroadcast](api/colls.html#ncclbroadcast)
          - [ncclReduce](api/colls.html#ncclreduce)
          - [ncclAllGather](api/colls.html#ncclallgather)
          - [ncclReduceScatter](api/colls.html#ncclreducescatter)
          - [ncclAlltoAll](api/colls.html#ncclalltoall)
          - [ncclGather](api/colls.html#ncclgather)
          - [ncclScatter](api/colls.html#ncclscatter)
      - [Group Calls](api/group.html)
          - [ncclGroupStart](api/group.html#ncclgroupstart)
          - [ncclGroupEnd](api/group.html#ncclgroupend)
          - [ncclGroupSimulateEnd](api/group.html#ncclgroupsimulateend)
      - [Point To Point Communication Functions](api/p2p.html)
          - [Two-Sided Point-to-Point Operations](api/p2p.html#two-sided-point-to-point-operations)
              - [ncclSend](api/p2p.html#ncclsend)
              - [ncclRecv](api/p2p.html#ncclrecv)
          - [One-Sided Point-to-Point Operations (RMA)](api/p2p.html#one-sided-point-to-point-operations-rma)
              - [ncclPutSignal](api/p2p.html#ncclputsignal)
              - [ncclSignal](api/p2p.html#ncclsignal)
              - [ncclWaitSignal](api/p2p.html#ncclwaitsignal)
      - [Types](api/types.html)
          - [ncclComm\_t](api/types.html#ncclcomm-t)
          - [ncclResult\_t](api/types.html#ncclresult-t)
          - [ncclDataType\_t](api/types.html#nccldatatype-t)
          - [ncclRedOp\_t](api/types.html#ncclredop-t)
          - [ncclScalarResidence\_t](api/types.html#ncclscalarresidence-t)
          - [ncclConfig\_t](api/types.html#ncclconfig-t)
          - [ncclSimInfo\_t](api/types.html#ncclsiminfo-t)
          - [ncclWindow\_t](api/types.html#ncclwindow-t)
      - [User Defined Reduction Operators](api/ops.html)
          - [ncclRedOpCreatePreMulSum](api/ops.html#ncclredopcreatepremulsum)
          - [ncclRedOpDestroy](api/ops.html#ncclredopdestroy)
      - [NCCL API Supported Flags](api/flags.html)
          - [Window Registration Flags](api/flags.html#window-registration-flags)
          - [NCCL Communicator CTA Policy Flags](api/flags.html#nccl-communicator-cta-policy-flags)
          - [Communicator Shrink Flags](api/flags.html#communicator-shrink-flags)
      - [Device API](api/device.html)
          - [Host-Side Setup](api/device.html#host-side-setup)
              - [ncclDevComm](api/device.html#nccldevcomm)
              - [ncclDevCommCreate](api/device.html#nccldevcommcreate)
              - [ncclDevCommDestroy](api/device.html#nccldevcommdestroy)
              - [ncclDevCommRequirements](api/device.html#nccldevcommrequirements)
              - [ncclCommQueryProperties](api/device.html#ncclcommqueryproperties)
              - [ncclCommProperties\_t](api/device.html#ncclcommproperties-t)
              - [ncclGinType\_t](api/device.html#ncclgintype-t)
          - [LSA](api/device.html#lsa)
              - [ncclLsaBarrierSession](api/device.html#nccllsabarriersession)
              - [ncclGetPeerPointer](api/device.html#ncclgetpeerpointer)
              - [ncclGetLsaPointer](api/device.html#ncclgetlsapointer)
              - [ncclGetLocalPointer](api/device.html#ncclgetlocalpointer)
          - [Multimem](api/device.html#multimem)
              - [ncclGetLsaMultimemPointer](api/device.html#ncclgetlsamultimempointer)
          - [Host-Accessible Device Pointer Functions](api/device.html#host-accessible-device-pointer-functions)
              - [ncclGetLsaMultimemDevicePointer](api/device.html#ncclgetlsamultimemdevicepointer)
              - [ncclGetMultimemDevicePointer](api/device.html#ncclgetmultimemdevicepointer)
              - [ncclGetLsaDevicePointer](api/device.html#ncclgetlsadevicepointer)
              - [ncclGetPeerDevicePointer](api/device.html#ncclgetpeerdevicepointer)
          - [GIN](api/device.html#gin)
              - [ncclGin](api/device.html#ncclgin)
              - [Signals and Counters](api/device.html#signals-and-counters)
              - [ncclGinBarrierSession](api/device.html#ncclginbarriersession)
  - [Migrating from NCCL 1 to NCCL 2](nccl1.html)
      - [Initialization](nccl1.html#initialization)
      - [Communication](nccl1.html#communication)
      - [Counts](nccl1.html#counts)
      - [In-place usage for AllGather and ReduceScatter](nccl1.html#in-place-usage-for-allgather-and-reducescatter)
      - [AllGather arguments order](nccl1.html#allgather-arguments-order)
      - [Datatypes](nccl1.html#datatypes)
      - [Error codes](nccl1.html#error-codes)
  - [Examples](examples.html)
      - [Communicator Creation and Destruction Examples](examples.html#communicator-creation-and-destruction-examples)
          - [Example 1: Single Process, Single Thread, Multiple Devices](examples.html#example-1-single-process-single-thread-multiple-devices)
          - [Example 2: One Device per Process or Thread](examples.html#example-2-one-device-per-process-or-thread)
          - [Example 3: Multiple Devices per Thread](examples.html#example-3-multiple-devices-per-thread)
          - [Example 4: Multiple communicators per device](examples.html#example-4-multiple-communicators-per-device)
      - [Communication Examples](examples.html#communication-examples)
          - [Example 1: One Device per Process or Thread](examples.html#example-1-one-device-per-process-or-thread)
          - [Example 2: Multiple Devices per Thread](examples.html#example-2-multiple-devices-per-thread)
  - [NCCL and MPI](mpi.html)
      - [API](mpi.html#api)
          - [Using multiple devices per process](mpi.html#using-multiple-devices-per-process)
          - [ReduceScatter operation](mpi.html#reducescatter-operation)
          - [Send and Receive counts](mpi.html#send-and-receive-counts)
          - [Other collectives and point-to-point operations](mpi.html#other-collectives-and-point-to-point-operations)
          - [In-place operations](mpi.html#in-place-operations)
      - [Using NCCL within an MPI Program](mpi.html#using-nccl-within-an-mpi-program)
          - [MPI Progress](mpi.html#mpi-progress)
          - [Inter-GPU Communication with CUDA-aware MPI](mpi.html#inter-gpu-communication-with-cuda-aware-mpi)
  - [Environment Variables](env.html)
      - [System configuration](env.html#system-configuration)
          - [NCCL\_SOCKET\_IFNAME](env.html#nccl-socket-ifname)
              - [Values accepted](env.html#values-accepted)
          - [NCCL\_SOCKET\_FAMILY](env.html#nccl-socket-family)
              - [Values accepted](env.html#id2)
          - [NCCL\_SOCKET\_RETRY\_CNT](env.html#nccl-socket-retry-cnt)
              - [Values accepted](env.html#id3)
          - [NCCL\_SOCKET\_RETRY\_SLEEP\_MSEC](env.html#nccl-socket-retry-sleep-msec)
              - [Values accepted](env.html#id4)
          - [NCCL\_SOCKET\_POLL\_TIMEOUT\_MSEC](env.html#nccl-socket-poll-timeout-msec)
              - [Values accepted](env.html#id5)
          - [NCCL\_SOCKET\_NTHREADS](env.html#nccl-socket-nthreads)
              - [Values accepted](env.html#id6)
          - [NCCL\_NSOCKS\_PERTHREAD](env.html#nccl-nsocks-perthread)
              - [Values accepted](env.html#id7)
          - [NCCL\_CROSS\_NIC](env.html#nccl-cross-nic)
              - [Values accepted](env.html#id8)
          - [NCCL\_IB\_HCA](env.html#nccl-ib-hca)
              - [Values accepted](env.html#id9)
          - [NCCL\_IB\_TIMEOUT](env.html#nccl-ib-timeout)
              - [Values accepted](env.html#id10)
          - [NCCL\_IB\_RETRY\_CNT](env.html#nccl-ib-retry-cnt)
              - [Values accepted](env.html#id11)
          - [NCCL\_IB\_GID\_INDEX](env.html#nccl-ib-gid-index)
              - [Values accepted](env.html#id12)
          - [NCCL\_IB\_ADDR\_FAMILY](env.html#nccl-ib-addr-family)
              - [Values accepted](env.html#id13)
          - [NCCL\_IB\_ADDR\_RANGE](env.html#nccl-ib-addr-range)
              - [Values accepted](env.html#id14)
          - [NCCL\_IB\_ROCE\_VERSION\_NUM](env.html#nccl-ib-roce-version-num)
              - [Values accepted](env.html#id15)
          - [NCCL\_IB\_SL](env.html#nccl-ib-sl)
              - [Values accepted](env.html#id16)
          - [NCCL\_IB\_TC](env.html#nccl-ib-tc)
              - [Values accepted](env.html#id17)
          - [NCCL\_IB\_FIFO\_TC](env.html#nccl-ib-fifo-tc)
              - [Values accepted](env.html#id18)
          - [NCCL\_IB\_RETURN\_ASYNC\_EVENTS](env.html#nccl-ib-return-async-events)
              - [Values accepted](env.html#id19)
          - [NCCL\_OOB\_NET\_ENABLE](env.html#nccl-oob-net-enable)
              - [Values accepted](env.html#id20)
          - [NCCL\_OOB\_NET\_IFNAME](env.html#nccl-oob-net-ifname)
              - [Values accepted](env.html#id21)
          - [NCCL\_UID\_STAGGER\_THRESHOLD](env.html#nccl-uid-stagger-threshold)
              - [Values accepted](env.html#id22)
          - [NCCL\_UID\_STAGGER\_RATE](env.html#nccl-uid-stagger-rate)
              - [Values accepted](env.html#id23)
          - [NCCL\_NET](env.html#nccl-net)
              - [Values accepted](env.html#id24)
          - [NCCL\_NET\_PLUGIN](env.html#nccl-net-plugin)
              - [Values accepted](env.html#id25)
          - [NCCL\_TUNER\_PLUGIN](env.html#nccl-tuner-plugin)
              - [Values accepted](env.html#id26)
          - [NCCL\_PROFILER\_PLUGIN](env.html#nccl-profiler-plugin)
              - [Values accepted](env.html#id27)
          - [NCCL\_ENV\_PLUGIN](env.html#nccl-env-plugin)
              - [Values accepted](env.html#id28)
          - [NCCL\_IGNORE\_CPU\_AFFINITY](env.html#nccl-ignore-cpu-affinity)
              - [Values accepted](env.html#id29)
          - [NCCL\_CONF\_FILE](env.html#nccl-conf-file)
              - [Values accepted](env.html#id30)
          - [NCCL\_DEBUG](env.html#nccl-debug)
              - [Values accepted](env.html#id32)
          - [NCCL\_DEBUG\_FILE](env.html#nccl-debug-file)
              - [Values accepted](env.html#id33)
          - [NCCL\_DEBUG\_SUBSYS](env.html#nccl-debug-subsys)
              - [Values accepted](env.html#id34)
          - [NCCL\_DEBUG\_TIMESTAMP\_FORMAT](env.html#nccl-debug-timestamp-format)
              - [Value accepted](env.html#value-accepted)
          - [NCCL\_DEBUG\_TIMESTAMP\_LEVELS](env.html#nccl-debug-timestamp-levels)
              - [Value accepted](env.html#id35)
          - [NCCL\_COLLNET\_ENABLE](env.html#nccl-collnet-enable)
              - [Value accepted](env.html#id36)
          - [NCCL\_COLLNET\_NODE\_THRESHOLD](env.html#nccl-collnet-node-threshold)
              - [Value accepted](env.html#id37)
          - [NCCL\_CTA\_POLICY](env.html#nccl-cta-policy)
              - [Value accepted](env.html#id38)
          - [NCCL\_NETDEVS\_POLICY](env.html#nccl-netdevs-policy)
              - [Value accepted](env.html#id39)
          - [NCCL\_TOPO\_FILE](env.html#nccl-topo-file)
              - [Value accepted](env.html#id40)
          - [NCCL\_TOPO\_DUMP\_FILE](env.html#nccl-topo-dump-file)
              - [Value accepted](env.html#id41)
          - [NCCL\_SET\_THREAD\_NAME](env.html#nccl-set-thread-name)
              - [Value accepted](env.html#id42)
      - [Debugging](env.html#debugging)
          - [NCCL\_P2P\_DISABLE](env.html#nccl-p2p-disable)
              - [Values accepted](env.html#id43)
          - [NCCL\_P2P\_LEVEL](env.html#nccl-p2p-level)
              - [Values accepted](env.html#id44)
              - [Integer Values (Legacy)](env.html#integer-values-legacy)
          - [NCCL\_P2P\_DIRECT\_DISABLE](env.html#nccl-p2p-direct-disable)
              - [Values accepted](env.html#id45)
          - [NCCL\_SHM\_DISABLE](env.html#nccl-shm-disable)
              - [Values accepted](env.html#id46)
          - [NCCL\_BUFFSIZE](env.html#nccl-buffsize)
              - [Values accepted](env.html#id47)
          - [NCCL\_NTHREADS](env.html#nccl-nthreads)
              - [Values accepted](env.html#id48)
          - [NCCL\_MAX\_NCHANNELS](env.html#nccl-max-nchannels)
              - [Values accepted](env.html#id49)
          - [NCCL\_MIN\_NCHANNELS](env.html#nccl-min-nchannels)
              - [Values accepted](env.html#id50)
          - [NCCL\_CHECKS\_DISABLE](env.html#nccl-checks-disable)
              - [Values accepted](env.html#id51)
          - [NCCL\_CHECK\_POINTERS](env.html#nccl-check-pointers)
              - [Values accepted](env.html#id52)
          - [NCCL\_LAUNCH\_MODE](env.html#nccl-launch-mode)
              - [Values accepted](env.html#id53)
          - [NCCL\_IB\_DISABLE](env.html#nccl-ib-disable)
              - [Values accepted](env.html#id54)
          - [NCCL\_IB\_AR\_THRESHOLD](env.html#nccl-ib-ar-threshold)
              - [Values accepted](env.html#id55)
          - [NCCL\_IB\_QPS\_PER\_CONNECTION](env.html#nccl-ib-qps-per-connection)
              - [Values accepted](env.html#id56)
          - [NCCL\_IB\_SPLIT\_DATA\_ON\_QPS](env.html#nccl-ib-split-data-on-qps)
              - [Values accepted](env.html#id57)
          - [NCCL\_IB\_CUDA\_SUPPORT](env.html#nccl-ib-cuda-support)
              - [Values accepted](env.html#id58)
          - [NCCL\_IB\_PCI\_RELAXED\_ORDERING](env.html#nccl-ib-pci-relaxed-ordering)
              - [Values accepted](env.html#id59)
          - [NCCL\_IB\_ADAPTIVE\_ROUTING](env.html#nccl-ib-adaptive-routing)
              - [Values accepted](env.html#id60)
          - [NCCL\_IB\_ECE\_ENABLE](env.html#nccl-ib-ece-enable)
              - [Values accepted](env.html#id61)
          - [NCCL\_MEM\_SYNC\_DOMAIN](env.html#nccl-mem-sync-domain)
              - [Values accepted](env.html#id62)
          - [NCCL\_CUMEM\_ENABLE](env.html#nccl-cumem-enable)
              - [Values accepted](env.html#id63)
          - [NCCL\_CUMEM\_HOST\_ENABLE](env.html#nccl-cumem-host-enable)
              - [Values accepted](env.html#id64)
          - [NCCL\_NET\_GDR\_LEVEL (formerly NCCL\_IB\_GDR\_LEVEL)](env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level)
              - [Values accepted](env.html#id65)
              - [Integer Values (Legacy)](env.html#id66)
          - [NCCL\_NET\_GDR\_C2C](env.html#nccl-net-gdr-c2c)
              - [Values accepted](env.html#id67)
          - [NCCL\_NET\_GDR\_READ](env.html#nccl-net-gdr-read)
              - [Values accepted](env.html#id68)
          - [NCCL\_NET\_SHARED\_BUFFERS](env.html#nccl-net-shared-buffers)
              - [Value accepted](env.html#id69)
          - [NCCL\_NET\_SHARED\_COMMS](env.html#nccl-net-shared-comms)
              - [Value accepted](env.html#id70)
          - [NCCL\_SINGLE\_RING\_THRESHOLD](env.html#nccl-single-ring-threshold)
              - [Values accepted](env.html#id71)
          - [NCCL\_LL\_THRESHOLD](env.html#nccl-ll-threshold)
              - [Values accepted](env.html#id72)
          - [NCCL\_TREE\_THRESHOLD](env.html#nccl-tree-threshold)
              - [Values accepted](env.html#id73)
          - [NCCL\_ALGO](env.html#nccl-algo)
              - [Values accepted](env.html#id74)
          - [NCCL\_PROTO](env.html#nccl-proto)
              - [Values accepted](env.html#id75)
          - [NCCL\_NVB\_DISABLE](env.html#nccl-nvb-disable)
              - [Value accepted](env.html#id76)
          - [NCCL\_PXN\_DISABLE](env.html#nccl-pxn-disable)
              - [Value accepted](env.html#id77)
          - [NCCL\_P2P\_PXN\_LEVEL](env.html#nccl-p2p-pxn-level)
              - [Value accepted](env.html#id78)
          - [NCCL\_PXN\_C2C](env.html#nccl-pxn-c2c)
              - [Value accepted](env.html#id79)
          - [NCCL\_RUNTIME\_CONNECT](env.html#nccl-runtime-connect)
              - [Value accepted](env.html#id80)
          - [NCCL\_GRAPH\_REGISTER](env.html#nccl-graph-register)
              - [Value accepted](env.html#id82)
          - [NCCL\_LOCAL\_REGISTER](env.html#nccl-local-register)
              - [Value accepted](env.html#id83)
          - [NCCL\_LEGACY\_CUDA\_REGISTER](env.html#nccl-legacy-cuda-register)
              - [Value accepted](env.html#id84)
          - [NCCL\_WIN\_ENABLE](env.html#nccl-win-enable)
              - [Value accepted](env.html#id85)
          - [NCCL\_SET\_STACK\_SIZE](env.html#nccl-set-stack-size)
              - [Value accepted](env.html#id86)
          - [NCCL\_GRAPH\_MIXING\_SUPPORT](env.html#nccl-graph-mixing-support)
              - [Value accepted](env.html#id88)
          - [NCCL\_DMABUF\_ENABLE](env.html#nccl-dmabuf-enable)
              - [Value accepted](env.html#id89)
          - [NCCL\_P2P\_NET\_CHUNKSIZE](env.html#nccl-p2p-net-chunksize)
              - [Values accepted](env.html#id90)
          - [NCCL\_P2P\_LL\_THRESHOLD](env.html#nccl-p2p-ll-threshold)
              - [Values accepted](env.html#id91)
          - [NCCL\_ALLOC\_P2P\_NET\_LL\_BUFFERS](env.html#nccl-alloc-p2p-net-ll-buffers)
              - [Values accepted](env.html#id92)
          - [NCCL\_COMM\_BLOCKING](env.html#nccl-comm-blocking)
              - [Values accepted](env.html#id93)
          - [NCCL\_CGA\_CLUSTER\_SIZE](env.html#nccl-cga-cluster-size)
              - [Values accepted](env.html#id94)
          - [NCCL\_MAX\_CTAS](env.html#nccl-max-ctas)
              - [Values accepted](env.html#id95)
          - [NCCL\_MIN\_CTAS](env.html#nccl-min-ctas)
              - [Values accepted](env.html#id96)
          - [NCCL\_NVLS\_ENABLE](env.html#nccl-nvls-enable)
              - [Values accepted](env.html#id97)
          - [NCCL\_IB\_MERGE\_NICS](env.html#nccl-ib-merge-nics)
              - [Values accepted](env.html#id98)
          - [NCCL\_MNNVL\_ENABLE](env.html#nccl-mnnvl-enable)
              - [Values accepted](env.html#id99)
          - [NCCL\_MNNVL\_UUID](env.html#nccl-mnnvl-uuid)
              - [Values accepted](env.html#id100)
          - [NCCL\_MNNVL\_CLIQUE\_ID](env.html#nccl-mnnvl-clique-id)
              - [Values accepted](env.html#id101)
          - [NCCL\_RAS\_ENABLE](env.html#nccl-ras-enable)
              - [Values accepted](env.html#id102)
          - [NCCL\_RAS\_ADDR](env.html#nccl-ras-addr)
              - [Values accepted](env.html#id103)
          - [NCCL\_RAS\_TIMEOUT\_FACTOR](env.html#nccl-ras-timeout-factor)
              - [Values accepted](env.html#id104)
          - [NCCL\_LAUNCH\_ORDER\_IMPLICIT](env.html#nccl-launch-order-implicit)
              - [Values accepted](env.html#id106)
          - [NCCL\_LAUNCH\_RACE\_FATAL](env.html#nccl-launch-race-fatal)
              - [Values accepted](env.html#id107)
          - [NCCL\_IPC\_USE\_ABSTRACT\_SOCKET](env.html#nccl-ipc-use-abstract-socket)
              - [Values accepted](env.html#id108)
  - [Troubleshooting](troubleshooting.html)
      - [Errors](troubleshooting.html#errors)
      - [RAS](troubleshooting.html#ras)
          - [RAS](troubleshooting/ras.html)
              - [Principle of Operation](troubleshooting/ras.html#principle-of-operation)
              - [RAS Queries](troubleshooting/ras.html#ras-queries)
              - [Sample Output](troubleshooting/ras.html#sample-output)
              - [JSON Output](troubleshooting/ras.html#json-output)
              - [Monitoring Mode](troubleshooting/ras.html#monitoring-mode)
      - [GPU Direct](troubleshooting.html#gpu-direct)
          - [GPU-to-GPU communication](troubleshooting.html#gpu-to-gpu-communication)
          - [GPU-to-NIC communication](troubleshooting.html#gpu-to-nic-communication)
          - [PCI Access Control Services (ACS)](troubleshooting.html#pci-access-control-services-acs)
      - [Topology detection](troubleshooting.html#topology-detection)
      - [Memory issues](troubleshooting.html#memory-issues)
          - [Shared memory](troubleshooting.html#shared-memory)
          - [Stack size](troubleshooting.html#stack-size)
          - [Unified Memory (UVM)](troubleshooting.html#unified-memory-uvm)
      - [Networking issues](troubleshooting.html#networking-issues)
          - [IP Network Interfaces](troubleshooting.html#ip-network-interfaces)
          - [IP Ports](troubleshooting.html#ip-ports)
          - [InfiniBand](troubleshooting.html#infiniband)
          - [RDMA over Converged Ethernet (RoCE)](troubleshooting.html#rdma-over-converged-ethernet-roce)

# Indices and tables

  - [Index](genindex.html)

  - [Module Index](py-modindex.html)

  - [Search Page](search.html)

---

# Overview of NCCL

The NVIDIA Collective Communications Library (NCCL, pronounced “Nickel”) is a library providing inter-GPU communication primitives that are topology-aware and can be easily integrated into applications.

NCCL implements both collective communication and point-to-point send/receive primitives. It is not a full-blown parallel programming framework; rather, it is a library focused on accelerating inter-GPU communication.

NCCL provides the following collective communication primitives :

  - AllReduce

  - Broadcast

  - Reduce

  - AllGather

  - ReduceScatter

  - AlltoAll

  - Gather

  - Scatter

Additionally, it allows for point-to-point send/receive communication which allows for scatter, gather, or all-to-all operations.

Tight synchronization between communicating processors is a key aspect of collective communication. CUDA based collectives would traditionally be realized through a combination of CUDA memory copy operations and CUDA kernels for local reductions. NCCL, on the other hand, implements each collective in a single kernel handling both communication and computation operations. This allows for fast synchronization and minimizes the resources needed to reach peak bandwidth.

NCCL conveniently removes the need for developers to optimize their applications for specific machines. NCCL provides fast collectives over multiple GPUs both within and across nodes. It supports a variety of interconnect technologies including PCIe, NVLINK, InfiniBand Verbs, and IP sockets.

Next to performance, ease of programming was the primary consideration in the design of NCCL. NCCL uses a simple C API, which can be easily accessed from a variety of programming languages. NCCL closely follows the popular collectives API defined by MPI (Message Passing Interface). Anyone familiar with MPI will thus find NCCL’s API very natural to use. In a minor departure from MPI, NCCL collectives take a “stream” argument which provides direct integration with the CUDA programming model. Finally, NCCL is compatible with virtually any multi-GPU parallelization model, for example:

  - single-threaded control of all GPUs

  - multi-threaded, for example, using one thread per GPU

  - multi-process, for example, MPI

NCCL has found great application in Deep Learning Frameworks, where the AllReduce collective is heavily used for neural network training. Efficient scaling of neural network training is possible with the multi-GPU and multi node communication provided by NCCL.

---

# Setup

NCCL is a communication library providing optimized GPU-to-GPU communication for high-performance applications. It is not, like MPI, providing a parallel environment including a process launcher and manager. NCCL relies therefore on the application’s process management system and CPU-side communication system for its own bootstrap.

Similarly to MPI and other libraries which are optimized for performance, NCCL does not provide secure network communication between GPUs. It is therefore the responsibility of the user to ensure NCCL operates over a secure network, both for bootstrap (controlled by [NCCL\_SOCKET\_IFNAME](env.html#nccl-socket-ifname)) and for high-speed communication.

---

# Using NCCL

Using NCCL is similar to using any other library in your code:

1.  Install the NCCL library on your system

2.  Modify your application to link to that library

3.  Include the header file nccl.h in your application

4.  Create a communicator (see [Creating a Communicator](usage/communicators.html#communicator-label))

5.  Use NCCL collective communication primitives to perform data communication. You can familiarize yourself with the [NCCL API](api.html#api-label) documentation to maximize your usage performance.

Collective communication primitives are common patterns of data transfer among a group of CUDA devices. A communication algorithm involves many processors that are communicating together. Each CUDA device is identified within the communication group by a zero-based index or rank. Each rank uses a communicator object to refer to the collection of GPUs that are intended to work together. The creation of a communicator is the first step needed before launching any communication operation.

  - [Creating a Communicator](usage/communicators.html)
      - [Creating a communicator with options](usage/communicators.html#creating-a-communicator-with-options)
      - [Creating a communicator using multiple ncclUniqueIds](usage/communicators.html#creating-a-communicator-using-multiple-nccluniqueids)
      - [Shrinking a communicator](usage/communicators.html#shrinking-a-communicator)
      - [Growing a communicator](usage/communicators.html#growing-a-communicator)
      - [Creating more communicators](usage/communicators.html#creating-more-communicators)
      - [Using multiple NCCL communicators concurrently](usage/communicators.html#using-multiple-nccl-communicators-concurrently)
      - [Finalizing a communicator](usage/communicators.html#finalizing-a-communicator)
      - [Destroying a communicator](usage/communicators.html#destroying-a-communicator)
  - [Error handling and communicator abort](usage/communicators.html#error-handling-and-communicator-abort)
      - [Asynchronous errors and error handling](usage/communicators.html#asynchronous-errors-and-error-handling)
  - [Fault Tolerance](usage/communicators.html#fault-tolerance)
  - [Quality of Service](usage/communicators.html#quality-of-service)
  - [Collective Operations](usage/collectives.html)
      - [AllReduce](usage/collectives.html#allreduce)
      - [Broadcast](usage/collectives.html#broadcast)
      - [Reduce](usage/collectives.html#reduce)
      - [AllGather](usage/collectives.html#allgather)
      - [ReduceScatter](usage/collectives.html#reducescatter)
      - [AlltoAll](usage/collectives.html#alltoall)
      - [Gather](usage/collectives.html#gather)
      - [Scatter](usage/collectives.html#scatter)
  - [Data Pointers](usage/data.html)
  - [CUDA Stream Semantics](usage/streams.html)
      - [Mixing Multiple Streams within the same ncclGroupStart/End() group](usage/streams.html#mixing-multiple-streams-within-the-same-ncclgroupstart-end-group)
  - [Group Calls](usage/groups.html)
      - [Management Of Multiple GPUs From One Thread](usage/groups.html#management-of-multiple-gpus-from-one-thread)
      - [Aggregated Operations (2.2 and later)](usage/groups.html#aggregated-operations-2-2-and-later)
      - [Group Operation Ordering Semantics](usage/groups.html#group-operation-ordering-semantics)
      - [Nonblocking Group Operation](usage/groups.html#nonblocking-group-operation)
  - [Point-to-point communication](usage/p2p.html)
      - [Two-sided communication](usage/p2p.html#two-sided-communication)
          - [Sendrecv](usage/p2p.html#sendrecv)
          - [One-to-all (scatter)](usage/p2p.html#one-to-all-scatter)
          - [All-to-one (gather)](usage/p2p.html#all-to-one-gather)
          - [All-to-all](usage/p2p.html#all-to-all)
          - [Neighbor exchange](usage/p2p.html#neighbor-exchange)
      - [One-sided communication](usage/p2p.html#one-sided-communication)
          - [PutSignal and WaitSignal](usage/p2p.html#putsignal-and-waitsignal)
          - [Barrier](usage/p2p.html#barrier)
          - [All-to-all](usage/p2p.html#id1)
  - [Thread Safety](usage/threadsafety.html)
  - [In-place Operations](usage/inplace.html)
  - [Using NCCL with CUDA Graphs](usage/cudagraph.html)
  - [User Buffer Registration](usage/bufferreg.html)
      - [NVLink Sharp Buffer Registration](usage/bufferreg.html#nvlink-sharp-buffer-registration)
      - [IB Sharp Buffer Registration](usage/bufferreg.html#ib-sharp-buffer-registration)
      - [General Buffer Registration](usage/bufferreg.html#general-buffer-registration)
      - [Buffer Registration and PXN](usage/bufferreg.html#buffer-registration-and-pxn)
      - [Memory Allocator](usage/bufferreg.html#memory-allocator)
      - [Window Registration](usage/bufferreg.html#window-registration)
      - [Zero-CTA Optimization](usage/bufferreg.html#zero-cta-optimization)
  - [Device-Initiated Communication](usage/deviceapi.html)
      - [Device API](usage/deviceapi.html#device-api)
      - [Requirements](usage/deviceapi.html#requirements)
      - [Host-Side Setup](usage/deviceapi.html#host-side-setup)
      - [Simple LSA Kernel](usage/deviceapi.html#simple-lsa-kernel)
      - [Multimem Device Kernel](usage/deviceapi.html#multimem-device-kernel)
      - [Thread Groups](usage/deviceapi.html#thread-groups)
      - [Teams](usage/deviceapi.html#teams)
      - [Host-Accessible Device Pointer Functions](usage/deviceapi.html#host-accessible-device-pointer-functions)
      - [GIN Device Kernel](usage/deviceapi.html#gin-device-kernel)

---

# Creating a Communicator

When creating a communicator, a unique rank between 0 and n-1 has to be assigned to each of the n CUDA devices which are part of the communicator. Using the same CUDA device multiple times as different ranks of the same NCCL communicator is not supported and may lead to hangs.

Given a static mapping of ranks to CUDA devices, the [`ncclCommInitRank()`](../api/comms.html#c.ncclCommInitRank "ncclCommInitRank"), [`ncclCommInitRankConfig()`](../api/comms.html#c.ncclCommInitRankConfig "ncclCommInitRankConfig") and [`ncclCommInitAll()`](../api/comms.html#c.ncclCommInitAll "ncclCommInitAll") functions will create communicator objects, each communicator object being associated to a fixed rank and CUDA device. Those objects will then be used to launch communication operations.

Before calling [`ncclCommInitRank()`](../api/comms.html#c.ncclCommInitRank "ncclCommInitRank"), you need to first create a unique object which will be used by all processes and threads to synchronize and understand they are part of the same communicator. This is done by calling the [`ncclGetUniqueId()`](../api/comms.html#c.ncclGetUniqueId "ncclGetUniqueId") function.

The [`ncclGetUniqueId()`](../api/comms.html#c.ncclGetUniqueId "ncclGetUniqueId") function returns an ID which has to be broadcast to all participating threads and processes using any CPU communication system, for example, passing the ID pointer to multiple threads, or broadcasting it to other processes using MPI or another parallel environment using, for example, sockets.

You can also call the ncclCommInitAll operation to create n communicator objects at once within a single process. As it is limited to a single process, this function does not permit inter-node communication. ncclCommInitAll is equivalent to calling a combination of ncclGetUniqueId and ncclCommInitRank.

The following sample code is a simplified implementation of ncclCommInitAll.

    ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
      ncclUniqueId Id;
      ncclGetUniqueId(&Id);
      ncclGroupStart();
      for (int i=0; i<ndev; i++) {
        cudaSetDevice(devlist[i]);
        ncclCommInitRank(comm+i, ndev, Id, i);
      }
      ncclGroupEnd();
    }

Related links:

>   - [`ncclCommInitAll()`](../api/comms.html#c.ncclCommInitAll "ncclCommInitAll")
> 
>   - [`ncclGetUniqueId()`](../api/comms.html#c.ncclGetUniqueId "ncclGetUniqueId")
> 
>   - [`ncclCommInitRank()`](../api/comms.html#c.ncclCommInitRank "ncclCommInitRank")

## Creating a communicator with options

The [`ncclCommInitRankConfig()`](../api/comms.html#c.ncclCommInitRankConfig "ncclCommInitRankConfig") function allows to create a NCCL communicator with specific options.

The config parameters NCCL supports are listed here [ncclConfig\_t](../api/types.html#ncclconfig).

For example, “blocking” can be set to 0 to ask NCCL to never block in any NCCL call, and at the same time other config parameters can be set as well to more precisely define communicator behavior. A simple example code is shown below:

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    config.minCTAs = 4;
    config.maxCTAs = 16;
    config.cgaClusterSize = 2;
    config.netName = "Socket";
    CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
    do {
      CHECK(ncclCommGetAsyncError(comm, &state));
      // Handle outside events, timeouts, progress, ...
    } while(state == ncclInProgress);

Related link: [`ncclCommGetAsyncError()`](../api/comms.html#c.ncclCommGetAsyncError "ncclCommGetAsyncError")

## Creating a communicator using multiple ncclUniqueIds

The [`ncclCommInitRankScalable()`](../api/comms.html#c.ncclCommInitRankScalable "ncclCommInitRankScalable") function enables the creation of a NCCL communicator using many ncclUniqueIds. All NCCL ranks have to provide the same array of ncclUniqueIds (same ncclUniqueIds, and in with the same order). For the best performance, we recommend distributing the ncclUniqueIds as evenly as possible amongst the NCCL ranks.

Internally, NCCL ranks will mostly communicate with a single ncclUniqueId. Therefore, to obtain the best results, we recommend to evenly distribute ncclUniqueIds accross the ranks.

The following function can be used to decide if a NCCL rank should create a ncclUniqueIds:

    bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
      const int rmr = nRanks % nIds;
      const int rpr = nRanks / nIds;
      const int rlim = rmr * (rpr+1);
      if (rank < rlim) {
        return !(rank % (rpr + 1));
      } else {
        return !((rank - rlim) % rpr);
      }
    }

For example, if 3 ncclUniqueIds are to be distributed accross 7 NCCL ranks, the first ncclUniqueId will be associated to ranks 0-2, while the others will be associated to ranks 3-4, and 5-6. This function will therefore return true on rank 0, 3, and 5, and false otherwise.

Note: only the first ncclUniqueId will be used to create the communicator hash id, which is used to identify the communicator in the log file and in the replay tool.

## Shrinking a communicator

The [`ncclCommShrink()`](../api/comms.html#c.ncclCommShrink "ncclCommShrink") function allows you to create a new communicator by removing specific ranks from an existing one. This is useful when you need to exclude certain GPUs or nodes from a collective operation, for example in fault tolerance scenarios or when dynamically adjusting resource utilization.

The following example demonstrates how to create a new communicator by excluding rank 1:

    int excludeRanks[] = {1};  // Rank to exclude
    int excludeCount = 1;      // Number of ranks to exclude
    ncclComm_t newcomm;
    
    // Only ranks that will be in the new communicator should call ncclCommShrink
    if (myRank != 1) {
      ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL, NCCL_SHRINK_DEFAULT);
      if (res != ncclSuccess) {
        // Handle error
      }
      // Use the new communicator for collective operations
      // ...
      // When done, destroy the new communicator
      ncclCommDestroy(newcomm);
    }

When recovering from communication errors, you may want to use the error mode:

    if (myRank != 1) {
      // When shrinking after an error, use NCCL_SHRINK_ABORT to abort operations on the parent communicator
      // This mode is also useful when there might be ongoing operations on the parent communicator
      ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL, NCCL_SHRINK_ABORT);
      // ...
    }

Note that:

1.  Only ranks that will be part of the new communicator should call [`ncclCommShrink()`](../api/comms.html#c.ncclCommShrink "ncclCommShrink").

2.  Ranks listed in the exclusion list should not call this function.

3.  The new communicator will have ranks re-ordered to maintain contiguous numbering.

4.  You can use the ncclGroupStart/ncclGroupEnd mechanism to synchronize the creation of new communicators.

Related link: [`ncclCommShrink()`](../api/comms.html#c.ncclCommShrink "ncclCommShrink")

## Growing a communicator

The [`ncclCommGrow()`](../api/comms.html#c.ncclCommGrow "ncclCommGrow") function allows you to create a new communicator by adding new ranks to an existing one. This is useful when you need to dynamically scale up your computation by adding more GPUs or nodes to a running collective operation.

Growing a communicator involves coordination between existing ranks (from the parent communicator) and new ranks (joining the communicator). The process requires a coordinator rank from the existing communicator to generate a unique identifier using [`ncclCommGetUniqueId()`](../api/comms.html#c.ncclCommGetUniqueId "ncclCommGetUniqueId"), which is then distributed to all new ranks through an out-of-band mechanism (e.g., MPI, sockets, or shared memory).

The following example demonstrates how to grow a 4-rank communicator to 8 ranks:

    // Step 1: Coordinator (e.g., rank 0) generates the grow identifier
    ncclUniqueId growId;
    if (myRank == 0) {
      ncclResult_t res = ncclCommGetUniqueId(comm, &growId);
      if (res != ncclSuccess) {
        // Handle error
      }
      // Distribute growId to all new ranks using out-of-band communication
      // (e.g., MPI_Send, sockets, shared memory, etc.)
    }
    
    // Step 2: All existing ranks call ncclCommGrow
    ncclComm_t newcomm;
    ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, NULL);
    if (res != ncclSuccess) {
      // Handle error
    }
    
    // Step 3: New ranks (4-7) call ncclCommGrow with the received growId
    cudaSetDevice(myDevice);
    ncclComm_t newcomm;
    ncclResult_t res = ncclCommGrow(NULL, 8, &growId, myNewRank, &newcomm, NULL);
    
    // Step 4: Wait for grow operation to complete (if non-blocking)
    ncclResult_t asyncErr;
    do {
      res = ncclCommGetAsyncError(newcomm, &asyncErr);
    } while (asyncErr == ncclInProgress);
    
    // Step 5: Use the new communicator for collective operations
    // ...
    
    // Step 6: Existing ranks should destroy the parent communicator
    ncclCommDestroy(comm);
    
    // Step 7: When done, destroy the new communicator
    ncclCommDestroy(newcomm);

For non-blocking grow operations with error handling:

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;  // Non-blocking mode
    
    // Existing ranks
    ncclComm_t newcomm;
    ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, &config);
    
    // Poll for completion
    ncclResult_t asyncErr;
    do {
      res = ncclCommGetAsyncError(newcomm, &asyncErr);
      if (res != ncclSuccess) {
        // Handle error
        ncclCommAbort(newcomm);
        break;
      }
      // Handle timeouts or other events
    } while (asyncErr == ncclInProgress);
    
    if (asyncErr == ncclSuccess) {
      // Grow completed successfully
      // Destroy parent communicator
      ncclCommDestroy(comm);
    }

Important considerations:

1.  **Coordinator selection**: Any rank from the existing communicator can be the coordinator. The coordinator calls [`ncclCommGetUniqueId()`](../api/comms.html#c.ncclCommGetUniqueId "ncclCommGetUniqueId") to generate the grow identifier.

2.  **Rank assignment**: Existing ranks retain their original rank numbers in the new communicator. New ranks must be assigned ranks starting from the size of the parent communicator.

3.  **Out-of-band communication**: The grow identifier must be distributed from the coordinator to all new ranks using a communication mechanism outside of NCCL (e.g., MPI, sockets, shared files).

4.  **Parent communicator cleanup**: After the grow operation completes successfully, existing ranks should destroy the parent communicator using [`ncclCommDestroy()`](../api/comms.html#c.ncclCommDestroy "ncclCommDestroy") to free resources.

5.  **No outstanding operations**: There should not be any outstanding NCCL operations on the parent communicator when calling [`ncclCommGrow()`](../api/comms.html#c.ncclCommGrow "ncclCommGrow") to avoid potential deadlocks.

6.  **Configuration inheritance**: The new communicator inherits the configuration from the parent communicator for existing ranks. New ranks use the provided configuration or default settings.

Related links:

  - [`ncclCommGrow()`](../api/comms.html#c.ncclCommGrow "ncclCommGrow")

  - [`ncclCommGetUniqueId()`](../api/comms.html#c.ncclCommGetUniqueId "ncclCommGetUniqueId")

## Creating more communicators

The ncclCommSplit function can be used to create communicators based on an existing one. This allows to split an existing communicator into multiple sub-partitions, duplicate an existing communicator, or even create a single communicator with fewer ranks.

The ncclCommSplit function needs to be called by all ranks in the original communicator. If some ranks will not be part of any sub-group, they still need to call ncclCommSplit with color being NCCL\_SPLIT\_NOCOLOR.

Newly created communicators will inherit the parent communicator configuration (e.g. non-blocking). If the parent communicator operates in non-blocking mode, a ncclCommSplit operation may be stopped by calling ncclCommAbort on the parent communicator, then on any new communicator returned. This is because a hang could happen during operations on any of the two communicators.

The following code duplicates an existing communicator:

    int rank;
    ncclCommUserRank(comm, &rank);
    ncclCommSplit(comm, 0, rank, &newcomm, NULL);

This splits a communicator in two halves:

    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);
    ncclCommSplit(comm, rank/(nranks/2), rank%(nranks/2), &newcomm, NULL);

This creates a communicator with only the first 2 ranks:

    int rank;
    ncclCommUserRank(comm, &rank);
    ncclCommSplit(comm, rank<2 ? 0 : NCCL_SPLIT_NOCOLOR, rank, &newcomm, NULL);

Related links:

>   - [`ncclCommSplit()`](../api/comms.html#c.ncclCommSplit "ncclCommSplit")

## Using multiple NCCL communicators concurrently

Prior to NCCL 2.26, using multiple NCCL communicators per-device required serializing the order of all communication operations (via CUDA stream dependencies or synchronization) into a consistent total global order otherwise deadlocks could ensue. As of 2.26, NCCL introduces [NCCL\_LAUNCH\_ORDER\_IMPLICIT](../env.html#nccl-launch-order-implicit) which when enabled implicitly creates this order dynamically by following the order operations are issued from the host. Thus to remain deadlock free, users must ensure the order of host-side launches matches for all devices. This is most easily accomplished by using a determinstic order issued from a single host thread per-device. For example:

    ncclAllReduce(..., comm1, stream1); // all ranks do this first
    ncclAllReduce(..., comm2, stream2); // and this second

When NCCL is captured in a CUDA graph the same rules apply to both capture time and launch time. At capture time this means NCCL calls in the same graph must be captured in the same order:

    // both stream1 and stream2 are capturing in the same graph
    ncclAllReduce(..., comm1, stream1); // all ranks do this first
    ncclAllReduce(..., comm2, stream2); // and this second

And at graph launch time different graphs must be launched in a globally consistent order:

    cudaGraphLaunch(graph1, stream1); // all ranks do this first
    cudaGraphLaunch(graph2, stream2); // and this second

When running on CUDA 12.3 or later, the implicit ordering of the operations is created using CUDA launch completion events which permits parallel execution of the two communicator’s kernels.

## Finalizing a communicator

ncclCommFinalize will transition a communicator from the *ncclSuccess* state to the *ncclInProgress* state, start completing all operations in the background and synchronize with other ranks which may be using resources for their communications with other ranks. All uncompleted operations and network-related resources associated to a communicator will be flushed and freed with ncclCommFinalize. Once all NCCL operations are complete, the communicator will transition to the *ncclSuccess* state. Users can query that state with ncclCommGetAsyncError. If a communicator is marked as nonblocking, this operation is nonblocking; otherwise, it is blocking.

Related link: [`ncclCommFinalize()`](../api/comms.html#c.ncclCommFinalize "ncclCommFinalize")

## Destroying a communicator

Once a communicator has been finalized, the next step is to free all resources, including the communicator itself. Local resources associated to a communicator can be destroyed with ncclCommDestroy. If the state of a communicator is *ncclSuccess* when calling ncclCommDestroy, the call is guaranteed to be nonblocking; otherwise ncclCommDestroy might block. In all cases, ncclCommDestroy call will free the resources of the communicator and return, and the communicator should no longer be accessed after ncclCommDestroy returns.

Related link: [`ncclCommDestroy()`](../api/comms.html#c.ncclCommDestroy "ncclCommDestroy")

# Error handling and communicator abort

All NCCL calls return a NCCL error code which is sumarized in the table below. If a NCCL call returns an error code different from ncclSuccess and ncclInternalError, and if NCCL\_DEBUG is set to WARN, NCCL will print a human-readable message explaining what happened. If NCCL\_DEBUG is set to INFO, NCCL will also print the call stack which led to the error. This message is intended to help the user fix the problem.

The table below summarizes how different errors should be understood and handled. Each case is explained in details in the following sections.

| Error                  | Description                               | Resolution                                      | Error handling         | Group behavior |
| ---------------------- | ----------------------------------------- | ----------------------------------------------- | ---------------------- | -------------- |
| ncclSuccess            | No error                                  | None                                            | None                   | None           |
| ncclUnhandledCudaError | Error during a CUDA call (1)              | CUDA configuration / usage (1)                  | Communicator abort (5) | Global (6)     |
| ncclSystemError        | Error during a system call (1)            | System configuration / usage (1)                | Communicator abort (5) | Global (6)     |
| ncclInternalError      | Error inside NCCL (2)                     | Fix in NCCL (2)                                 | Communicator abort (5) | Global (6)     |
| ncclInvalidArgument    | An argument to a NCCL call is invalid (3) | Fix in the application (3)                      | None (3)               | Individual (3) |
| ncclInvalidUsage       | The usage of NCCL calls is invalid (4)    | Fix in the application (4)                      | Communicator abort (5) | Global (6)     |
| ncclInProgress         | The NCCL call is still in progress        | Poll for completion using ncclCommGetAsyncError | None                   | None           |

NCCL Errors

(1) ncclUnhandledCudaError and ncclSystemError indicate that a call NCCL made to an external component failed, which caused the NCCL operation to fail. The error message should explain which component the user should look at and try to fix, potentially with the help of the administrators of the system.

(2) ncclInternalError denotes a NCCL bug. It might not report a message with NCCL\_DEBUG=WARN since it requires a fix in the NCCL source code. NCCL\_DEBUG=INFO will print the back trace which led to the error.

(3) ncclInvalidArgument indicates an argument value is incorrect, like a NULL pointer or an out-of-bounds value. When this error is returned, the NCCL call had no effect. The group state remains unchanged, the communicator is still functioning normally. The application can call ncclCommAbort or continue as if the call did not happen. This error will be returned immediately for a call happening within a group and applies to that specific NCCL call. It will not be returned by ncclGroupEnd since ncclGroupEnd takes no argument.

(4) ncclInvalidUsage is returned when a dynamic condition causes a failure, which denotes an incorrect usage of the NCCL API.

(5) These errors are fatal for the communicator. To recover, the application needs to call ncclCommAbort on the communicator and re-create it.

(6) Dynamic errors for operations within a group are always reported by ncclGroupEnd and apply to all operations within the group, which may or may not have completed. The application must call ncclCommAbort on all communicators within the group.

## Asynchronous errors and error handling

Some communication errors, and in particular network errors, are reported through the ncclCommGetAsyncError function. Operations experiencing an asynchronous error will usually not progress and never complete. When an asynchronous error happens, the operation should be aborted and the communicator destroyed using ncclCommAbort. When waiting for NCCL operations to complete, applications should call ncclCommGetAsyncError and destroy the communicator when an error happens.

The following code shows how to wait on NCCL operations and poll for asynchronous errors, instead of using cudaStreamSynchronize.

    int ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm) {
      cudaError_t cudaErr;
      ncclResult_t ncclErr, ncclAsyncErr;
      while (1) {
       cudaErr = cudaStreamQuery(stream);
       if (cudaErr == cudaSuccess)
         return 0;
    
       if (cudaErr != cudaErrorNotReady) {
         printf("CUDA Error : cudaStreamQuery returned %d\n", cudaErr);
         return 1;
       }
    
       ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
       if (ncclErr != ncclSuccess) {
         printf("NCCL Error : ncclCommGetAsyncError returned %d\n", ncclErr);
         return 1;
       }
    
       if (ncclAsyncErr != ncclSuccess) {
         // An asynchronous error happened. Stop the operation and destroy
         // the communicator
         ncclErr = ncclCommAbort(comm);
         if (ncclErr != ncclSuccess)
           printf("NCCL Error : ncclCommDestroy returned %d\n", ncclErr);
         // Caller may abort or try to create a new communicator.
         return 2;
       }
    
       // We might want to let other threads (including NCCL threads) use the CPU.
       sched_yield();
      }
    }

Related links:

>   - [`ncclCommGetAsyncError()`](../api/comms.html#c.ncclCommGetAsyncError "ncclCommGetAsyncError")
> 
>   - [`ncclCommAbort()`](../api/comms.html#c.ncclCommAbort "ncclCommAbort")

# Fault Tolerance

NCCL provides a set of features to allow applications to recover from fatal errors such as a network failure, a node failure, or a process failure. When such an error happens, the application should be able to call *ncclCommAbort* on the communicator to free all resources, then create a new communicator to continue.

For more advanced recovery, the *ncclCommShrink* function with *NCCL\_SHRINK\_ABORT* can be used to create a new communicator by removing failed ranks from the existing communicator while safely handling in-progress operations. This approach is particularly useful in distributed environments where only some ranks have failed.

In order to abort NCCL communicators safely, NCCL requires applications to set communicators as nonblocking and make sure no thread is calling any NCCL operations while calling *ncclCommAbort*. After nonblocking is set, all NCCL calls (except *ncclCommDestroy/Abort*) become nonblocking so that *ncclCommAbort* can be called at any point, during initialization, communication or finalizing the communicator. If NCCL communicators are set blocking, the thread can possibly get stuck inside NCCL calls due to network errors; in this case, NCCL communicators might hang forever.

To correctly abort, when any rank in a communicator fails (e.g., due to a segmentation fault), all other ranks need to call *ncclCommAbort* to abort their own NCCL communicator. Users can implement methods to decide when and whether to abort the communicators and restart the NCCL operation. Here is an example showing how to initialize and split a communicator in a non-blocking manner, allowing for an abort at any point:

    bool globalFlag;
    bool abortFlag = false;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    /* set communicator as nonblocking */
    config.blocking = 0;
    CHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
    do {
      CHECK(ncclCommGetAsyncError(comm, &state));
    } while(state == ncclInProgress && checkTimeout() != true);
    
    if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;
    
    /* sync abortFlag among all healthy ranks. */
    reportErrorGlobally(abortFlag, &globalFlag);
    
    if (globalFlag) {
      /* time is out or initialization failed: every rank needs to abort and restart. */
      ncclCommAbort(comm);
      /* restart NCCL; this is a user implemented function, it might include
       * resource cleanup and ncclCommInitRankConfig() to create new communicators. */
      restartNCCL(&comm);
    }
    
    /* nonblocking communicator split. */
    CHECK(ncclCommSplit(comm, color, key, &childComm, &config));
    do {
      CHECK(ncclCommGetAsyncError(comm, &state));
    } while(state == ncclInProgress && checkTimeout() != true);
    
    if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;
    
    /* sync abortFlag among all healthy ranks. */
    reportErrorGlobally(abortFlag, &globalFlag);
    
    if (globalFlag) {
      ncclCommAbort(comm);
      /* if chilComm is not NCCL_COMM_NULL, user should abort child communicator
       * here as well for resource reclamation. */
      if (childComm != NCCL_COMM_NULL) ncclCommAbort(childComm);
      restartNCCL(&comm);
    }
    /* application workload */

The *checkTimeout* function needs to be provided by users to determine what is the longest time the application should wait for NCCL initialization; likewise, users can apply other methods to detect errors besides a timeout function. Similar methods can be applied to NCCL finalization as well.

# Quality of Service

Applications which overlap communication may benefit from network Quality of Service (QoS) features. NCCL allows an application to assign a traffic class (TC) to each communicator to identify the communication requirements of the communicator. All network operations on a communicator will use the assigned TC.

The meaning of TC is specific to the network plugin in use by the communicator (e.g. IB networks use service level, RoCE networks use type of service). TCs are defined by the system configuration. Applications must understand the TCs available on a system and their relative behavior in order to use them effectively.

TC is specified during communicator creation using [ncclConfig\_t](../api/types.html#ncclconfig).

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.trafficClass = 1;
    CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

Infiniband networks support QoS through the use of Service Levels (SL). Each IB SL is mapped to Virtual Lane (VL), which defines the relative priority of traffic. SL behavior is defined within the subnet manager, such as OpenSM. Refer to subnet manager documentation for more detail. An example configuration is shown below.

    ...
    qos_max_vls 2
    qos_high_limit 255
    qos_vlarb_high 1:4
    qos_vlarb_low 0:1,1:4
    qos_sl2vl 0,1
    
    max_op_vls 2
    ....

The example defines one low priority and one high priority VL which are mapped to SL 0 and 1, respectively. The high priority SL will be given a larger share of network bandwidth at each port. In NCCL, the communicator’s traffic class corresponds to the SL on IB networks. Using this configuration, applications can assign TC 0 to low-priority communicators and TC 1 to high-priority ones.

On RoCE networks, the NCCL communicator trafficClass is interpreted as an IP Type of Service (ToS). Refer to network management tools to understand how to configure QoS for a given workload.

---

# Collective Operations

Collective operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form a complete collective operation. Failure to do so will result in undefined behavior, including hangs, crashes, or data corruption.

## AllReduce

The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank.

In a *sum* allreduce operation between *k* ranks, each rank will provide an array in of N values, and receive identical results in array out of N values, where out\[i\] = in0\[i\]+in1\[i\]+…+in(k-1)\[i\].

[Figure: All-Reduce operation: each rank receives the reduction of input values across ranks.]

Related links: [`ncclAllReduce()`](../api/colls.html#c.ncclAllReduce "ncclAllReduce").

## Broadcast

The Broadcast operation copies an N-element buffer from the root rank to all the ranks.

[Figure: Broadcast operation: all ranks receive data from a “root” rank.]

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: [`ncclBroadcast()`](../api/colls.html#c.ncclBroadcast "ncclBroadcast").

## Reduce

The Reduce operation performs the same operation as AllReduce, but stores the result only in the receive buffer of a specified root rank.

[Figure: Reduce operation: one rank receives the reduction of input values across ranks.]

Important note: The root argument is one of the ranks (not a device number), and is therefore impacted by a different rank to device mapping.

Note: A Reduce, followed by a Broadcast, is equivalent to the AllReduce operation.

Related links: [`ncclReduce()`](../api/colls.html#c.ncclReduce "ncclReduce").

## AllGather

The AllGather operation gathers N values from k ranks into an output buffer of size k\*N, and distributes that result to all ranks.

The output is ordered by the rank index. The AllGather operation is therefore impacted by a different rank to device mapping.

[Figure: AllGather operation: each rank receives the aggregation of data from all ranks in the order of the ranks.]

Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

Related links: [`ncclAllGather()`](../api/colls.html#c.ncclAllGather "ncclAllGather").

## ReduceScatter

The ReduceScatter operation performs the same operation as Reduce, except that the result is scattered in equal-sized blocks between ranks, each rank getting a chunk of data based on its rank index.

The ReduceScatter operation is impacted by a different rank to device mapping since the ranks determine the data layout.

[Figure: Reduce-Scatter operation: input values are reduced across ranks, with each rank receiving a subpart of the result.]

Related links: [`ncclReduceScatter()`](../api/colls.html#c.ncclReduceScatter "ncclReduceScatter")

## AlltoAll

In an AlltoAll operation between k ranks, each rank provides an input buffer of size k\*N values, where the j-th chunk of N values is sent to destination rank j. Each rank receives an output buffer of size k\*N values, where the i-th chunk of N values comes from source rank i.

[Figure: AlltoAll operation: exchanges data between all ranks, where each rank sends different data to every other rank and receives different data from every other rank.]

Related links: [`ncclAlltoAll()`](../api/colls.html#c.ncclAlltoAll "ncclAlltoAll").

## Gather

The Gather operation gathers N values from k ranks into an output buffer on the root rank of size k\*N.

[Figure: Gather operation: root rank receives data from all ranks.]

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: [`ncclGather()`](../api/colls.html#c.ncclGather "ncclGather").

## Scatter

The Scatter operation distributes a total of N\*k values from the root rank to k ranks, each rank receiving N values.

[Figure: Scatter operation: root rank distributes data to all ranks.]

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: [`ncclScatter()`](../api/colls.html#c.ncclScatter "ncclScatter").

---

# Data Pointers

In general NCCL will accept any CUDA pointers that are accessible from the CUDA device associated to the communicator object. This includes:

>   - device memory local to the CUDA device
> 
>   - host memory registered using CUDA SDK APIs cudaHostRegister or cudaGetDevicePointer
> 
>   - managed and unified memory

The only exception is device memory located on another device but accessible from the current device using peer access. NCCL will return an error in that case to avoid programming errors (only when NCCL\_CHECK\_POINTERS=1 since 2.2.12).

---

# CUDA Stream Semantics

NCCL calls are associated to a stream which is passed as the last argument of the collective communication function. The NCCL call returns when the operation has been effectively enqueued to the given stream, or returns an error. The collective operation is then executed asynchronously on the CUDA device. The operation status can be queried using standard CUDA semantics, for example, calling cudaStreamSynchronize or using CUDA events.

## Mixing Multiple Streams within the same ncclGroupStart/End() group

NCCL allows for using multiple streams within a group call. This will enforce a stream dependency of all streams before the NCCL kernel starts and block all streams until the NCCL kernel completes.

It will behave as if the NCCL group operation was posted on every stream, but given it is a single operation, it will cause a global synchronization point between the streams.

---

# Group Calls

Group functions (ncclGroupStart/ncclGroupEnd) can be used to merge multiple calls into one. This is needed for three purposes: managing multiple GPUs from one thread (to avoid deadlocks), aggregating communication operations to improve performance, or merging multiple send/receive point-to-point operations (see [Point-to-point communication](p2p.html#point-to-point) section). All three usages can be combined together, with one exception : calls to [`ncclCommInitRank()`](../api/comms.html#c.ncclCommInitRank "ncclCommInitRank") cannot be merged with others.

## Management Of Multiple GPUs From One Thread

When a single thread is managing multiple devices, group semantics must be used. This is because every NCCL call may have to block, waiting for other threads/ranks to arrive, before effectively posting the NCCL operation on the given stream. Hence, a simple loop on multiple devices like shown below could block on the first call waiting for the other ones:

    for (int i=0; i<nLocalDevs; i++) {
      ncclAllReduce(..., comm[i], stream[i]);
    }

To define that these calls are part of the same collective operation, ncclGroupStart and ncclGroupEnd should be used:

    ncclGroupStart();
    for (int i=0; i<nLocalDevs; i++) {
      ncclAllReduce(..., comm[i], stream[i]);
    }
    ncclGroupEnd();

This will tell NCCL to treat all calls between ncclGroupStart and ncclGroupEnd as a single call to many devices.

Caution: When called inside a group, stream operations (like ncclAllReduce) can return without having enqueued the operation on the stream. Stream operations like cudaStreamSynchronize can therefore be called only after ncclGroupEnd returns.

Group calls must also be used to create a communicator when one thread manages more than one device:

    ncclGroupStart();
    for (int i=0; i<nLocalDevs; i++) {
      cudaSetDevice(device[i]);
      ncclCommInitRank(comms+i, nranks, commId, rank[i]);
    }
    ncclGroupEnd();

Note: Contrary to NCCL 1.x, there is no need to set the CUDA device before every NCCL communication call within a group, but it is still needed when calling ncclCommInitRank within a group.

Related links:

  - [`ncclGroupStart()`](../api/group.html#c.ncclGroupStart "ncclGroupStart")

  - [`ncclGroupEnd()`](../api/group.html#c.ncclGroupEnd "ncclGroupEnd")

## Aggregated Operations (2.2 and later)

The group semantics can also be used to have multiple collective operations performed within a single NCCL launch. This is useful for reducing the launch overhead, in other words, latency, as it only occurs once for multiple operations. Init functions cannot be aggregated with other init functions, nor with communication functions.

Aggregation of collective operations can be done simply by having multiple calls to NCCL within a ncclGroupStart / ncclGroupEnd section.

In the following example, we launch one broadcast and two allReduce operations together as a single NCCL launch.

    ncclGroupStart();
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm, stream);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm, stream);
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm, stream);
    ncclGroupEnd();

It is permitted to combine aggregation with multi-GPU launch and use different communicators in a group launch as shown in the Management Of Multiple GPUs From One Thread topic. When combining multi-GPU launch and aggregation, ncclGroupStart and ncclGroupEnd can be either used once or at each level. The following example groups the allReduce operations from different layers and on multiple CUDA devices :

    ncclGroupStart();
    for (int i=0; i<nlayers; i++) {
      ncclGroupStart();
      for (int g=0; g<ngpus; g++) {
        ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], comms[g], streams[g]);
      }
      ncclGroupEnd();
    }
    ncclGroupEnd();

Note: The NCCL operation will only be started as a whole during the last call to ncclGroupEnd. The ncclGroupStart and ncclGroupEnd calls within the for loop are not necessary and do nothing.

Related links:

  - [`ncclGroupStart()`](../api/group.html#c.ncclGroupStart "ncclGroupStart")

  - [`ncclGroupEnd()`](../api/group.html#c.ncclGroupEnd "ncclGroupEnd")

## Group Operation Ordering Semantics

Although NCCL group allows different operations to be issued in one shot, users still need to guarantee the same issuing order of the operations among different GPUs no matter whether the operations are issued to the same or different communicators.

For example, the following code provides the correct order of the operations. In this example, *comm0* and *comm1* are duplicated independent communicators that include rank 0 and 1.

    RANK0/GPU0/Process0:
    ncclGroupStart();
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
    ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
    ncclGroupEnd();
    
    RANK1/GPU1/Process1:
    ncclGroupStart();
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
    ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
    ncclGroupEnd();

However, changing the order of the any operations will lead to incorrect results or hang as shown in the following 2 examples:

    RANK0/GPU0/Process0:
    ncclGroupStart();
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream); // WRONG: reversed order
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream); // WRONG: reversed order
    ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
    ncclGroupEnd();
    
    RANK1/GPU1/Process1:
    ncclGroupStart();
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream); // WRONG: reversed order
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream); // WRONG: reversed order
    ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream);
    ncclGroupEnd();

    RANK0/GPU0/Process0:
    ncclGroupStart();
    ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream); // WRONG: reversed order
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
    ncclGroupEnd();
    
    RANK1/GPU1/Process1:
    ncclGroupStart();
    ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm0, stream);
    ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm0, stream);
    ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm0, stream);
    ncclAllReduce(sendbuff4, recvbuff4, count4, datatype, comm1, stream); // WRONG: reversed order
    ncclGroupEnd();

## Nonblocking Group Operation

If a communicator is marked as nonblocking through ncclCommInitRankConfig, the group functions become asynchronous correspondingly. In this case, if users issue multiple NCCL operations in one group, returning from ncclGroupEnd() might not mean the NCCL communication kernels have been issued to CUDA streams. If ncclGroupEnd() returns ncclSuccess, it means NCCL kernels have been issued to streams; if it returns ncclInProgress, it means NCCL kernels are being issued to streams in the background. It is users’ responsibility to make sure the state of the communicator changes into ncclSuccess before calling related CUDA calls (e.g. cudaStreamSynchronize):

    ncclGroupStart();
      for (int g=0; g<ngpus; g++) {
        ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], comms[g], streams[g]);
      }
    ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
       for (int g=0; g<ngpus; g++) {
         do {
           ncclCommGetAsyncError(comms[g], &state);
         } while (state == ncclInProgress);
       }
    } else if (ret == ncclSuccess) {
       /* Successfully issued */
       printf("NCCL kernel issue succeeded\n");
    } else {
       /* Errors happen */
       reportErrorAndRestart();
    }
    
    for (int g=0; g<ngpus; g++) {
      cudaStreamSynchronize(streams[g]);
    }

Related links:

  - [`ncclCommInitRankConfig()`](../api/comms.html#c.ncclCommInitRankConfig "ncclCommInitRankConfig")

  - [`ncclCommGetAsyncError()`](../api/comms.html#c.ncclCommGetAsyncError "ncclCommGetAsyncError")

---

# Point-to-point communication

## Two-sided communication

(Since NCCL 2.7) Point-to-point communication can be used to express any communication pattern between ranks. Any point-to-point communication needs two NCCL calls : a call to [`ncclSend()`](../api/p2p.html#c.ncclSend "ncclSend") on one rank and a corresponding [`ncclRecv()`](../api/p2p.html#c.ncclRecv "ncclRecv") on the other rank, with the same count and data type.

Multiple calls to [`ncclSend()`](../api/p2p.html#c.ncclSend "ncclSend") and [`ncclRecv()`](../api/p2p.html#c.ncclRecv "ncclRecv") targeting different peers can be fused together with [`ncclGroupStart()`](../api/group.html#c.ncclGroupStart "ncclGroupStart") and [`ncclGroupEnd()`](../api/group.html#c.ncclGroupEnd "ncclGroupEnd") to form more complex communication patterns such as one-to-all (scatter), all-to-one (gather), all-to-all or communication with neighbors in an N-dimensional space.

Point-to-point calls within a group will be blocking until that group of calls completes, but calls within a group can be seen as progressing independently, hence should never block each other. It is therefore important to merge calls that need to progress concurrently to avoid deadlocks. The only exception is point-to-point calls within a group targeting the *same* peer, which are executed in order.

Below are a few examples of classic point-to-point communication patterns used by parallel applications. NCCL semantics allow for all variants with different sizes, datatypes, and buffers, per rank.

### Sendrecv

In MPI terms, a sendrecv operation is when two ranks exchange data, both sending and receiving at the same time. This can be done by merging both ncclSend and ncclRecv calls into one :

    ncclGroupStart();
    ncclSend(sendbuff, sendcount, sendtype, peer, comm, stream);
    ncclRecv(recvbuff, recvcount, recvtype, peer, comm, stream);
    ncclGroupEnd();

### One-to-all (scatter)

A one-to-all operation from a `root` rank can be expressed by merging all send and receive operations in a group :

    ncclGroupStart();
    if (rank == root) {
      for (int r=0; r<nranks; r++)
        ncclSend(sendbuff[r], size, type, r, comm, stream);
    }
    ncclRecv(recvbuff, size, type, root, comm, stream);
    ncclGroupEnd();

### All-to-one (gather)

Similarly, an all-to-one operations to a `root` rank would be implemented this way :

    ncclGroupStart();
    if (rank == root) {
      for (int r=0; r<nranks; r++)
        ncclRecv(recvbuff[r], size, type, r, comm, stream);
    }
    ncclSend(sendbuff, size, type, root, comm, stream);
    ncclGroupEnd();

### All-to-all

An all-to-all operation would be a merged loop of send/recv operations to/from all peers :

    ncclGroupStart();
    for (int r=0; r<nranks; r++) {
      ncclSend(sendbuff[r], sendcount, sendtype, r, comm, stream);
      ncclRecv(recvbuff[r], recvcount, recvtype, r, comm, stream);
    }
    ncclGroupEnd();

### Neighbor exchange

Finally, exchanging data with neighbors in an N-dimensions space could be done with :

    ncclGroupStart();
    for (int d=0; d<ndims; d++) {
      ncclSend(sendbuff[d], sendcount, sendtype, next[d], comm, stream);
      ncclRecv(recvbuff[d], recvcount, recvtype, prev[d], comm, stream);
    }
    ncclGroupEnd();

## One-sided communication

(Since NCCL 2.29) One-sided communication enables a rank to write data to remote memory using [`ncclPutSignal()`](../api/p2p.html#c.ncclPutSignal "ncclPutSignal") without requiring the target rank to issue a matching operation. The target memory must be pre-registered using [`ncclCommWindowRegister()`](../api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister"). Point-to-point synchronization can be achieved by having the target rank call [`ncclWaitSignal()`](../api/p2p.html#c.ncclWaitSignal "ncclWaitSignal") to wait for signals.

Multiple [`ncclPutSignal()`](../api/p2p.html#c.ncclPutSignal "ncclPutSignal") calls can be grouped using [`ncclGroupStart()`](../api/group.html#c.ncclGroupStart "ncclGroupStart") and [`ncclGroupEnd()`](../api/group.html#c.ncclGroupEnd "ncclGroupEnd"). Operations to different peers or contexts within a group may execute concurrently and complete in any order. The completion of [`ncclGroupEnd()`](../api/group.html#c.ncclGroupEnd "ncclGroupEnd") guarantees that all operations in the group have achieved completion. Operations to the same peer and context are executed in order: both data delivery and signal updates on the remote peer follow the program order.

Below are a few examples of classic one-sided communication patterns used by parallel applications.

### PutSignal and WaitSignal

A ping-pong pattern using [`ncclPutSignal()`](../api/p2p.html#c.ncclPutSignal "ncclPutSignal") and [`ncclWaitSignal()`](../api/p2p.html#c.ncclWaitSignal "ncclWaitSignal"). This example shows the full setup including memory allocation and window registration:

    // Allocate symmetric memory for RMA operations
    void *sendbuff, *recvbuff;
    NCCLCHECK(ncclMemAlloc((void**)&sendbuff, size));
    NCCLCHECK(ncclMemAlloc((void**)&recvbuff, size));
    
    // Register buffers as symmetric windows
    ncclWindow_t sendWindow, recvWindow;
    NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, size, &sendWindow, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, size, &recvWindow, NCCL_WIN_COLL_SYMMETRIC));
    
    int peer = (rank == 0) ? 1 : 0;
    ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = peer, .sigIdx = 0, .ctx = ctx};
    
    if (rank == 0) {
      // Rank 0: wait then put
      NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
      NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                        0, 0, 0, comm, stream));
    } else {
      // Rank 1: put then wait
      NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                        0, 0, 0, comm, stream));
      NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
    }
    
    CUDACHECK(cudaStreamSynchronize(stream));
    
    // Cleanup
    NCCLCHECK(ncclCommWindowDeregister(comm, sendWindow));
    NCCLCHECK(ncclCommWindowDeregister(comm, recvWindow));
    NCCLCHECK(ncclMemFree(sendbuff));
    NCCLCHECK(ncclMemFree(recvbuff));

### Barrier

A barrier pattern using [`ncclSignal()`](../api/p2p.html#c.ncclSignal "ncclSignal") and [`ncclWaitSignal()`](../api/p2p.html#c.ncclWaitSignal "ncclWaitSignal"). Each rank signals to all other ranks and waits for signals from all ranks:

    ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
    for (int r = 0; r < nranks; r++) {
      waitDescs[r].opCnt = 1;
      waitDescs[r].peer = r;
      waitDescs[r].sigIdx = 0;
      waitDescs[r].ctx = 0;
    }
    
    ncclGroupStart();
    for (int r = 0; r < nranks; r++) {
      ncclSignal(r, 0, 0, 0, comm, stream);
    }
    ncclGroupEnd();
    
    ncclWaitSignal(nranks, waitDescs, comm, stream);

### All-to-all

An all-to-all operation using [`ncclPutSignal()`](../api/p2p.html#c.ncclPutSignal "ncclPutSignal"). Each rank sends data to all other ranks and waits for signals from all ranks. User needs to register the memory window for each peer using [`ncclCommWindowRegister()`](../api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister") in advance. User needs to guarantee the buffers are ready before calling [`ncclPutSignal()`](../api/p2p.html#c.ncclPutSignal "ncclPutSignal"). This could be done with the barrier shown above.

    size_t offset[nranks];
    ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
    for (int r = 0; r < nranks; r++) {
      offset[r] = r * count * wordSize(datatype);
      waitDescs[r].opCnt = 1;
      waitDescs[r].peer = r;
      waitDescs[r].sigIdx = 0;
      waitDescs[r].ctx = 0;
    }
    
    ncclGroupStart();
    for (int r = 0; r < nranks; r++) {
      ncclPutSignal(sendbuff[r], count, datatype, r, window, offset[r],
              0, 0, 0, comm, stream);
    }
    ncclGroupEnd();
    
    ncclWaitSignal(nranks, waitDescs, comm, stream);

---

# Thread Safety

NCCL primitives are generally not thread-safe, however, they are reentrant. Under multi-thread environment, it is not allowed to issue NCCL operations to a single communicator in parallel with multiple threads; it is not safe to issue NCCL operations in parallel to independent communicators located on the same device with multiple threads (see [Using multiple NCCL communicators concurrently](communicators.html#multi-thread-concurrent-usage)). If the child communicator shares the resources with the parent communicator (i.e., [ncclConfig\_t](../api/types.html#ncclconfig) by splitShare), it is not allowed to issue NCCL operations to the child and parent communicators in parallel.

It is safe to operate a communicator from multiple threads as long as users can guarantee only one thread operates the communicator at a time. However, for any grouped NCCL operations, users need to ensure only one thread issues the all operations in the group.

For example, the following code provides a simple thread-safe example where threads are executed in sequence and only one thread is accessing the communicator at a time.

    Thread 0:
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      cudaSetDevice(0);
      ncclCommInitRankConfig(&comm, nranks, id, rank, &config);
      ncclGroupStart();
      ncclAllReduce(sendbuff0, recvbuff0, count0, datatype, redOp, comm, stream);
      ncclAllReduce(sendbuff1, recvbuff1, count1, datatype, redOp, comm, stream);
      ncclGroupEnd();
      thread_exit();
    Thread 1:
      ncclResult_t state = ncclSuccess;
      // wait for previous issued allreduce ops by Thread 0
      do {
        ncclCommGetAsyncError(comm, &state);
      } while (state == ncclInProgress);
      assert(state == ncclSuccess);
      ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, redOp, comm, stream);
      do {
        ncclCommGetAsyncError(comm, &state);
      } while (state == ncclInProgress);
      assert(state == ncclSuccess);

It is also valid to issue grouped NCCL operations from one thread and poll the status of each NCCL communicator with one thread as shown in the following code.

    Thread 0:
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      ncclGroupStart();
      for (int i = 0; i < nGpus; i++) {
        cudaSetDevice(i);
        ncclCommInitRankConfig(&comms[i], nranks, id, ranks[i], &config);
      }
      ncclGroupEnd();
    Thread 0/1/2/3:
      ncclResult_t state = ncclSuccess;
      // wait for previous issued init ops by Thread 0
      do {
        ncclCommGetAsyncError(comms[thread_id], &state);
      } while (state == ncclInProgress);
      assert(state == ncclSuccess);
      ncclAllReduce(sendbuff, recvbuff, count, datatype, redOp, comms[thread_id], stream);
      do {
        ncclCommGetAsyncError(comms[thread_id], &state);
      } while (state == ncclInProgress);
      assert(state == ncclSuccess);

---

# In-place Operations

Contrary to MPI, NCCL does not define a special “in-place” value to replace pointers. Instead, NCCL optimizes the case where the provided pointers are effectively “in place”.

For ncclBroadcast, ncclReduce and ncclAllreduce functions, this means that passing `sendBuff == recvBuff` will perform in place operations, storing final results at the same place as initial data was read from.

For ncclReduceScatter and ncclAllGather, in place operations are done when the per-rank pointer is located at the rank offset of the global buffer. More precisely, these calls are considered in place :

    ncclReduceScatter(data, data+rank*recvcount, recvcount, datatype, op, comm, stream);
    ncclAllGather(data+rank*sendcount, data, sendcount, datatype, op, comm, stream);

---

# Using NCCL with CUDA Graphs

Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs.

CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. More details about CUDA Graphs can be found in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).

NCCL’s collective, P2P and group operations all support CUDA Graph captures. This support requires a minimum CUDA version of 11.3.

Whether an operation launch is graph-captured is considered a collective property of that operation and therefore must be uniform over all ranks participating in the launch (for collectives this is all ranks in the communicator, for peer-to-peer this is both the sender and receiver). The launch of a graph (via cudaGraphLaunch, etc.) containing a captured NCCL operation is considered collective for the same set of ranks that were present in the capture, and each of those ranks must be using the graph derived from that collective capture.

The following sample code shows how to capture computational kernels and NCCL operations in a CUDA Graph:

    cudaGraph_t graph;
    cudaStreamBeginCapture(stream);
    kernel_A<<< ..., stream >>>(...);
    kernel_B<<< ..., stream >>>(...);
    ncclAllreduce(..., stream);
    kernel_C<<< ..., stream >>>(...);
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);

Starting with NCCL 2.11, when NCCL communication is captured and the CollNet algorithm is used, NCCL allows for further performance improvement via user buffer registration. For details, please see the environment variable [NCCL\_GRAPH\_REGISTER](../env.html#nccl-graph-register).

Having multiple outstanding NCCL operations that are any combination of graph-captured or non-captured is supported. There is a caveat that the mechanism NCCL uses internally to accomplish this has been seen to cause CUDA to deadlock when the graphs of multiple communicators are cudaGraphLaunch()’d from the same thread. To disable this mechansim see the environment variable [NCCL\_GRAPH\_MIXING\_SUPPORT](../env.html#nccl-graph-mixing-support).

---

# User Buffer Registration

User Buffer Registration is a feature that allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy (zero-copy). It can accelerate collectives and greatly reduce the resource usage (e.g. \#channel usage). NCCL provides two ways to register user buffers; one is *CUDA Graph* registration, and the other is *Local* registration. NCCL requires that for all NCCL communication function calls (e.g., allreduce, sendrecv, and so on), if any rank in a communicator passes registered buffers to a NCCL communication function, all other ranks in the same communicator must pass their registered buffers; otherwise, mixing registered and non-registered buffers can result in undefined behavior; in addition, source and destination buffers must be registered in order to enable user buffer registration for NCCL operations.

## NVLink Sharp Buffer Registration

Since 2.19.x, NCCL supports user buffer registration for NVLink Sharp (NVLS); any NCCL collectives (e.g., allreduce) that support NVLS algorithm can utilize this feature.

To enable the *CUDA Graph* based buffer registration for NVLS, users have to comply with several requirements:

>   - The buffer is allocated through [`ncclMemAlloc()`](../api/comms.html#c.ncclMemAlloc "ncclMemAlloc") or a qualified allocator (see [Memory Allocator](#mem-allocator)).
> 
>   - The NCCL operation is launched on a stream captured by a CUDA graph for each rank.
> 
>   - Offset to the head address of the buffer is the same in collectives for each rank.

Registered buffers will be deregistered when the CUDA graph is destroyed. Here is a CUDA graph based buffer registration example:

    void* sendbuff;
    void* recvbuff;
    size_t count = 1 << 25;
    CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
    CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));
    
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
    // Same offset to the sendbuff and recvbuff head address for each rank
    CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
    CHECK(cudaStreamEndCapture(stream, &graph));
    
    cudaGraphExec_t instance;
    CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    CHECK(cudaGraphLaunch(instance, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaGraphExecDestroy(instance));
    CHECK(cudaGraphDestroy(graph));
    
    CHECK(ncclMemFree(sendbuff));
    CHECK(ncclMemFree(recvbuff));

On the other hand, to enable the *Local* based buffer registration for NVLS, users have to comply with the following requirements:

>   - The buffer is allocated through [`ncclMemAlloc()`](../api/comms.html#c.ncclMemAlloc "ncclMemAlloc") or a qualified allocator (see [Memory Allocator](#mem-allocator)).
> 
>   - Register buffer with [`ncclCommRegister()`](../api/comms.html#c.ncclCommRegister "ncclCommRegister") before calling collectives for each rank.
> 
>   - Call NCCL collectives as usual but similarly keep the offset to the head address of the buffer the same for each rank.

Registered buffers will be deregistered when users explicitly call [`ncclCommDeregister()`](../api/comms.html#c.ncclCommDeregister "ncclCommDeregister"). Here is a local based buffer registration example:

    void* sendbuff;
    void* recvbuff;
    size_t count = 1 << 25;
    void* sendRegHandle;
    void* recvRegHandle;
    CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
    CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));
    
    CHECK(ncclCommRegister(comm, sendbuff, count * sizeof(float), &sendRegHandle));
    CHECK(ncclCommRegister(comm, recvbuff, count * sizeof(float), &recvRegHandle));
    
    CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
    CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
    CHECK(cudaStreamSynchronize(stream));
    
    CHECK(ncclCommDeregister(comm, sendRegHandle));
    CHECK(ncclCommDeregister(comm, recvRegHandle));
    
    CHECK(ncclMemFree(sendbuff));
    CHECK(ncclMemFree(recvbuff));

For local based registration, users can register the buffer once at the beginning of the program and reuse the buffer multiple times to utilize registration benefits.

To save the memory, it is also valid to allocate a large chunk of buffer and register it once. sendbuff and recvbuff can be further allocated through the big chunk for zero-copy NCCL operations as long as sendbuff and recvbuff satisfy the offset requirements. The following example shows a use case:

    void* buffer;
    void* handle;
    void* sendbuff;
    void* recvbuff;
    size_t size = 1 << 29;
    
    CHECK(ncclMemAlloc(&buffer, size));
    CHECK(ncclCommRegister(comm, buffer, size, &handle));
    
    // assign buffer chunk to sendbuff and recvbuff
    sendbuff = buffer;
    recvbuff = (void*)((uint8_t*)buffer + (1 << 20));
    
    CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
    CHECK(ncclAllGather(sendbuff, recvbuff, 1024, ncclInt8, comm, stream));
    CHECK(cudaStreamSynchronize(stream));
    
    CHECK(ncclCommDeregister(comm, handle));
    
    CHECK(ncclMemFree(sendbuff));

## IB Sharp Buffer Registration

NCCL 2.21.x supports IB Sharp buffer registration, any NCCL collectives that support IB Sharp algorithm can benefit from the feature such as allreduce, reducescatter, and allgather. Currently, NCCL only supports IB Sharp buffer registration for the communicators which contain 1 rank per node, and the registration can reduce the number of NCCL SM usage down to 1.

To enable IB Sharp buffer registration by CUDA graph:

>   - Allocate send and recv buffer with any CUDA allcator (e.g., cudaMalloc/ncclMemAlloc)
> 
>   - Launch NCCL collectives with CUDA graph

To enable IB Sharp buffer registration by local registration:

>   - Allocate send and recv buffer with any CUDA allcator (e.g., cudaMalloc/ncclMemAlloc)
> 
>   - Register send and recv buffer for each rank in the communicator with ncclCommRegister
> 
>   - Launch NCCL collectives

## General Buffer Registration

Since 2.23.x, NCCL supports intra-node buffer registration, which targets all peer-to-peer intra-node communications (e.g., Allgather Ring) and brings less memory pressure, better communication and computation overlap performance. Either registering buffers by ncclCommRegister in the beginning or applying CUDA graph can enable intra-node buffer registration for NCCL collectives and sendrecv.

The user buffers can be allocated through VMM API (i.e., cuMem\*), any VMM-based allocators ([Memory Allocator](#mem-allocator)) or ncclMemAlloc will work. The buffers allocated through legacy cuda API (e.g., cudaMalloc) can also be used for registration. However, it is not safe due to the potential hang during execution and segmentation fault during failure and abort, so using legacy buffers for registration is not recommended; currently, legacy buffer registration is disabled by default, users can set NCCL\_LEGACY\_CUDA\_REGISTER=1 to enable it.

## Buffer Registration and PXN

Buffer registration for network communication (e.g., InfiniBand) and PXN are inherently incompatible. PXN is enabled by default in NCCL as long as the platform supports it, and it can be used for sendrecv-based operations and collectives. When PXN is enabled, the network buffer registration will not be enabled even if users have called ncclCommRegister to register the buffers. To enable network buffer registration, users can set NCCL\_PXN\_DISABLE=1 to disable PXN.

## Memory Allocator

For convenience, NCCL provides ncclMemAlloc function to help users to allocate buffers through VMM API, which can be used for NCCL registration later. It is only designed for NCCL so that it is not recommended to use ncclMemAlloc allocated buffers everywhere in the applications.

For advanced users, if you want to create your own memory allocator for NVLS UB, the allocated buffer of the allocator needs to satisfy the following requirements:

>   - Allocate buffer with shared flag CU\_MEM\_HANDLE\_TYPE\_POSIX\_FILE\_DESCRIPTOR and also CU\_MEM\_HANDLE\_TYPE\_FABRIC on GPUs where it’s supported.
> 
>   - Buffer physical memory size is multiple of CUMEM recommended granularity (i.e. cuMemGetAllocationGranularity(…, CU\_MEM\_ALLOC\_GRANULARITY\_RECOMMENDED\`))
> 
>   - Buffer virtual head address is at least aligned to CUMEM recommended granularity and size is multiple of CUMEM recommended granularity.

For general buffer registration with VMM API, the allocator needs to satisfy the same requirements as NVLS UB allocators.

## Window Registration

Since 2.27, NCCL supports window registration, which allows users to register local buffers into NCCL window and enables extremely low latency and high bandwith communication in NCCL. Currently, window registration supports input buffers only from VMM-based allocators ([Memory Allocator](#mem-allocator)) and ncclMemAlloc; any other type of cuda buffers will fail to be registered.

NCCL window registration is enabled by default. However, if users do not use window registration and need to turn it off, set NCCL\_WIN\_ENABLE=0 to disable it. In addition, users can also control the behavior of window registration through flags in [Window Registration Flags](../api/flags.html#win-flags).

The following example shows how to register buffers into NCCL window and use it for communication:

    void* src;
    void* dst;
    ncclWindow_t src_win;
    ncclWindow_t dst_win;
    
    CHECK(ncclMemAlloc(&src, src_size));
    CHECK(ncclMemAlloc(&dst, dst_size));
    // Passing NCCL_WIN_COLL_SYMMETRIC requires users to provide the symmetric buffers among all ranks in collectives.
    // Every rank needs to call ncclCommWindowRegister to register its buffers.
    CHECK(ncclCommWindowRegister(comm, src, src_size, &src_win, NCCL_WIN_COLL_SYMMETRIC));
    CHECK(ncclCommWindowRegister(comm, dst, dst_size, &dst_win, NCCL_WIN_COLL_SYMMETRIC));
    // Use the registered buffers for communication to enable symmetric communication benefits.
    // In this example, every rank has 0x1000 offset and 0x2000 offset from the head address of
    // src and dst respectively, which satisfies the symmetric buffer requirement.
    CHECK(ncclAllGather((uint8_t*)src + 0x1000, (uint8_t*)dst + 0x2000, 1, ncclInt8, comm, stream));
    CHECK(cudaStreamSynchronize(stream));
    
    CHECK(ncclCommWindowDeregister(comm, src_win));
    CHECK(ncclCommWindowDeregister(comm, dst_win));
    
    CHECK(ncclMemFree(src));
    CHECK(ncclMemFree(dst));

See the description of [`ncclCommWindowRegister()`](../api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister") and [`ncclCommWindowDeregister()`](../api/comms.html#c.ncclCommWindowDeregister "ncclCommWindowDeregister") for additional details.

## Zero-CTA Optimization

Since NCCL version 2.28, NCCL supports zero-CTA optimization. Zero-CTA optimization aims to avoid the use of CTA for communication and to overlap communication and computation.

Current zero-CTA optimization supports using the Copy Engine (CE) to perform the communication. The following are the requirements to enable zero-CTA optimization with CE:

>   - CUDA driver version \>= 12.5
> 
>   - Collectives run within a single NVL or MNNVL domain (does not support network, e.g., IB/ROCE)
> 
>   - The buffer is symmetrically registered with the NCCL window
> 
>   - The communicator is configured with the `NCCL_CTA_POLICY_ZERO` flag (please see [NCCL Communicator CTA Policy Flags](../api/flags.html#cta-policy-flags))
> 
>   - Supported collectives are AlltoAll, AllGather, Scatter, and Gather

The following example shows how to enable zero-CTA optimization:

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    // NCCL_CTA_POLICY_ZERO to enable zero-CTA optimization whenever possible
    config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
    
    void* src;
    void* dst;
    ncclWindow_t src_win;
    ncclWindow_t dst_win;
    
    CHECK(ncclMemAlloc(&src, src_size));
    CHECK(ncclMemAlloc(&dst, dst_size));
    
    // Register the buffers into NCCL symmetric window
    CHECK(ncclCommWindowRegister(comm, src, src_size, &src_win, NCCL_WIN_COLL_SYMMETRIC));
    CHECK(ncclCommWindowRegister(comm, dst, dst_size, &dst_win, NCCL_WIN_COLL_SYMMETRIC));
    
    CHECK(ncclAllGather(src, dst, 1, ncclInt8, comm, stream));
    CHECK(cudaStreamSynchronize(stream));
    
    CHECK(ncclCommWindowDeregister(comm, src_win));
    CHECK(ncclCommWindowDeregister(comm, dst_win));
    
    CHECK(ncclMemFree(src));
    CHECK(ncclMemFree(dst));

---

# Device-Initiated Communication

Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication primitives directly from user CUDA kernels.

## Device API

Device API consists of the following modules:

>   - **LSA (Load/Store Accessible)** – for communication between devices accessible via memory load/store operations, using CUDA P2P. This includes devices connected over NVLink and some devices connected over PCIe, so long as they have P2P connectivity with each other (as indicated by `nvidia-smi topo -p2p p`). Up to NCCL 2.28.3, the availability of LSA was also subject to the [NCCL\_P2P\_LEVEL](../env.html#env-nccl-p2p-level) distance check, but that is no longer the case with newer versions.
> 
>   - **Multimem** – for communication between devices using the hardware multicast feature provided by NVLink SHARP (available on some datacenter GPUs since the Hopper generation).
> 
>   - **GIN (GPU-Initiated Networking)** – for communication over the network (since NCCL 2.28.7).

## Requirements

The device API relies on symmetric memory (see [Window Registration](bufferreg.html#window-reg)), which in turn depends on GPU virtual memory management (see [NCCL\_CUMEM\_ENABLE](../env.html#env-nccl-cumem-enable)) and optionally – for multimem support – on NVLink SHARP (see [NCCL\_NVLS\_ENABLE](../env.html#env-nccl-nvls-enable)).

GIN has the following requirements:

  - CUDA 12.2 or later when compiling the GPU code

  - NVIDIA GPUs: Volta or newer. NVIDIA GPU drivers \>= 510.40.3

  - NVIDIA NICs: CX4 or newer. rdma-core \>= 44.0

  - GPU Direct RDMA: GIN host proxy requires DMA-BUF or nvidia-peermem support. GIN GDAKI requires DMA-BUF with kernel version \>= 6.1 or nvidia-peermem support

  - Network topology: Requires full NIC connectivity. Does not support topologies where NICs cannot communicate across rails. Also does not support `NCCL_CROSS_NIC=0`.

  - Fused NICs are not supported. To use GIN on dual-port NICs, set `NCCL_IB_MERGE_NICS=0`

Using the host RMA API requires CUDA 12.5 or greater.

Building with EMIT\_LLVM\_IR=1 (to generate readable LLCM intermediate representation code) requires CUDA 12.

## Host-Side Setup

To perform communication from the device kernel, a device communicator needs to be created first, using [`ncclDevCommCreate()`](../api/device.html#c.ncclDevCommCreate "ncclDevCommCreate"). Data transfer operations on buffers require symmetric memory windows (see [Window Registration](bufferreg.html#window-reg)). A custom communication kernel can then be launched using the standard CUDA syntax. The code excerpt below demonstrates these steps:

    int main() {
      [...]
      NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    
      /* Buffer initialization and window creation */
      char* buffer;
      size_t size = 256*1048576;
      NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
      ncclWindow_t win;
      NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));
    
      /* Get device communicator */
      ncclDevComm devComm;
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 16;
      reqs.lsaBarrierCount = nCTAs;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    
      /* Launch user kernel */
      customKernel<<<nCTAs, 512>>>(devComm, win);
      [...]
    }

Depending on the kernel and application requirements, the same window can be used for input and output, or multiple windows may be needed. When creating a device communicator, the resources that the kernel will need should be specified via the requirements list (see [`ncclDevCommRequirements`](../api/device.html#c.ncclDevCommRequirements "ncclDevCommRequirements")). In the above example we specify just the number of barriers that our LSA kernel will need, in this case one for each CTA the kernel is to be launched on (16, each CTA running 512 threads).

## Simple LSA Kernel

    template <typename T>
    __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
      ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x };
      bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    
      const int rank = devComm.lsaRank, nRanks = devComm.lsaSize;
      const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
      const int globalNthreads = blockDim.x * gridDim.x * nRanks;
    
      for (size_t o = globalTid; o < count; o += globalNthreads) {
        T v = 0;
        for (int peer = 0; peer < nRanks; peer++) {
          T* inputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
          v += inputPtr[o];
        }
        for (int peer = 0; peer < nRanks; peer++) {
          T* outputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
          outputPtr[o] = v;
        }
      }
    
      bar.sync(ncclCoopCta(), cuda::memory_order_release);
    }

The above code excerpt shows a simple device kernel – an in-place variant (the input buffer is reused for the output) of AllReduce, utilizing LSA support (data is transferred via memory load/store instructions).

The start of the buffer is specified as a (byte-based) *offset* within the previously registered window *win* (see [Window Registration](bufferreg.html#window-reg)); the buffer consists of *count* elements of type *T*.

Before the kernel can start processing data, it needs to ensure that all participants are ready. It creates a memory barrier session *bar* (see `ncclLsaBarrierSession`) and uses it to synchronize across all the threads of the CTA (*ncclCoopCta()*; see [Thread Groups](#devapi-coops)) and the ranks of the communicator (*devComm*). *ncclTeamTagLsa* indicates the subset of ranks the barrier will apply to (see [Teams](#devapi-teams)) – this kernel assumes that all ranks are LSA-connected. *blockIdx.x* is the CTA’s local index, used to select the barrier.

The kernel then calculates a globally unique index for each thread as well as the overall thread count, and can finally start processing data, using an all-to-all communication pattern. In each iteration of the outer loop, every participating thread loads a single input element from each communicator rank (the first inner loop). `ncclGetLsaPointer()` is used to calculate the locally-accessible address of the start of the buffer within each rank (remote device memory was previously mapped into the local address space – see [Window Registration](bufferreg.html#window-reg)). Extracted input data is accumulated and the result is stored back at each rank (the second inner loop). Before the kernel terminates, another memory synchronization needs to take place to ensure that all participants have finished processing their data.

Note that this simple implementation would likely fall short of achieving the peak bandwidth, as it utilizes neither vectorization nor loop unrolling.

## Multimem Device Kernel

    int main() {
      [...]
      reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 16;
      reqs.lsaBarrierCount = nCTAs;
      reqs.lsaMultimem = true;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
      [...]
    }
    
    template <typename T>
    __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
      ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem*/true };
      [...]
      T* mmPtr = (T*)ncclGetLsaMultimemPointer(win, offset, devComm);
      for (size_t o = globalTid; o < count; o += globalNthreads) {
        T v = multimem_sum(mmPtr+o);
        multimem_st(mmPtr+o, v);
      }
      [...]
    }

The above code excerpt demonstrates modifications needed to the earlier code segments to enable multimem support (the lines with critical changes are highlighted). On the host side, `lsaMultimem` needs to be set in the requirements prior to creating the device communicator ([`ncclDevCommCreate()`](../api/device.html#c.ncclDevCommCreate "ncclDevCommCreate") will fail if the necessary hardware support is unavailable).

Within the device kernel, we can switch the memory barrier to a multimem-optimized variant by adding an extra argument to the constructor. The processing loop is actually simpler with multimem: `ncclGetLsaMultimemPointer()` needs to be invoked just once per kernel. The returned multicast memory pointer enables access to the device memory of all the ranks of the communicator without having to iterate over them, and the data can be reduced in hardware. To keep this example simple, the implementations of `multimem_sum` and `multimem_st` are not included; they need to be implemented using PTX, e.g., `multimem.ld_reduce.global.add` and `multimem.st.global`.

## Thread Groups

Many functions in the device API take a thread cooperative group as input to indicate which threads within the CTA will take part in the operation. NCCL provides three predefined ones: `ncclCoopThread()`, `ncclCoopWarp()`, and (the most commonly used) `ncclCoopCta()`.

Users may also pass CUDA cooperative groups, or any class which provides `thread_rank()`, `size()`, and `sync()` methods.

## Teams

To address remote ranks or perform barriers, NCCL refers to subsets of ranks within a communicator as “teams”. NCCL provides three predefined ones:

>   - `ncclTeamWorld()` – the “world” team, encompassing all the ranks of a given communicator.
> 
>   - `ncclTeamLsa()` – all the peers accessible from the local rank using load/store operations.
> 
>   - `ncclTeamRail()` – the set of peers directly accessible from the local rank over the network, assuming that the network fabric is rail-optimized (see [NCCL\_CROSS\_NIC](../env.html#env-nccl-cross-nic)).

The `ncclTeam` structure contains fairly self-explanatory elements `nRanks`, `rank`, and `stride`. The device API contains functions to verify team membership, convert rank numbers between teams, etc. The world and LSA teams are always contiguous (stride `1`), whereas the rail team is typically not – its stride equals the size of the LSA team (the assumption is thus that each rank *n* within the local LSA team has direct network connectivity with corresponding ranks *n* of all remote LSA teams).

## Host-Accessible Device Pointer Functions

Starting with version 2.29, NCCL provides host-accessible functions that enable host code to obtain pointers to LSA memory regions.

The four functions are [`ncclGetLsaMultimemDevicePointer()`](../api/device.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer") (multimem base pointer), [`ncclGetMultimemDevicePointer()`](../api/device.html#c.ncclGetMultimemDevicePointer "ncclGetMultimemDevicePointer") (multimem base pointer with custom handle), [`ncclGetLsaDevicePointer()`](../api/device.html#c.ncclGetLsaDevicePointer "ncclGetLsaDevicePointer") (LSA peer pointer), and [`ncclGetPeerDevicePointer()`](../api/device.html#c.ncclGetPeerDevicePointer "ncclGetPeerDevicePointer") (world rank peer pointer). Functions automatically discover the associated communicator from the window object and return `ncclResult_t` error codes.

Usage Example:

    int main() {
      [...]
      // Allocate symmetric memory buffer
      char* buffer;
      size_t size = 256 * 1024 * 1024;  // 256 MB buffer
      NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
    
      // Create window with the allocated buffer
      ncclWindow_t win;
      NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));
    
      // Get host-accessible pointers
      void* multimemPtr;
      void* lsaPtr;
      void* peerPtr;
    
      // Get multimem pointer (returns nullptr if multimem not supported)
      NCCLCHECK(ncclGetLsaMultimemDevicePointer(win, 0, &multimemPtr));
      if (multimemPtr == nullptr) {
          // Multimem not available, use fallback
      }
    
      // Get LSA pointer for peer 1
      NCCLCHECK(ncclGetLsaDevicePointer(win, 0, 1, &lsaPtr));
    
      // Get peer pointer for world rank 2
      NCCLCHECK(ncclGetPeerDevicePointer(win, 0, 2, &peerPtr));
    
      // Use pointers in custom kernels or legacy code
      customKernel<<<nCTAs, 256>>>(multimemPtr, lsaPtr, peerPtr);
    
      // Cleanup
      NCCLCHECK(ncclCommWindowDeregister(comm, &win));
      // Device pointers are invalidated after window deregistration
      NCCLCHECK(ncclMemFree(buffer));
      [...]
    }

Important notes: Pointer lifetime is limited to the shorter of Window and Communicator lifetime. Functions should be called once and pointers cached for reuse. For detailed function documentation, see [Host-Accessible Device Pointer Functions](../api/device.html#device-api-host-functions).

## GIN Device Kernel

    int main() {
      [...]
      reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 1;
      reqs.railGinBarrierCount = nCTAs;
      reqs.ginSignalCount = 1;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
      [...]
    }
    
    template <typename T>
    __global__ void ginAlltoAllKernel(ncclDevComm devComm, ncclWindow_t win,
                                      size_t inputOffset, size_t outputOffset, size_t count) {
      int ginContext = 0;
      ncclGinSignal_t signalIndex = 0;
      ncclGin gin { devComm, ginContext };
      uint64_t signalValue = gin.readSignal(signalIndex);
    
      ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamWorld(devComm),
                                               devComm.railGinBarrier, blockIdx.x };
      bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    
      const int rank = devComm.rank, nRanks = devComm.nRanks;
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      const int nThreads = blockDim.x * gridDim.x;
    
      const size_t size = count * sizeof(T);
      for (int peer = tid; peer < nRanks; peer += nThreads) {
        gin.put(ncclTeamWorld(devComm), peer, win, outputOffset + rank * size,
                win, inputOffset + peer * size, size, ncclGin_SignalInc{signalIndex});
      }
    
      gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + nRanks);
      gin.flush(ncclCoopCta());
    }

The above code excerpt demonstrates modifications needed to the earlier host code to enable GIN support, available since NCCL 2.28.7 (the lines with critical changes are highlighted), and also includes a GIN AlltoAll kernel. On the host side, compared to the LSA kernels, we request a launch on just a single CTA (because our kernel doesn’t have much to do) and we set [`railGinBarrierCount`](../api/device.html#c.railGinBarrierCount "railGinBarrierCount") and [`ginSignalCount`](../api/device.html#c.ginSignalCount "ginSignalCount") to request GIN-specific barriers and signals ([`ncclDevCommCreate()`](../api/device.html#c.ncclDevCommCreate "ncclDevCommCreate") will fail if GIN support is unavailable). As with LSA barriers, we need as many of them as CTAs, but signals (used for completion notifications) can be shared between CTAs so, for this simple example, we’ll use just one per rank (for performance-oriented kernels, keeping signals exclusive to each CTA can improve performance).

On the device side, GIN API centers around the `ncclGin` object, initialized using the device communicator and a GIN context index (`0` will do for this simple example but, for performance-oriented kernels, using multiple contexts can provide a performance boost). To avoid race conditions, the initial value of the signal must be read *prior to* the synchronizing barrier. GIN-specific barriers look much like their LSA counterparts, being local to each CTA, but communicating over the network, not memory. *ncclTeamWorld* indicates all the ranks of a communicator (this kernel assumes that all the ranks can reach one another over the network, which in general need not be the case – see [NCCL\_CROSS\_NIC](../env.html#env-nccl-cross-nic)).

Unlike with the AllReduce kernels, for AlltoAll the calculated thread index needs to be unique only locally within each rank. This is then used to determine the destination peer. The main GIN data transfer operation is the one-sided `put()`, here launched in parallel on all participating threads, one per each destination peer (the loop is needed merely if the total rank count exceeds the local thread count – this is why we launched on just a single CTA). `put()` takes the usual arguments such as the destination rank and buffer address, the source buffer, and the transfer size. It also accepts several optional arguments; the above example takes advantage of the *remoteAction*, requesting that the destination peer increments the value of its local signal once the payload has been settled.

Once the local signal has been incremented by *nRanks*, we know that every peer has deposited their data in this rank’s output buffer and thus that the buffer is ready; `waitSignal()` can be used to block until that happens. Before terminating, the kernel still needs to `flush()` all the previously initiated outgoing `put()` operations – while that does not guarantee remote completion, it does ensure that the local input buffer is safe to reuse. We can skip an explicit barrier at the end, since `waitSignal()` and `flush()` together ensure that nobody else is using this rank’s buffers.

---

# NCCL API

The following sections describe the NCCL methods and operations.

  - [Communicator Creation and Management Functions](api/comms.html)
      - [ncclGetLastError](api/comms.html#ncclgetlasterror)
      - [ncclGetErrorString](api/comms.html#ncclgeterrorstring)
      - [ncclGetVersion](api/comms.html#ncclgetversion)
      - [ncclGetUniqueId](api/comms.html#ncclgetuniqueid)
      - [ncclCommInitRank](api/comms.html#ncclcomminitrank)
      - [ncclCommInitAll](api/comms.html#ncclcomminitall)
      - [ncclCommInitRankConfig](api/comms.html#ncclcomminitrankconfig)
      - [ncclCommInitRankScalable](api/comms.html#ncclcomminitrankscalable)
      - [ncclCommSplit](api/comms.html#ncclcommsplit)
      - [ncclCommShrink](api/comms.html#ncclcommshrink)
      - [ncclCommGetUniqueId](api/comms.html#ncclcommgetuniqueid)
      - [ncclCommGrow](api/comms.html#ncclcommgrow)
      - [ncclCommFinalize](api/comms.html#ncclcommfinalize)
      - [ncclCommRevoke](api/comms.html#ncclcommrevoke)
      - [ncclCommDestroy](api/comms.html#ncclcommdestroy)
      - [ncclCommAbort](api/comms.html#ncclcommabort)
      - [ncclCommGetAsyncError](api/comms.html#ncclcommgetasyncerror)
      - [ncclCommCount](api/comms.html#ncclcommcount)
      - [ncclCommCuDevice](api/comms.html#ncclcommcudevice)
      - [ncclCommUserRank](api/comms.html#ncclcommuserrank)
      - [ncclCommRegister](api/comms.html#ncclcommregister)
      - [ncclCommDeregister](api/comms.html#ncclcommderegister)
      - [ncclCommWindowRegister](api/comms.html#ncclcommwindowregister)
      - [ncclCommWindowDeregister](api/comms.html#ncclcommwindowderegister)
      - [ncclMemAlloc](api/comms.html#ncclmemalloc)
      - [ncclMemFree](api/comms.html#ncclmemfree)
  - [Collective Communication Functions](api/colls.html)
      - [ncclAllReduce](api/colls.html#ncclallreduce)
      - [ncclBroadcast](api/colls.html#ncclbroadcast)
      - [ncclReduce](api/colls.html#ncclreduce)
      - [ncclAllGather](api/colls.html#ncclallgather)
      - [ncclReduceScatter](api/colls.html#ncclreducescatter)
      - [ncclAlltoAll](api/colls.html#ncclalltoall)
      - [ncclGather](api/colls.html#ncclgather)
      - [ncclScatter](api/colls.html#ncclscatter)
  - [Group Calls](api/group.html)
      - [ncclGroupStart](api/group.html#ncclgroupstart)
      - [ncclGroupEnd](api/group.html#ncclgroupend)
      - [ncclGroupSimulateEnd](api/group.html#ncclgroupsimulateend)
  - [Point To Point Communication Functions](api/p2p.html)
      - [Two-Sided Point-to-Point Operations](api/p2p.html#two-sided-point-to-point-operations)
          - [ncclSend](api/p2p.html#ncclsend)
          - [ncclRecv](api/p2p.html#ncclrecv)
      - [One-Sided Point-to-Point Operations (RMA)](api/p2p.html#one-sided-point-to-point-operations-rma)
          - [ncclPutSignal](api/p2p.html#ncclputsignal)
          - [ncclSignal](api/p2p.html#ncclsignal)
          - [ncclWaitSignal](api/p2p.html#ncclwaitsignal)
  - [Types](api/types.html)
      - [ncclComm\_t](api/types.html#ncclcomm-t)
      - [ncclResult\_t](api/types.html#ncclresult-t)
      - [ncclDataType\_t](api/types.html#nccldatatype-t)
      - [ncclRedOp\_t](api/types.html#ncclredop-t)
      - [ncclScalarResidence\_t](api/types.html#ncclscalarresidence-t)
      - [ncclConfig\_t](api/types.html#ncclconfig-t)
      - [ncclSimInfo\_t](api/types.html#ncclsiminfo-t)
      - [ncclWindow\_t](api/types.html#ncclwindow-t)
  - [User Defined Reduction Operators](api/ops.html)
      - [ncclRedOpCreatePreMulSum](api/ops.html#ncclredopcreatepremulsum)
      - [ncclRedOpDestroy](api/ops.html#ncclredopdestroy)
  - [NCCL API Supported Flags](api/flags.html)
      - [Window Registration Flags](api/flags.html#window-registration-flags)
      - [NCCL Communicator CTA Policy Flags](api/flags.html#nccl-communicator-cta-policy-flags)
      - [Communicator Shrink Flags](api/flags.html#communicator-shrink-flags)
  - [Device API](api/device.html)
      - [Host-Side Setup](api/device.html#host-side-setup)
          - [ncclDevComm](api/device.html#nccldevcomm)
          - [ncclDevCommCreate](api/device.html#nccldevcommcreate)
          - [ncclDevCommDestroy](api/device.html#nccldevcommdestroy)
          - [ncclDevCommRequirements](api/device.html#nccldevcommrequirements)
          - [ncclCommQueryProperties](api/device.html#ncclcommqueryproperties)
          - [ncclCommProperties\_t](api/device.html#ncclcommproperties-t)
          - [ncclGinType\_t](api/device.html#ncclgintype-t)
      - [LSA](api/device.html#lsa)
          - [ncclLsaBarrierSession](api/device.html#nccllsabarriersession)
          - [ncclGetPeerPointer](api/device.html#ncclgetpeerpointer)
          - [ncclGetLsaPointer](api/device.html#ncclgetlsapointer)
          - [ncclGetLocalPointer](api/device.html#ncclgetlocalpointer)
      - [Multimem](api/device.html#multimem)
          - [ncclGetLsaMultimemPointer](api/device.html#ncclgetlsamultimempointer)
      - [Host-Accessible Device Pointer Functions](api/device.html#host-accessible-device-pointer-functions)
          - [ncclGetLsaMultimemDevicePointer](api/device.html#ncclgetlsamultimemdevicepointer)
          - [ncclGetMultimemDevicePointer](api/device.html#ncclgetmultimemdevicepointer)
          - [ncclGetLsaDevicePointer](api/device.html#ncclgetlsadevicepointer)
          - [ncclGetPeerDevicePointer](api/device.html#ncclgetpeerdevicepointer)
      - [GIN](api/device.html#gin)
          - [ncclGin](api/device.html#ncclgin)
          - [Signals and Counters](api/device.html#signals-and-counters)
          - [ncclGinBarrierSession](api/device.html#ncclginbarriersession)

---

# Communicator Creation and Management Functions

The following functions are public APIs exposed by NCCL to create and manage the collective communication operations.

## ncclGetLastError

  - const char \*ncclGetLastError([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm)  

Returns a human-readable string corresponding to the last error that occurred in NCCL. Note: The error is not cleared by calling this function. Please note that the string returned by ncclGetLastError could be unrelated to the current call and can be a result of previously launched asynchronous operations, if any.

## ncclGetErrorString

  - const char \*ncclGetErrorString([ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") result)  

Returns a human-readable string corresponding to the passed error code.

## ncclGetVersion

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGetVersion(int \*version)  

The ncclGetVersion function returns the version number of the currently linked NCCL library. The NCCL version number is returned in *version* and encoded as an integer which includes the `NCCL_MAJOR`, `NCCL_MINOR` and `NCCL_PATCH` levels. The version number returned will be the same as the `NCCL_VERSION_CODE` defined in *nccl.h*. NCCL version numbers can be compared using the supplied macro `NCCL_VERSION` as `NCCL_VERSION(MAJOR,MINOR,PATCH)`

## ncclGetUniqueId

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGetUniqueId(ncclUniqueId \*uniqueId)  

Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be called once when creating a communicator and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank. *uniqueId* should point to a ncclUniqueId object allocated by the user.

## ncclCommInitRank

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommInitRank([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*comm, int nranks, ncclUniqueId commId, int rank)  

Creates a new communicator (multi thread/process version). *rank* must be between 0 and *nranks*-1 and unique within a communicator clique. Each rank is associated to a CUDA device, which has to be set before calling ncclCommInitRank. ncclCommInitRank implicitly synchronizes with other ranks, hence it must be called by different threads/processes or used within ncclGroupStart/ncclGroupEnd.

## ncclCommInitAll

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommInitAll([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*comms, int ndev, const int \*devlist)  

Creates a clique of communicators (single process version) in a blocking way. This is a convenience function to create a single-process communicator clique. Returns an array of *ndev* newly initialized communicators in *comms*. *comms* should be pre-allocated with size at least ndev\*sizeof([`ncclComm_t`](types.html#c.ncclComm_t "ncclComm_t")). *devlist* defines the CUDA devices associated with each rank. If *devlist* is NULL, the first *ndev* CUDA devices are used, in order.

## ncclCommInitRankConfig

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommInitRankConfig([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*comm, int nranks, ncclUniqueId commId, int rank, [ncclConfig\_t](types.html#c.ncclConfig_t "ncclConfig_t") \*config)  

This function works the same way as *ncclCommInitRank* but accepts a configuration argument of extra attributes for the communicator. If config is passed as NULL, the communicator will have the default behavior, as if ncclCommInitRank was called.

See the [Creating a communicator with options](../usage/communicators.html#init-rank-config) section for details on configuration options.

## ncclCommInitRankScalable

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommInitRankScalable([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*newcomm, int nranks, int myrank, int nId, ncclUniqueId \*commIds, [ncclConfig\_t](types.html#c.ncclConfig_t "ncclConfig_t") \*config)  

This function works the same way as *ncclCommInitRankConfig* but accepts a list of ncclUniqueIds instead of a single one. If only one ncclUniqueId is passed, the communicator will be initialized as if ncclCommInitRankConfig was called. The provided ncclUniqueIds will all be used to initalize the single communicator given in argument.

See the [Creating a communicator with options](../usage/communicators.html#init-rank-config) section for details on how to create and distribute the list of ncclUniqueIds.

## ncclCommSplit

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommSplit([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int color, int key, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*newcomm, [ncclConfig\_t](types.html#c.ncclConfig_t "ncclConfig_t") \*config)  

The *ncclCommSplit* is a collective function and creates a set of new communicators from an existing one. Ranks which pass the same *color* value will be part of the same group; color must be a non-negative value. If it is passed as *NCCL\_SPLIT\_NOCOLOR*, it means that the rank will not be part of any group, therefore returning NULL as newcomm. The value of key will determine the rank order, and the smaller key means the smaller rank in new communicator. If keys are equal between ranks, then the rank in the original communicator will be used to order ranks. If the new communicator needs to have a special configuration, it can be passed as *config*, otherwise setting config to NULL will make the new communicator inherit the original communicator’s configuration. When split, there should not be any outstanding NCCL operations on the *comm*. Otherwise, it might cause a deadlock.

## ncclCommShrink

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommShrink([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int \*excludeRanksList, int excludeRanksCount, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*newcomm, [ncclConfig\_t](types.html#c.ncclConfig_t "ncclConfig_t") \*config, int shrinkFlags)  

The *ncclCommShrink* function creates a new communicator by removing specified ranks from an existing communicator. It is a collective function that must be called by all participating ranks in the newly created communicator. Ranks that are part of *excludeRanksList* should not call this function. The original ranks listed in *excludeRanksList* (of size *excludeRanksCount*) will be excluded from the new communicator. Within the new communicator, ranks will be updated to maintain a contiguous set of ids. If the new communicator needs a special configuration, it can be passed as *config*; otherwise, setting config to NULL will make the new communicator inherit the configuration of the parent communicator.

The *shrinkFlags* parameter controls the behavior of the operation. Use *NCCL\_SHRINK\_DEFAULT* (or *0*) for normal operation, or *NCCL\_SHRINK\_ABORT* when shrinking after an error on the parent communicator. Specifically, when using *NCCL\_SHRINK\_DEFAULT*, there should not be any outstanding NCCL operations on the *comm* to avoid potential deadlocks. Further, if the parent communicator has the flag config.shrinkShare set to 1, NCCL will reuse the parent communicator resources. On the other hand, when using *NCCL\_SHRINK\_ABORT*, NCCL will automatically abort any outstanding operations on the parent communicator, and no resources will be shared between the parent and the newly created communicator.

## ncclCommGetUniqueId

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommGetUniqueId([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, ncclUniqueId \*uniqueId)  

The *ncclCommGetUniqueId* function generates a unique identifier for growing an existing communicator. This function must be called by one rank (the coordinator) from the existing communicator, which will then distribute the *uniqueId* to all new ranks that will join the communicator via *ncclCommGrow*. The coordinator rank broadcasts the grow handle internally to boundary ranks (rank 0 and rank N-1) of the existing communicator to ensure proper coordination during the grow operation. This function should only be called when there are no outstanding NCCL operations on the communicator.

## ncclCommGrow

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommGrow([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int nRanks, const ncclUniqueId \*uniqueId, int rank, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") \*newcomm, [ncclConfig\_t](types.html#c.ncclConfig_t "ncclConfig_t") \*config)  

The *ncclCommGrow* function creates a new communicator by adding new ranks to an existing communicator. It must be called by both existing ranks (from the parent communicator) and new ranks (joining the communicator).

**For existing ranks:**

  - *comm* should be the parent communicator

  - *rank* must be set to *-1* (existing ranks retain their original rank in the new communicator)

  - *uniqueId* should be *NULL* (existing ranks receive coordination information internally)

  - The function creates *newcomm* with the same rank as in the parent communicator

**For new ranks:**

  - *comm* should be *NULL*

  - *rank* must be set to the desired rank in the new communicator (must be \>= parent communicator size)

  - *uniqueId* must be the unique identifier obtained from *ncclCommGetUniqueId* called by the coordinator

The *nRanks* parameter specifies the total number of ranks in the new communicator and must be greater than the size of the parent communicator. If the new communicator needs a special configuration, it can be passed as *config*; otherwise, setting config to NULL will make the new communicator inherit the configuration of the parent communicator (for existing ranks) or use default configuration (for new ranks).

There should not be any outstanding NCCL operations on the parent communicator when calling this function to avoid potential deadlocks. After the grow operation completes, the parent communicator should be destroyed using *ncclCommDestroy* to free resources.

**Example workflow:**

1.  Coordinator rank calls *ncclCommGetUniqueId* to generate the grow identifier

2.  Coordinator distributes the *uniqueId* to all new ranks (out-of-band)

3.  All existing ranks call *ncclCommGrow* with *comm*=parent, *rank*=-1, *uniqueId*=NULL (except for Coordinator rank which passes the *uniqueId*)

4.  All new ranks call *ncclCommGrow* with *comm*=NULL, *rank*=new\_rank, *uniqueId*=received\_id

## ncclCommFinalize

## ncclCommRevoke

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommRevoke([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int revokeFlags)  

Revokes in-flight operations on a communicator without destroying resources. Successful return may be *ncclInProgress* (non-blocking) while revocation completes asynchronously; applications can query *ncclCommGetAsyncError* until it returns *ncclSuccess*.

*revokeFlags* must be set to *NCCL\_REVOKE\_DEFAULT* (0). Other values are reserved for future use.

After revoke completes, the communicator is quiesced and safe for destroy, split, and shrink. Launching new collectives on a revoked communicator returns *ncclInvalidUsage*. Calling *ncclCommFinalize* after revoke is not supported. Resource sharing via *splitShare*/*shrinkShare* is disabled when the parent communicator is revoked.

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommFinalize([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm)  

Finalize a communicator object *comm*. When the communicator is marked as nonblocking, *ncclCommFinalize* is a nonblocking function. Successful return from it will set communicator state as *ncclInProgress* and indicates the communicator is under finalization where all uncompleted operations and the network-related resources are being flushed and freed. Once all NCCL operations are complete, the communicator will transition to the *ncclSuccess* state. Users can query that state with *ncclCommGetAsyncError*.

## ncclCommDestroy

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommDestroy([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm)  

Destroy a communicator object *comm*. *ncclCommDestroy* only frees the local resources that are allocated to the communicator object *comm* if *ncclCommFinalize* was previously called on the communicator; otherwise, *ncclCommDestroy* will call ncclCommFinalize internally. If *ncclCommFinalize* is called by users, users should guarantee that the state of the communicator becomes *ncclSuccess* before calling *ncclCommDestroy*. In all cases, the communicator should no longer be accessed after ncclCommDestroy returns. It is recommended that users call *ncclCommFinalize* and then *ncclCommDestroy*. This function is an intra-node collective call, which all ranks on the same node should call to avoid a hang.

## ncclCommAbort

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommAbort([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm)  

*ncclCommAbort* frees resources that are allocated to a communicator object *comm* and aborts any uncompleted operations before destroying the communicator. All active ranks are required to call this function in order to abort the NCCL communicator successfully. For more use cases, please check [Fault Tolerance](../usage/communicators.html#ft).

## ncclCommGetAsyncError

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommGetAsyncError([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") \*asyncError)  

Queries the progress and potential errors of asynchronous NCCL operations. Operations which do not require a stream argument (e.g. ncclCommFinalize) can be considered complete as soon as the function returns *ncclSuccess*; operations with a stream argument (e.g. ncclAllReduce) will return *ncclSuccess* as soon as the operation is posted on the stream but may also report errors through ncclCommGetAsyncError() until they are completed. If the return code of any NCCL function is *ncclInProgress*, it means the operation is in the process of being enqueued in the background, and users must query the states of the communicators until all the states become *ncclSuccess* before calling another NCCL function. Before the states change into *ncclSuccess*, users are not allowed to issue CUDA kernel to the streams being used by NCCL. If there has been an error on the communicator, user should destroy the communicator with [`ncclCommAbort()`](#c.ncclCommAbort "ncclCommAbort"). If an error occurs on the communicator, nothing can be assumed about the completion or correctness of operations enqueued on that communicator.

## ncclCommCount

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommCount(const [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int \*count)  

Returns in *count* the number of ranks in the NCCL communicator *comm*.

## ncclCommCuDevice

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommCuDevice(const [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int \*device)  

Returns in *device* the CUDA device associated with the NCCL communicator *comm*.

## ncclCommUserRank

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommUserRank(const [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, int \*rank)  

Returns in *rank* the rank of the caller in the NCCL communicator *comm*.

## ncclCommRegister

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommRegister(const [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, void \*buff, size\_t size, void \*\*handle)  

Registers the buffer *buff* with *size* under communicator *comm* for zero-copy communication; *handle* is returned for future deregistration. See *buff* and *size* requirements and more instructions in [User Buffer Registration](../usage/bufferreg.html#user-buffer-reg).

## ncclCommDeregister

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommDeregister(const [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, void \*handle)  

Deregister buffer represented by *handle* under communicator *comm*.

## ncclCommWindowRegister

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommWindowRegister([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, void \*buff, size\_t size, [ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") \*win, int winFlags)  

Collectively register local buffer *buff* with *size* under communicator *comm* into NCCL window. Since this is a collective call, every rank in the communicator needs to participate in the registration, and *size* by default needs to be equal among the ranks. *win* is returned for future deregistration (if called within a group, the value may not be filled in until ncclGroupEnd() has completed). See *buff* requirement and more instructions in [User Buffer Registration](../usage/bufferreg.html#user-buffer-reg). User can also pass different win flags to control the registration behavior. For more win flags information, please refer to [Window Registration Flags](flags.html#win-flags).

## ncclCommWindowDeregister

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommWindowDeregister([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, [ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") win)  

Deregister NCCL window represented by *win* under communicator *comm*. Deregistration is local to the rank, and caller needs to make sure the corresponding buffer within the window is not being accessed by any NCCL operation.

## ncclMemAlloc

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclMemAlloc(void \*\*ptr, size\_t size)  

Allocate a GPU buffer with *size*. Allocated buffer head address will be returned by *ptr*, and the actual allocated size can be larger than requested because of the buffer granularity requirements from all types of NCCL optimizations.

## ncclMemFree

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclMemFree(void \*ptr)  

Free memory allocated by *ncclMemAlloc()*.

---

# Collective Communication Functions

The following NCCL APIs provide some commonly used collective operations.

## ncclAllReduce

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclAllReduce(const void \*sendbuff, void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclRedOp\_t](types.html#c.ncclRedOp_t "ncclRedOp_t") op, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Reduces data arrays of length `count` in `sendbuff` using the `op` operation and leaves identical copies of the result in each `recvbuff`.
    
    In-place operation will happen if `sendbuff == recvbuff`.

Related links: [AllReduce](../usage/collectives.html#allreduce).

## ncclBroadcast

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclBroadcast(const void \*sendbuff, void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int root, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Copies `count` elements from `sendbuff` on the `root` rank to all ranks’ `recvbuff`. `sendbuff` is only used on rank `root` and ignored for other ranks.
    
    In-place operation will happen if `sendbuff == recvbuff`.

<!-- end list -->

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclBcast(void \*buff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int root, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Legacy in-place version of `ncclBroadcast` in a similar fashion to MPI\_Bcast. A call to
    
        ncclBcast(buff, count, datatype, root, comm, stream)
    
    is equivalent to
    
        ncclBroadcast(buff, buff, count, datatype, root, comm, stream)

Related links: [Broadcast](../usage/collectives.html#broadcast)

## ncclReduce

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclReduce(const void \*sendbuff, void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclRedOp\_t](types.html#c.ncclRedOp_t "ncclRedOp_t") op, int root, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Reduce data arrays of length `count` in `sendbuff` into `recvbuff` on the `root` rank using the `op` operation. `recvbuff` is only used on rank `root` and ignored for other ranks.
    
    In-place operation will happen if `sendbuff == recvbuff`.

Related links: [Reduce](../usage/collectives.html#reduce).

## ncclAllGather

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclAllGather(const void \*sendbuff, void \*recvbuff, size\_t sendcount, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Gathers `sendcount` values from all GPUs and leaves identical copies of the result in each `recvbuff`, receiving data from rank `i` at offset `i*sendcount`.
    
    Note: This assumes the receive count is equal to `nranks*sendcount`, which means that `recvbuff` should have a size of at least `nranks*sendcount` elements.
    
    In-place operation will happen if `sendbuff == recvbuff + rank * sendcount`.

Related links: [AllGather](../usage/collectives.html#allgather), [In-place Operations](../usage/inplace.html#in-place-operations).

## ncclReduceScatter

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclReduceScatter(const void \*sendbuff, void \*recvbuff, size\_t recvcount, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclRedOp\_t](types.html#c.ncclRedOp_t "ncclRedOp_t") op, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Reduce data in `sendbuff` from all GPUs using the `op` operation and leave the reduced result scattered over the devices so that the `recvbuff` on rank `i` will contain the i-th block of the result.
    
    Note: This assumes the send count is equal to `nranks*recvcount`, which means that `sendbuff` should have a size of at least `nranks*recvcount` elements.
    
    In-place operation will happen if `recvbuff == sendbuff + rank * recvcount`.

Related links: [ReduceScatter](../usage/collectives.html#reducescatter), [In-place Operations](../usage/inplace.html#in-place-operations).

## ncclAlltoAll

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclAlltoAll(const void \*sendbuff, void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Each rank sends `count` values to all other ranks and receives `count` values from all other ranks. Data to send to destination rank `j` is taken from `sendbuff+j*count` and data received from source rank `i` is placed at `recvbuff+i*count`.
    
    Note: This assumes the both total send and receive count is equal to `nranks*count`, which means that `sendbuff` and `recvbuff` should have a size of at least `nranks*count` elements.
    
    In-place operation is currently not supported.

Related links: [AlltoAll](../usage/collectives.html#alltoall).

## ncclGather

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGather(const void \*sendbuff, void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int root, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Each rank sends `count` elements from `sendbuff` to the `root` rank. On the `root` rank, data from rank `i` is placed at `recvbuff + i*count`. On non-root ranks, `recvbuff` is not used.
    
    Note: This assumes the receive count is equal to `nranks*count`, which means that `recvbuff` should have a size of at least `nranks*count` elements.
    
    In-place operation will happen if `sendbuff == recvbuff + root * count`.

Related links: [Gather](../usage/collectives.html#gather).

## ncclScatter

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclScatter(const void \*sendbuff, void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int root, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Each rank receives `count` elements from the `root` rank. On the `root` rank, `count` elements from `sendbuff + i*count` are sent to rank `i`. On non-root ranks, `sendbuff` is not used.
    
    Note: This assumes the send count is equal to `nranks*count`, which means that `sendbuff` should have a size of at least `nranks*count` elements.
    
    In-place operation will happen if `recvbuff == sendbuff + root * count`.

Related links: [Scatter](../usage/collectives.html#scatter).

---

# Group Calls

Group primitives define the behavior of the current thread to avoid blocking. They can therefore be used from multiple threads independently.

Related links: [Group Calls](../usage/groups.html#group-calls).

## ncclGroupStart

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGroupStart()  
      
    Start a group call.
    
    All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization.

## ncclGroupEnd

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGroupEnd()  
      
    End a group call.
    
    Returns when all operations since ncclGroupStart have been processed. This means the communication primitives have been enqueued to the provided streams, but are not necessarily complete.
    
    When used with the ncclCommInitRank call, the ncclGroupEnd call waits for all communicators to be initialized.

## ncclGroupSimulateEnd

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGroupSimulateEnd([ncclSimInfo\_t](types.html#c.ncclSimInfo_t "ncclSimInfo_t") \*simInfo)  
      
    Simulate a ncclGroupEnd() call and return NCCL’s simulation info in a structure passed as an argument.

---

# Point To Point Communication Functions

NCCL provides two types of point-to-point communication primitives: two-sided operations and one-sided operations.

## Two-Sided Point-to-Point Operations

(Since NCCL 2.7) Two-sided point-to-point communication primitives need to be used when ranks need to send and receive arbitrary data from each other, which cannot be expressed as a broadcast or allgather, i.e. when all data sent and received is different. Both sender and receiver must explicitly participate.

### ncclSend

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclSend(const void \*sendbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int peer, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Send data from `sendbuff` to rank `peer`.
    
    Rank `peer` needs to call ncclRecv with the same `datatype` and the same `count` as this rank.
    
    This operation is blocking for the GPU. If multiple [`ncclSend()`](#c.ncclSend "ncclSend") and [`ncclRecv()`](#c.ncclRecv "ncclRecv") operations need to progress concurrently to complete, they must be fused within a [`ncclGroupStart()`](group.html#c.ncclGroupStart "ncclGroupStart")/ [`ncclGroupEnd()`](group.html#c.ncclGroupEnd "ncclGroupEnd") section.

Related links: [Point-to-point communication](../usage/p2p.html#point-to-point).

### ncclRecv

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclRecv(void \*recvbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int peer, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Receive data from rank `peer` into `recvbuff`.
    
    Rank `peer` needs to call ncclSend with the same `datatype` and the same `count` as this rank.
    
    This operation is blocking for the GPU. If multiple [`ncclSend()`](#c.ncclSend "ncclSend") and [`ncclRecv()`](#c.ncclRecv "ncclRecv") operations need to progress concurrently to complete, they must be fused within a [`ncclGroupStart()`](group.html#c.ncclGroupStart "ncclGroupStart")/ [`ncclGroupEnd()`](group.html#c.ncclGroupEnd "ncclGroupEnd") section.

Related links: [Point-to-point communication](../usage/p2p.html#point-to-point).

## One-Sided Point-to-Point Operations (RMA)

One-sided Remote Memory Access (RMA) operations enable ranks to directly access remote memory without explicit participation from the target process. These operations require the target memory to be pre-registered within a symmetric memory window using [`ncclCommWindowRegister()`](comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister").

### ncclPutSignal

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclPutSignal(const void \*localbuff, size\_t count, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, int peer, [ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") peerWin, size\_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Write data from `localbuff` to rank `peer`’s registered memory window `peerWin` at offset `peerWinOffset` and subsequently updating a remote signal.
    
    The target memory window `peerWin` must be registered using [`ncclCommWindowRegister()`](comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister").
    
    The `sigIdx` is the signal index identifier for the operation. It must be set to 0 for now.
    
    The `ctx` is the context identifier for the operation. It must be set to 0 for now.
    
    The `flags` parameter is reserved for future use. It must be set to 0 for now.
    
    The return of [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") to the CPU thread indicates that the operation has been successfully enqueued to the CUDA stream. At the completion of [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") on the CUDA stream, the `localbuff` is safe to reuse or modify. When a signal is updated on the remote peer, it guarantees that the data from the corresponding [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") operation has been delivered to the remote memory. All prior [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") and [`ncclSignal()`](#c.ncclSignal "ncclSignal") operations to the same peer and context have also completed their signal updates.

Related links: [Point-to-point communication](../usage/p2p.html#point-to-point).

### ncclSignal

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Send a signal to rank `peer` without transferring data.
    
    The `sigIdx` is the signal index identifier for the operation. It must be set to 0 for now.
    
    The `ctx` is the context identifier for the operation. It must be set to 0 for now.
    
    The `flags` parameter is reserved for future use. It must be set to 0 for now.
    
    When a signal is updated on the remote peer, all prior [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") and [`ncclSignal()`](#c.ncclSignal "ncclSignal") operations to the same peer and context have also completed their signal updates.

Related links: [Point-to-point communication](../usage/p2p.html#point-to-point).

### ncclWaitSignal

  - type ncclWaitSignalDesc\_t  
      
    Descriptor that specifies how many signal operations to wait for from a particular rank on a given signal index and context.
    
      - int opCnt  
          
        Number of signal operations to wait for.
    
    <!-- end list -->
    
      - int peer  
          
        Target peer to wait for signals from.
    
    <!-- end list -->
    
      - int sigIdx  
          
        Signal index identifier. Must be set to 0 for now.
    
    <!-- end list -->
    
      - int ctx  
          
        Context identifier. Must be set to 0 for now.

<!-- end list -->

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclWaitSignal(int nDesc, [ncclWaitSignalDesc\_t](#c.ncclWaitSignalDesc_t "ncclWaitSignalDesc_t") \*signalDescs, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream\_t stream)  
      
    Wait for signals as described in the signal descriptor array.
    
    The `nDesc` parameter specifies the number of signal descriptors in the `signalDescs` array. Each descriptor indicates how many signals (`opCnt`) to expect from a specific `peer` on a particular signal index (`sigIdx`) and context (`ctx`).
    
    The return of [`ncclWaitSignal()`](#c.ncclWaitSignal "ncclWaitSignal") to the CPU thread indicates that the operation has been successfully enqueued to the CUDA stream. At the completion of [`ncclWaitSignal()`](#c.ncclWaitSignal "ncclWaitSignal") on the CUDA stream, all specified signal operations have been received and the corresponding data is visible in local memory.

Related links: [Point-to-point communication](../usage/p2p.html#point-to-point).

---

# Types

The following types are used by the NCCL library.

## ncclComm\_t

  - type ncclComm\_t  
      
    NCCL communicator. Points to an opaque structure inside NCCL.

## ncclResult\_t

  - type ncclResult\_t  
      
    Return values for all NCCL functions. Possible values are :
    
      - ncclSuccess  
          
        (`0`) Function succeeded.
    
    <!-- end list -->
    
      - ncclUnhandledCudaError  
          
        (`1`) A call to a CUDA function failed.
    
    <!-- end list -->
    
      - ncclSystemError  
          
        (`2`) A call to the system failed.
    
    <!-- end list -->
    
      - ncclInternalError  
          
        (`3`) An internal check failed. This is due to either a bug in NCCL or a memory corruption.
    
    <!-- end list -->
    
      - ncclInvalidArgument  
          
        (`4`) An argument has an invalid value.
    
    <!-- end list -->
    
      - ncclInvalidUsage  
          
        (`5`) The call to NCCL is incorrect. This is usually reflecting a programming error.
    
    <!-- end list -->
    
      - ncclRemoteError  
          
        (`6`) A call failed possibly due to a network error or a remote process exiting prematurely.
    
    <!-- end list -->
    
      - ncclInProgress  
          
        (`7`) A NCCL operation on the communicator is being enqueued and is being progressed in the background.
    
    Whenever a function returns an error (neither ncclSuccess nor ncclInProgress), NCCL should print a more detailed message when the environment variable [NCCL\_DEBUG](../env.html#nccl-debug) is set to “WARN”.

## ncclDataType\_t

  - type ncclDataType\_t  
      
    NCCL defines the following integral and floating data-types.
    
      - ncclInt8  
          
        Signed 8-bits integer
    
    <!-- end list -->
    
      - ncclChar  
          
        Signed 8-bits integer
    
    <!-- end list -->
    
      - ncclUint8  
          
        Unsigned 8-bits integer
    
    <!-- end list -->
    
      - ncclInt32  
          
        Signed 32-bits integer
    
    <!-- end list -->
    
      - ncclInt  
          
        Signed 32-bits integer
    
    <!-- end list -->
    
      - ncclUint32  
          
        Unsigned 32-bits integer
    
    <!-- end list -->
    
      - ncclInt64  
          
        Signed 64-bits integer
    
    <!-- end list -->
    
      - ncclUint64  
          
        Unsigned 64-bits integer
    
    <!-- end list -->
    
      - ncclFloat16  
          
        16-bits floating point number (half precision)
    
    <!-- end list -->
    
      - ncclHalf  
          
        16-bits floating point number (half precision)
    
    <!-- end list -->
    
      - ncclFloat32  
          
        32-bits floating point number (single precision)
    
    <!-- end list -->
    
      - ncclFloat  
          
        32-bits floating point number (single precision)
    
    <!-- end list -->
    
      - ncclFloat64  
          
        64-bits floating point number (double precision)
    
    <!-- end list -->
    
      - ncclDouble  
          
        64-bits floating point number (double precision)
    
    <!-- end list -->
    
      - ncclBfloat16  
          
        16-bits floating point number (truncated precision in bfloat16 format, CUDA 11 or later)
    
    <!-- end list -->
    
      - ncclFloat8e4m3  
          
        8-bits floating point number, 4 exponent bits, 3 mantissa bits (CUDA \>= 11.8 and SM \>= 90)
    
    <!-- end list -->
    
      - ncclFloat8e5m2  
          
        8-bits floating point number, 5 exponent bits, 2 mantissa bits (CUDA \>= 11.8 and SM \>= 90)

## ncclRedOp\_t

  - type ncclRedOp\_t  
      
    Defines the reduction operation.
    
      - ncclSum  
          
        Perform a sum (+) operation
    
    <!-- end list -->
    
      - ncclProd  
          
        Perform a product (\*) operation
    
    <!-- end list -->
    
      - ncclMin  
          
        Perform a min operation
    
    <!-- end list -->
    
      - ncclMax  
    
    Perform a max operation
    
      - ncclAvg  
    
    Perform an average operation, i.e. a sum across all ranks, divided by the number of ranks.

## ncclScalarResidence\_t

  - type ncclScalarResidence\_t  
      
    Indicates where (memory space) scalar arguments reside and when they can be dereferenced.
    
      - ncclScalarHostImmediate  
          
        The scalar resides in host memory and should be derefenced in the most immediate way.
    
    <!-- end list -->
    
      - ncclScalarDevice  
          
        The scalar resides on device visible memory and should be dereferenced once needed.

## ncclConfig\_t

  - type ncclConfig\_t  
      
    A structure-based configuration users can set to initialize a communicator; a newly created configuration must be initialized by NCCL\_CONFIG\_INITIALIZER.
    
      - NCCL\_CONFIG\_INITIALIZER  
          
        A configuration macro initializer which must be assigned to a newly created configuration.
    
    <!-- end list -->
    
      - blocking  
          
        This attribute can be set as integer 0 or 1 to indicate nonblocking or blocking communicator behavior correspondingly. Blocking is the default behavior.
    
    <!-- end list -->
    
      - cgaClusterSize  
          
        Set Cooperative Group Array (CGA) size of kernels launched by NCCL. This attribute can be set between 0 and 8, and the default value is 4 since sm90 architecture and 0 for older architectures.
    
    <!-- end list -->
    
      - minCTAs  
          
        Set the minimal number of CTAs NCCL should use for each kernel. Set to a positive integer value, up to 32. The default value is 1.
    
    <!-- end list -->
    
      - maxCTAs  
          
        Set the maximal number of CTAs NCCL should use for each kernel. Set to a positive integer value, up to 32. The default value is 32.
    
    <!-- end list -->
    
      - netName  
          
        Specify the network module name NCCL should use for network communication. The value of netName must match exactly the name of the network module (case-insensitive). NCCL internal network module names are “IB” (generic IB verbs) and “Socket” (TCP/IP sockets). External network plugins define their own names. The default value is undefined, and NCCL will choose the network module automatically.
    
    <!-- end list -->
    
      - splitShare  
          
        Specify whether to share resources with child communicator during communicator split. Set the value of splitShare to 0 or 1. The default value is 0. When the parent communicator is created with splitShare=1 during ncclCommInitRankConfig, the child communicator can share internal resources of the parent during communicator split. Split communicators are in the same family. When resources are shared, aborting any communicator can result in other communicators in the same family becoming unusable. Irrespective of whether sharing resources or not, users should always abort/destroy all no longer needed communicators to free up resources. Note: when the parent communicator has been revoked, resource sharing during split is disabled regardless of this flag.
    
    <!-- end list -->
    
      - shrinkShare  
          
        Specify whether to share resources with child communicator during communicator shrink. Set the value of shrinkShare to 0 or 1. The default value is 0. Note: when shrink is used with NCCL\_SHRINK\_ABORT, the value of shrinkShare is ignored and no resources are shared. When the parent communicator has been revoked, resource sharing is also disabled. The behavior of this flag is similar to splitShare, see above.
    
    <!-- end list -->
    
      - trafficClass  
          
        Set the traffic class (TC) to use for network operations on the communicator. The meaning of TC is specific to the network plugin in use by the communicator (e.g. IB networks use service level, RoCE networks use type of service). Assigning different TCs to each communicator can benefit workloads which overlap communication. TCs are defined by the system configuration and should be greater than or equal to 0. Note that environment variables, such as NCCL\_IB\_SL and NCCL\_IB\_TC, take precedence over user-specified TC values. To utilize user-defined TCs, ensure that these environment variables are unset.
    
    <!-- end list -->
    
      - collnetEnable  
          
        Set 1/0 to enable/disable IB SHARP on the communicator. The default value is 0 (disabled).
    
    <!-- end list -->
    
      - CTAPolicy  
          
        Set the policy for the communicator. The full list of supported policies can be found in [NCCL Communicator CTA Policy Flags](flags.html#cta-policy-flags). The default value is NCCL\_CTA\_POLICY\_DEFAULT.
    
    <!-- end list -->
    
      - nvlsCTAs  
          
        Set the total number of CTAs NCCL should use for NVLS kernels. Set to a positive integer value. By default, NCCL will automatically determine the best number of CTAs based on the system configuration.
    
    <!-- end list -->
    
      - commName  
          
        Specify the user defined name for the communicator. The communicator name can be used by NCCL to enrich logging and profiling.

<!-- end list -->

  - nChannelsPerNetPeer  
    
    > Set the number of network channels to be used for pairwise communication. The value must be a positive integer and will be round up to the next power of 2. The default value is optimized for the AlltoAll communication pattern. Consider increasing the value to increase the bandwidth for send/recv communication.
    
      - graphUsageMode  
          
        Set the graph usage mode for the communicator. It support three possible values: 0 (no graphs), 1 (one graph) and 2 (either multiple graphs or mix of graph and non-graph). The default value is 2.

## ncclSimInfo\_t

  - type ncclSimInfo\_t  
      
    This struct will be used by ncclGroupSimulateEnd() to return information about the calls.
    
      - NCCL\_SIM\_INFO\_INITIALIZER  
    
    NCCL\_SIM\_INFO\_INITIALIZER is a configuration macro initializer which must be assigned to a newly created ncclSimInfo\_t struct.
    
      - estimatedTime  
    
    Estimated time for the operation(s) in the group call will be returned in this attribute.

## ncclWindow\_t

  - type ncclWindow\_t  
      
    NCCL window object for window registration and deregistration.

---

# User Defined Reduction Operators

The following functions are public APIs exposed by NCCL to create and destroy custom reduction operators for use in reduction collectives.

## ncclRedOpCreatePreMulSum

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclRedOpCreatePreMulSum([ncclRedOp\_t](types.html#c.ncclRedOp_t "ncclRedOp_t") \*op, void \*scalar, [ncclDataType\_t](types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclScalarResidence\_t](types.html#c.ncclScalarResidence_t "ncclScalarResidence_t") residence, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm)  

Creates a new reduction operator which pre-multiplies input values by a given scalar locally before reducing them with peer values via summation. Both the input values and the scalar are of type *datatype*. For use only with collectives launched against *comm* and *datatype*. The *residence* argument indicates whether the memory pointed to by *scalar* should be dereferenced immediately by the host before this function returns (ncclScalarHostImmediate), or by the device during execution of the reduction collective (ncclScalarDevice). Upon return, the newly created operator’s handle is stored in *op*.

## ncclRedOpDestroy

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclRedOpDestroy([ncclRedOp\_t](types.html#c.ncclRedOp_t "ncclRedOp_t") op, [ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm)  

Destroys the reduction operator *op*. The operator must have been created by ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be destroyed as soon as the last NCCL function which is given that operator returns.

---

# NCCL API Supported Flags

The following show all flags which are supported by NCCL APIs.

## Window Registration Flags

  - NCCL\_WIN\_DEFAULT  
      
    Register buffer into NCCL window with default behavior. The default behavior allows users to pass any offset to the buffer head address as the input of NCCL collective operations. However, this behavior can cause suboptimal performance in NCCL due to the asymmetric buffer usage.

<!-- end list -->

  - NCCL\_WIN\_COLL\_SYMMETRIC  
      
    Register buffer into NCCL window, and users need to guarantee the offset to the buffer head address from all ranks must be equal when calling NCCL collective operations. It allows NCCL to operate buffer in a symmetric way and provide the best performance.

## NCCL Communicator CTA Policy Flags

  - NCCL\_CTA\_POLICY\_DEFAULT  
      
    Use the default CTA policy for NCCL communicator. In this policy, NCCL will automatically adjust resource usage and achieve maximal performance. This policy is suitable for most applications.

<!-- end list -->

  - NCCL\_CTA\_POLICY\_EFFICIENCY  
      
    Use the CTA efficiency policy for NCCL communicator. In this policy, NCCL will optimize CTA usage and use minimal number of CTAs to achieve the decent performance when possible. This policy is suitable for applications which require better compute and communication overlap.

<!-- end list -->

  - NCCL\_CTA\_POLICY\_ZERO  
      
    Use the Zero-CTA policy for NCCL communicator. In this policy, NCCL will use zero CTA whenever it can, even when that choice may sacrifice some performance. Select this mode when your application must preserve the maximum number of CTAs for compute kernels.

## Communicator Shrink Flags

These flags modify the behavior of the `ncclCommShrink` operation.

  - NCCL\_SHRINK\_DEFAULT  
      
    Default behavior. Shrink the parent communicator without affecting ongoing operations. Value: `0x00`.

<!-- end list -->

  - NCCL\_SHRINK\_ABORT  
      
    First, terminate ongoing parent communicator operations, and then proceed with shrinking the communicator. This is used for error recovery scenarios where the parent communicator might be in a hung state. Resources of parent comm are still not freed, users should decide whether to call ncclCommAbort on the parent communicator after shrink. Value: `0x01`.

---

# Device API

## Host-Side Setup

### ncclDevComm

  - type ncclDevComm  
      
    A structure describing a device communicator, as created on the host side using [`ncclDevCommCreate()`](#c.ncclDevCommCreate "ncclDevCommCreate"). The structure is used primarily on the device side; elements that could be of particular interest include:
    
      - int rank  
    
    The rank within the communicator.
    
      - int nRanks  
    
    The size of the communicator.
    
      - int lsaRank  
    
    <!-- end list -->
    
      - int lsaSize  
    
    Rank within the local LSA team and its size (see [Teams](../usage/deviceapi.html#devapi-teams)).
    
      - uint8\_t ginContextCount  
    
    The number of supported GIN contexts (see [`ncclGin`](#_CPPv47ncclGin "ncclGin"); available since NCCL 2.28.7).

### ncclDevCommCreate

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclDevCommCreate([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, struct [ncclDevCommRequirements](#c.ncclDevCommRequirements "ncclDevCommRequirements") const \*reqs, struct [ncclDevComm](#c.ncclDevComm "ncclDevComm") \*outDevComm)  

Creates a new device communicator (see [`ncclDevComm`](#c.ncclDevComm "ncclDevComm")) corresponding to the supplied host-side communicator *comm*. The result is returned in the *outDevComm* buffer (which needs to be supplied by the caller). The caller needs to also provide a filled-in list of requirements via the *reqs* argument (see [`ncclDevCommRequirements`](#c.ncclDevCommRequirements "ncclDevCommRequirements")); the function will allocate any necessary resources to meet them. It is recommended to call [`ncclCommQueryProperties()`](#c.ncclCommQueryProperties "ncclCommQueryProperties") before calling the function; the function will fail if the specified requirements are not supported. Since this is a collective call, every rank in the communicator needs to participate. If called within a group, *outDevComm* may not be filled in until `ncclGroupEnd()` has completed.

Note that this is a *host-side* function.

### ncclDevCommDestroy

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclDevCommDestroy([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, struct [ncclDevComm](#c.ncclDevComm "ncclDevComm") const \*devComm)  

Destroys a device communicator (see [`ncclDevComm`](#c.ncclDevComm "ncclDevComm")) previously created using [`ncclDevCommCreate()`](#c.ncclDevCommCreate "ncclDevCommCreate") and releases any allocated resources. The caller must ensure that no device kernel that uses this device communicator could be running at the time this function is invoked.

Note that this is a *host-side* function.

### ncclDevCommRequirements

  - type ncclDevCommRequirements  

A host-side structure specifying the list of requirements when creating device communicators (see [`ncclDevComm`](#c.ncclDevComm "ncclDevComm")). Since NCCL 2.29, this struct must be initialized using `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER`.

>   - bool lsaMultimem  
> 
> Specifies whether multimem support is required for all LSA ranks.
> 
>   - int lsaBarrierCount  
> 
> Specifies the number of memory barriers to allocate (see [`ncclLsaBarrierSession`](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession")).
> 
>   - int railGinBarrierCount  
> 
> Specifies the number of network barriers to allocate (see [`ncclGinBarrierSession`](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession"); available since NCCL 2.28.7).
> 
>   - int barrierCount  
> 
> Specifies the minimum number for both the memory and network barriers (see above; available since NCCL 2.28.7).
> 
>   - int ginSignalCount  
> 
> Specifies the number of network signals to allocate (see [`ncclGinSignal_t`](#_CPPv415ncclGinSignal_t "ncclGinSignal_t"); available since NCCL 2.28.7).
> 
>   - int ginCounterCount  
> 
> Specifies the number of network counters to allocate (see [`ncclGinCounter_t`](#_CPPv416ncclGinCounter_t "ncclGinCounter_t"); available since NCCL 2.28.7).
> 
>   - ncclDevResourceRequirements\_t \*resourceRequirementsList  
> 
> Specifies a list of resource requirements. This is best set to NULL for now.
> 
>   - ncclTeamRequirements\_t \*teamRequirementsList  
> 
> Specifies a list of requirements for particular teams. This is best set to NULL for now.

### ncclCommQueryProperties

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclCommQueryProperties([ncclComm\_t](types.html#c.ncclComm_t "ncclComm_t") comm, [ncclCommProperties\_t](#c.ncclCommProperties_t "ncclCommProperties_t") \*props)  

Exposes communicator properties by filling in *props*. Before calling this function, *props* must be initialized using `NCCL_COMM_PROPERTIES_INITIALIZER`. Introduced in NCCL 2.29.

Note that this is a *host-side* function.

### ncclCommProperties\_t

  - type ncclCommProperties\_t  

A structure describing the properties of the communicator. Introduced in NCCL 2.29. Properties include:

>   - int rank  
> 
> Rank within the communicator.
> 
>   - int nRanks  
> 
> Size of the communicator.
> 
>   - int cudaDev  
> 
> CUDA device index.
> 
>   - int nvmlDev  
> 
> NVML device index.
> 
>   - bool deviceApiSupport  
> 
> Whether the device API is supported. If false, a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created.
> 
>   - bool multimemSupport  
> 
> Whether ranks in the same LSA team can communicate using multimem. If false, a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created with multimem resources.
> 
>   - [ncclGinType\_t](#c.ncclGinType_t "ncclGinType_t") ginType  
> 
> The GIN type supported by the communicator. If equal to [`NCCL_GIN_TYPE_NONE`](#c.NCCL_GIN_TYPE_NONE "NCCL_GIN_TYPE_NONE"), a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created with GIN resources.

### ncclGinType\_t

  - type ncclGinType\_t  

GIN type. Communication between different GIN types is not supported. Possible values include:

>   - NCCL\_GIN\_TYPE\_NONE  
>       
>     GIN is not supported.
> 
> <!-- end list -->
> 
>   - NCCL\_GIN\_TYPE\_PROXY  
>       
>     Host Proxy GIN type.
> 
> <!-- end list -->
> 
>   - NCCL\_GIN\_TYPE\_GDAKI  
>       
>     GPUDirect Async Kernel-Initiated (GDAKI) GIN type.

## LSA

All functionality described from this point on is available on the device side only.

### ncclLsaBarrierSession

  - template\<typename Coop\>  
    class ncclLsaBarrierSession  
      
    A class representing a memory barrier session.
    
      - ncclLsaBarrierSession([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop") coop, ncclDevComm const \&comm, ncclTeamTagLsa tag, uint32\_t index, bool multimem = false)  
          
        Initializes a new memory barrier session. *coop* represents a cooperative group (see [Teams](../usage/deviceapi.html#devapi-teams)). *comm* is the device communicator created using [`ncclDevCommCreate()`](#c.ncclDevCommCreate "ncclDevCommCreate"). *ncclTeamTagLsa* is here to indicate which subset of ranks the barrier will apply to. The identifier of the underlying barrier to use is provided by *index* (it should be different for each *coop*; typically set to `blockIdx.x` to ensure uniqueness between CTAs). *multimem* requests a hardware-accelerated implementation using memory multicast.
    
    <!-- end list -->
    
      - void arrive([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop"), cuda::memory\_order order)  
    
    Signals the arrival of the thread at the barrier session.
    
      - void wait([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop"), cuda::memory\_order order)  
    
    Blocks until all threads of all team members arrive at the barrier session.
    
      - void sync([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop"), cuda::memory\_order order)  
    
    Synchronizes all threads of all team members that participate in the barrier session (combines `arrive` and `wait`).

### ncclGetPeerPointer

  - void \*ncclGetPeerPointer(ncclWindow\_t w, size\_t offset, int peer)  

Returns a load/store accessible pointer to the memory buffer of device *peer* within the window *w*. *offset* is byte-based. *peer* is a rank index within the world team (see [Teams](../usage/deviceapi.html#devapi-teams)). This function will return NULL if the *peer* is not within the LSA team.

### ncclGetLsaPointer

  - void \*ncclGetLsaPointer(ncclWindow\_t w, size\_t offset, int lsaPeer)  

Returns a load/store accessible pointer to the memory buffer of device *lsaPeer* within the window *w*. *offset* is byte-based. This is similar to `ncclGetPeerPointer`, but here *lsaPeer* is a rank index with the LSA team (see [Teams](../usage/deviceapi.html#devapi-teams)).

### ncclGetLocalPointer

  - void \*ncclGetLocalPointer(ncclWindow\_t w, size\_t offset)  

Returns a load-store accessible pointer to the memory buffer of the current device within the window *w*. *offset* is byte-based. This is just a shortcut version of `ncclGetPeerPointer` with *devComm.rank* as *peer*, or `ncclGetLsaPointer` with *devComm.lsaRank* as *lsaPeer*.

## Multimem

### ncclGetLsaMultimemPointer

  - void \*ncclGetLsaMultimemPointer(ncclWindow\_t w, size\_t offset, ncclDevComm const \&devComm)  

Returns a multicast memory pointer associated with the window *w* and device communicator *devComm*. *offset* is byte-based. Availability of multicast memory is hardware-dependent.

## Host-Accessible Device Pointer Functions

The following functions provide host-side access to device pointer functionality, enabling host code to obtain pointers to LSA memory regions.

All functions return `ncclResult_t` error codes. On success, `ncclSuccess` is returned. On failure, appropriate error codes are returned (e.g., `ncclInvalidArgument` for invalid parameters, `ncclInternalError` for internal failures), unless otherwise specified.

The returned pointers are valid for the lifetime of the window. Pointers should not be used after either the window or communicator is destroyed. Obtained pointers are device pointers.

### ncclGetLsaMultimemDevicePointer

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGetLsaMultimemDevicePointer([ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") window, size\_t offset, void \*\*outPtr)  

Returns a multimem base pointer for the LSA team associated with the given window. This function provides host-side access to the multimem memory functionality.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window. *outPtr* is the output parameter for the multimem pointer (must not be NULL).

This function requires LSA multimem support (multicast capability on the system). The window must be registered with a communicator that supports symmetric memory, and the hardware must support NVLink SHARP multicast functionality.

Note

If the system does not support multimem, the function returns `ncclSuccess` with `*outPtr` set to `nullptr`. This allows applications to gracefully detect and handle the absence of multimem support without breaking the communicator. Users should check if the returned pointer is `nullptr` to determine availability.

  - Example:
    
        void* multimemPtr;
        ncclResult_t result = ncclGetLsaMultimemDevicePointer(window, 0, &multimemPtr);
        if (result == ncclSuccess) {
            if (multimemPtr != nullptr) {
                // Use multimemPtr for multimem operations
            } else {
                // Multimem not supported, use fallback approach
            }
        }

### ncclGetMultimemDevicePointer

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGetMultimemDevicePointer([ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") window, size\_t offset, ncclMultimemHandle multimem, void \*\*outPtr)  

Returns a multimem base pointer using a provided multimem handle instead of the window’s internal multimem. This function enables using external or custom multimem handles for pointer calculation.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window. *multimem* is the multimem handle containing the multimem base pointer (multimem.mcBasePtr must not be NULL). *outPtr* is the output parameter for the multimem pointer (must not be NULL).

This function requires LSA multimem support (multicast capability on the system).

Note

If the system does not support multimem, the function returns `ncclSuccess` with `*outPtr` set to `nullptr`. The function validates that `multimem.mcBasePtr` is not nullptr before proceeding.

  - Example:
    
        // Get multimem handle from device communicator setup
        ncclMultimemHandle customHandle;
        // ... (obtain handle)
        
        void* multimemPtr;
        ncclResult_t result = ncclGetMultimemDevicePointer(window, 0, customHandle, &multimemPtr);
        if (result == ncclSuccess) {
            if (multimemPtr != nullptr) {
                // Use multimemPtr for multimem operations with custom handle
            } else {
                // Multimem not supported, use fallback approach
            }
        }

### ncclGetLsaDevicePointer

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGetLsaDevicePointer([ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") window, size\_t offset, int lsaRank, void \*\*outPtr)  

Returns a load/store accessible pointer to the memory buffer of a specific LSA peer within the window. This function provides host-side access to LSA pointer functionality using LSA rank directly.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window (must be \>= 0 and \< window size). *lsaRank* is the LSA rank of the target peer (must be \>= 0 and \< LSA team size). *outPtr* is the output parameter for the LSA pointer (must not be NULL).

On success, `ncclSuccess` is returned and the LSA pointer is returned in `outPtr`.

The window must be registered with a communicator that supports LSA. The LSA rank must be within the valid range for the LSA team, and the target peer must be load/store accessible (P2P connectivity required).

  - Example:
    
        void* lsaPtr;
        ncclResult_t result = ncclGetLsaDevicePointer(window, 0, 1, &lsaPtr);
        if (result == ncclSuccess) {
            // Use lsaPtr to access LSA peer 1's memory
        }

### ncclGetPeerDevicePointer

  - [ncclResult\_t](types.html#c.ncclResult_t "ncclResult_t") ncclGetPeerDevicePointer([ncclWindow\_t](types.html#c.ncclWindow_t "ncclWindow_t") window, size\_t offset, int peer, void \*\*outPtr)  

Returns a load/store accessible pointer to the memory buffer of a specific world rank peer within the window. This function converts world rank to LSA rank internally and provides host-side access to peer pointer functionality.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window. *peer* is the world rank of the target peer (must be \>= 0 and \< communicator size). *outPtr* is the output parameter for the peer pointer (must not be NULL).

On success, `ncclSuccess` is returned and the peer pointer is returned in `outPtr`.

If the peer is not reachable via LSA (not in LSA team), `outPtr` is set to NULL and `ncclSuccess` is returned. This matches the behavior of the device-side `ncclGetPeerPointer` function.

The window must be registered with a communicator that supports LSA. The peer rank must be within the valid range for the communicator, and the target peer must be load/store accessible (P2P connectivity required).

  - Example:
    
        void* peerPtr;
        ncclResult_t result = ncclGetPeerDevicePointer(window, 0, 2, &peerPtr);
        if (result == ncclSuccess) {
            if (peerPtr != NULL) {
                // Use peerPtr to access world rank 2's memory
            } else {
                // Peer 2 is not reachable via LSA
            }
        }

## GIN

GIN is supported since NCCL 2.28.7.

### ncclGin

  - class ncclGin  
      
    A class encompassing major elements of the GIN support.
    
      - ncclGin(ncclDevComm const \&comm, int contextIndex)  
          
        Initializes a new `ncclGin` object. *comm* is the device communicator created using [`ncclDevCommCreate()`](#c.ncclDevCommCreate "ncclDevCommCreate"). *contextIndex* is the index of the GIN context – a network communication channel. Using multiple GIN contexts allows the implementation to spread traffic onto multiple connections, avoiding locking and bottlenecks. Therefore, performance-oriented kernels should cycle among the available contexts to improve resource utilization (the number of available contexts is available via `ginContextCount`).
    
    <!-- end list -->
    
      - void put(ncclTeam team, int peer, ncclWindow\_t dstWnd, size\_t dstOffset, ncclWindow\_t srcWnd, size\_t srcOffset, size\_t bytes, RemoteAction remoteAction, LocalAction localAction, Coop coop, DescriptorSmem descriptor, cuda::thread\_scope alreadyReleased, cuda::thread\_scope expected\_scope)  
          
        Schedules a device-initiated, one-sided data transfer operation from a local buffer to a remote buffer on a peer.
        
        *peer* is a rank within *team* (see [Teams](../usage/deviceapi.html#devapi-teams)); it may refer to the local rank (a loopback). The destination and source buffers are each specified using the window (*dstWnd*, *srcWnd*) and a byte-based offset (*dstOffset*, *srcOffset*). *bytes* specifies the data transfer count in bytes.
        
        Arguments beyond the first seven are optional. *remoteAction* and *localAction* specify actions to undertake on the destination peer and on the local rank when the payload has been settled and the input has been consumed (respectively). They default to `ncclGin_None` (no action); other options include `ncclGin_Signal{Inc|Add}` (for *remoteAction*) and `ncclGin_CounterInc` (for *localAction*); see [Signals and Counters](#devapi-signals) below for more details. *coop* indicates the set of threads participating in this operation (see [Thread Groups](../usage/deviceapi.html#devapi-coops)); it defaults to `ncclCoopThread` (a single device thread), which is the recommended model.
        
        The visibility of the signal on the destination peer implies the visibility of the put data it is attached to *and all the preceding puts to the same peer, provided that they were issued using the same GIN context*.
        
        The API also defines an alternative, “convenience” variant of this method that uses `ncclSymPtr` types to specify the buffers and expects size to be conveyed in terms of the number of elements instead of the byte count. There are also two `putValue` variants that take a single element at a time (no greater than eight bytes), passed by value.
    
    <!-- end list -->
    
      - void flush(Coop coop, cuda::memory\_order ord = cuda::memory\_order\_acquire)  
          
        Ensures that all the pending transfer operations scheduled by any threads of *coop* are locally consumed, meaning that their source buffers are safe to reuse. Makes no claims regarding the completion status on the remote peer(s).

### Signals and Counters

  - type ncclGinSignal\_t  

Signals are used to trigger actions on remote peers, most commonly on the completion of a [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put") operation. They each have a 64-bit integer value associated with them that can be manipulated atomically.

  - class ncclGin\_SignalAdd  
    
      - [ncclGinSignal\_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal  
    
    <!-- end list -->
    
      - uint64\_t value  

<!-- end list -->

  - class ncclGin\_SignalInc  
    
      - [ncclGinSignal\_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal  

These objects can be passed as the *remoteAction* arguments of methods such as [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put") and [`ncclGin::signal()`](#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::signal") to describe the actions to perform on the peer on receipt – in this case, increase the value of a *signal* specified by index. `ncclGin_SignalInc{signalIdx}` is functionally equivalent to `ncclGin_SignalAdd{signalIdx, 1}`; however, it may not be mixed with other signal-modifying operations without an intervening signal reset (see below). Signal values use “rolling” comparison logic to ensure that an unsigned overflow maintains the property of `x < x + 1`.

**Signal methods of ncclGin:**

  - void [ncclGin](#_CPPv47ncclGin "ncclGin")::signal(ncclTeam team, int peer, RemoteAction remoteAction, Coop coop, DescriptorSmem descriptor, cuda::thread\_scope alreadyReleased, cuda::thread\_scope expected\_scope)  

<!-- end list -->

  - uint64\_t [ncclGin](#_CPPv47ncclGin "ncclGin")::readSignal([ncclGinSignal\_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal, int bits = 64, cuda::memory\_order ord = cuda::memory\_order\_acquire)  

<!-- end list -->

  - void [ncclGin](#_CPPv47ncclGin "ncclGin")::waitSignal(Coop coop, [ncclGinSignal\_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal, uint64\_t least, int bits = 64, cuda::memory\_order ord = cuda::memory\_order\_acquire)  

<!-- end list -->

  - void [ncclGin](#_CPPv47ncclGin "ncclGin")::resetSignal([ncclGinSignal\_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal)  

These are signal-specific methods of [`ncclGin`](#_CPPv47ncclGin "ncclGin"). [`ncclGin::signal()`](#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::signal") implements an explicit signal notification without an accompanying data transfer operation; it takes a subset of arguments of [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put"). [`ncclGin::readSignal()`](#_CPPv4N7ncclGin10readSignalE15ncclGinSignal_tiN4cuda12memory_orderE "ncclGin::readSignal") returns the bottom *bits* of the value of the *signal*. [`ncclGin::waitSignal()`](#_CPPv4N7ncclGin10waitSignalE4Coop15ncclGinSignal_t8uint64_tiN4cuda12memory_orderE "ncclGin::waitSignal") waits for the bottom *bits* of the *signal* value to meet or exceed *least*. Finally, [`ncclGin::resetSignal()`](#_CPPv4N7ncclGin11resetSignalE15ncclGinSignal_t "ncclGin::resetSignal") resets the *signal* value to `0` (this method may not race with concurrent modifications to the signal).

  - type ncclGinCounter\_t  

Counters are used to trigger actions on the local rank; as such, they are complementary to signals, which are meant for remote actions. Like signals, they use “rolling” comparison logic, but they are limited to storing values of at most 56 bits.

  - class ncclGin\_CounterInc  
    
      - [ncclGinCounter\_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter  

This object can be passed as the *localAction* argument of methods such as [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put"). It is the only action defined for counters.

**Counter methods of ncclGin:**

  - uint64\_t [ncclGin](#_CPPv47ncclGin "ncclGin")::readCounter([ncclGinCounter\_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter, int bits = 56, cuda::memory\_order ord = cuda::memory\_order\_acquire)  

<!-- end list -->

  - void [ncclGin](#_CPPv47ncclGin "ncclGin")::waitCounter(Coop coop, [ncclGinCounter\_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter, uint64\_t least, int bits = 56, cuda::memory\_order ord = cuda::memory\_order\_acquire)  

<!-- end list -->

  - void [ncclGin](#_CPPv47ncclGin "ncclGin")::resetCounter([ncclGinCounter\_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter)  

These are counter-specific methods of [`ncclGin`](#_CPPv47ncclGin "ncclGin") and they are functionally equivalent to their signal counterparts discussed above.

### ncclGinBarrierSession

  - template\<typename Coop\>  
    class ncclGinBarrierSession  
      
    A class representing a network barrier session.
    
      - ncclGinBarrierSession([Coop](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession::Coop") coop, [ncclGin](#_CPPv47ncclGin "ncclGin") gin, ncclTeamTagRail tag, uint32\_t index)  
          
        Initializes a new network barrier session. *coop* represents a cooperative group (see [Thread Groups](../usage/deviceapi.html#devapi-coops)). *gin* is a previously initialized [`ncclGin`](#_CPPv47ncclGin "ncclGin") object. *ncclTeamTagRail* indicates that the barrier will apply to all peers on the same rail as the local rank (see [Teams](../usage/deviceapi.html#devapi-teams)). *index* identifies the underlying barrier to use (it should be different for each *coop*; typically set to `blockIdx.x` to ensure uniqueness between CTAs).
    
    <!-- end list -->
    
      - ncclGinBarrierSession([Coop](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession::Coop") coop, [ncclGin](#_CPPv47ncclGin "ncclGin") gin, ncclTeam team, ncclGinBarrierHandle handle, uint32\_t index)  
          
        Initializes a new network barrier session. This is the general-purpose variant to be used, e.g., when communicating with ranks from the world team (see [Teams](../usage/deviceapi.html#devapi-teams)), whereas the previous variant was specific to the rail team. This variant expects *team* to be passed as an argument, and also takes an extra *handle* argument indicating the location of the underlying barriers (typically set to the `railGinBarrier` field of the device communicator).
    
    <!-- end list -->
    
      - void sync([Coop](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession::Coop") coop, cuda::memory\_order order, ncclGinFenceLevel fence)  
          
        Synchronizes all threads of all team members that participate in the barrier session. `ncclGinFenceLevel::Relaxed` is the only defined value for *fence* for now.

---

# Migrating from NCCL 1 to NCCL 2

If you are using NCCL 1.x and want to move to NCCL 2.x, be aware that the APIs have changed slightly. NCCL 2.x supports all of the collectives that NCCL 1.x supports, but with slight modifications to the API.

In addition, NCCL 2.x also requires the usage of the “Group API” when a single thread manages NCCL calls for multiple GPUs.

The following list summarizes the changes that may be required in usage of NCCL API when using an application that has a single thread that manages NCCL calls for multiple GPUs, and is ported from NCCL 1.x to 2.x:

## Initialization

In versions 1.x, NCCL had to be initialized using ncclCommInitAll at a single thread or having one thread per GPU concurrently call ncclCommInitRank. NCCL 2.x retains these two modes of initialization. It adds a new mode with the Group API where ncclCommInitRank can be called in a loop, like a communication call, as shown below. The loop has to be guarded by the Group start and end API.

    ncclGroupStart();
    for (int i=0; i<ngpus; i++) {
      cudaSetDevice(i);
      ncclCommInitRank(comms+i, ngpus, id, i);
    }
    ncclGroupEnd();

## Communication

In NCCL 2.x, the collective operation can be initiated for different devices by making calls in a loop, on a single thread. This is similar to the usage in NCCL 1.x. However, this loop has to be guarded by the Group API in 2.x. Unlike in 1.x, the application does not have to select the relevant CUDA device before making the communication API call. NCCL runtime internally selects the device associated with the NCCL communicator handle. For example:

    ncclGroupStart();
    for (int i=0; i<nLocalDevs; i++) {
      ncclAllReduce(..., comm[i], stream[i]);
    }
    ncclGroupEnd();

When using only one device per thread or one device per process, the general usage of the API remains unchanged from NCCL 1.x to 2.x. The usage of the group API is not required in this case.

## Counts

Counts provided as arguments are now of type size\_t instead of integer.

## In-place usage for AllGather and ReduceScatter

For more information, see “In-place Operations”.

## AllGather arguments order

The AllGather function had its arguments reordered. The prototype changed from:

    ncclResult_t  ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
       void* recvbuff, ncclComm_t comm, cudaStream_t stream);

to:

    ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
       ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

The recvbuff argument has been moved after sendbuff to be consistent with all the other operations.

## Datatypes

New datatypes have been added in NCCL 2.x. The ones present in NCCL 1.x did not change and are still usable in NCCL 2.x.

## Error codes

Error codes have been merged into the ncclInvalidArgument category and have been simplified. A new ncclInvalidUsage code has been created to cover new programming errors.

---

# Examples

The examples in this section provide an overall view of how to use NCCL in various environments, combining one or multiple techniques:

  - using multiple GPUs per thread/process

  - using multiple threads

  - using multiple processes - the examples with multiple processes use MPI as parallel runtime environment, but any multi-process system should be able to work similarly.

Ensure that you always check the return codes from the NCCL functions. For clarity, the following examples do not contain error checking.

## Communicator Creation and Destruction Examples

The following examples demonstrate common use cases for NCCL initialization.

### Example 1: Single Process, Single Thread, Multiple Devices

In the specific case of a single process, ncclCommInitAll can be used. Here is an example creating a communicator for 4 devices, therefore, there are 4 communicator objects:

    ncclComm_t comms[4];
    int devs[4] = { 0, 1, 2, 3 };
    ncclCommInitAll(comms, 4, devs);

Next, you can call NCCL collective operations using a single thread and group calls, or multiple threads, each provided with a comm object.

At the end of the program, all of the communicator objects are destroyed:

    for (int i=0; i<4; i++)
      ncclCommDestroy(comms[i]);

The following code depicts a complete working example with a single process that manages multiple devices:

    #include <stdlib.h>
    #include <stdio.h>
    #include "cuda_runtime.h"
    #include "nccl.h"
    
    #define CUDACHECK(cmd) do {                         \
      cudaError_t err = cmd;                            \
      if (err != cudaSuccess) {                         \
        printf("Failed: Cuda error %s:%d '%s'\n",       \
            __FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    #define NCCLCHECK(cmd) do {                         \
      ncclResult_t res = cmd;                           \
      if (res != ncclSuccess) {                         \
        printf("Failed, NCCL error %s:%d '%s'\n",       \
            __FILE__,__LINE__,ncclGetErrorString(res)); \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    int main(int argc, char* argv[])
    {
      ncclComm_t comms[4];
    
    
      //managing 4 devices
      int nDev = 4;
      int size = 32*1024*1024;
      int devs[4] = { 0, 1, 2, 3 };
    
    
      //allocating and initializing device buffers
      float** sendbuff = (float**)malloc(nDev * sizeof(float*));
      float** recvbuff = (float**)malloc(nDev * sizeof(float*));
      cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    
    
      for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
      }
    
    
      //initializing NCCL
      NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    
    
       //calling NCCL communication API. Group API is required when using
       //multiple devices per thread
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
            comms[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
    
    
      //synchronizing on CUDA streams to wait for completion of NCCL operation
      for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
      }
    
    
      //free device buffers
      for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
      }
    
    
      //finalizing NCCL
      for(int i = 0; i < nDev; ++i)
          ncclCommDestroy(comms[i]);
    
    
      printf("Success \n");
      return 0;
    }

### Example 2: One Device per Process or Thread

When a process or host thread is responsible for at most one GPU, ncclCommInitRank can be used as a collective call to create a communicator. Each thread or process will get its own object.

The following code is an example of a communicator creation in the context of MPI, using one device per MPI rank.

First, we retrieve MPI information about processes:

    int myRank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

Next, a single rank will create a unique ID and send it to all other ranks to make sure everyone has it:

    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

Finally, we create the communicator:

    ncclComm_t comm;
    ncclCommInitRank(&comm, nRanks, id, myRank);

We can now call the NCCL collective operations using the communicator.

    ncclAllReduce( ... , comm);

Finally, we destroy the communicator object:

    ncclCommDestroy(comm);

The following code depicts a complete working example with multiple MPI processes and one device per process:

    #include <stdio.h>
    #include "cuda_runtime.h"
    #include "nccl.h"
    #include "mpi.h"
    #include <unistd.h>
    #include <stdint.h>
    #include <stdlib.h>
    
    
    #define MPICHECK(cmd) do {                          \
      int e = cmd;                                      \
      if( e != MPI_SUCCESS ) {                          \
        printf("Failed: MPI error %s:%d '%d'\n",        \
            __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    #define CUDACHECK(cmd) do {                         \
      cudaError_t e = cmd;                              \
      if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",             \
            __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    #define NCCLCHECK(cmd) do {                         \
      ncclResult_t r = cmd;                             \
      if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",             \
            __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    static uint64_t getHash(const char* string, size_t n) {
      // Based on DJB2a, result = result * 33 ^ char
      uint64_t result = 5381;
      for (size_t c = 0; c < n; c++){
        result = ((result << 5) + result) ^ string[c];
      }
      return result;
    }
    
    /* Generate a hash of the unique identifying string for this host
     * that will be unique for both bare-metal and container instances
     * Equivalent of a hash of;
     *
     * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
     *
     */
    #define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
    static uint64_t getHostHash(const char* hostname) {
      char hostHash[1024];
    
      // Fall back is the hostname if something fails
      (void) strncpy(hostHash, hostname, sizeof(hostHash));
      int offset = strlen(hostHash);
    
      FILE *file = fopen(HOSTID_FILE, "r");
      if (file != NULL) {
        char *p;
        if (fscanf(file, "%ms", &p) == 1) {
            strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
            free(p);
        }
      }
      fclose(file);
    
      // Make sure the string is terminated
      hostHash[sizeof(hostHash)-1]='\0';
    
      return getHash(hostHash, strlen(hostHash));
    }
    
    static void getHostName(char* hostname, int maxlen) {
      gethostname(hostname, maxlen);
      for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
      }
    }
    
    
    int main(int argc, char* argv[])
    {
      int size = 32*1024*1024;
    
    
      int myRank, nRanks, localRank = 0;
    
    
      //initializing MPI
      MPICHECK(MPI_Init(&argc, &argv));
      MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
      MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    
    
      //calculating localRank based on hostname which is used in selecting a GPU
      uint64_t hostHashs[nRanks];
      char hostname[1024];
      getHostName(hostname, 1024);
      hostHashs[myRank] = getHostHash(hostname);
      MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
      for (int p=0; p<nRanks; p++) {
         if (p == myRank) break;
         if (hostHashs[p] == hostHashs[myRank]) localRank++;
      }
    
    
      ncclUniqueId id;
      ncclComm_t comm;
      float *sendbuff, *recvbuff;
      cudaStream_t s;
    
    
      //get NCCL unique ID at rank 0 and broadcast it to all others
      if (myRank == 0) ncclGetUniqueId(&id);
      MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    
    
      //picking a GPU based on localRank, allocate device buffers
      CUDACHECK(cudaSetDevice(localRank));
      CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
      CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
      CUDACHECK(cudaStreamCreate(&s));
    
    
      //initializing NCCL
      NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
    
    
      //communicating using NCCL
      NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
            comm, s));
    
    
      //completing NCCL operation by synchronizing on the CUDA stream
      CUDACHECK(cudaStreamSynchronize(s));
    
    
      //free device buffers
      CUDACHECK(cudaFree(sendbuff));
      CUDACHECK(cudaFree(recvbuff));
    
    
      //finalizing NCCL
      ncclCommDestroy(comm);
    
    
      //finalizing MPI
      MPICHECK(MPI_Finalize());
    
    
      printf("[MPI Rank %d] Success \n", myRank);
      return 0;
    }

### Example 3: Multiple Devices per Thread

You can combine both multiple process or threads and multiple device per process or thread. In this case, we need to use group semantics.

The following example combines MPI and multiple devices per process (=MPI rank).

First, we retrieve MPI information about processes:

    int myRank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

Next, a single rank will create a unique ID and send it to all other ranks to make sure everyone has it:

    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(id, sizeof(id), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

Then, we create our ngpus communicator objects, which are part of a larger group of ngpus\*nRanks:

    ncclComm_t comms[ngpus];
    ncclGroupStart();
    for (int i=0; i<ngpus; i++) {
      cudaSetDevice(devs[i]);
      ncclCommInitRank(comms+i, ngpus*nRanks, id, myRank*ngpus+i);
    }
    ncclGroupEnd();

Next, we call NCCL collective operations using a single thread and group calls, or multiple threads, each provided with a comm object.

At the end of the program, we destroy all communicators objects:

    for (int i=0; i<ngpus; i++)
      ncclCommDestroy(comms[i]);

The following code depicts a complete working example with multiple MPI processes and multiple devices per process:

    #include <stdio.h>
    #include "cuda_runtime.h"
    #include "nccl.h"
    #include "mpi.h"
    #include <unistd.h>
    #include <stdint.h>
    
    
    #define MPICHECK(cmd) do {                          \
      int e = cmd;                                      \
      if( e != MPI_SUCCESS ) {                          \
        printf("Failed: MPI error %s:%d '%d'\n",        \
            __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    #define CUDACHECK(cmd) do {                         \
      cudaError_t e = cmd;                              \
      if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",             \
            __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    #define NCCLCHECK(cmd) do {                         \
      ncclResult_t r = cmd;                             \
      if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",             \
            __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    
    
    static uint64_t getHash(const char* string) {
      // Based on DJB2a, result = result * 33 ^ char
      uint64_t result = 5381;
      for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) ^ string[c];
      }
      return result;
    }
    
    /* Generate a hash of the unique identifying string for this host
     * that will be unique for both bare-metal and container instances
     * Equivalent of a hash of;
     *
     * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
     *
     */
    #define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
    static uint64_t getHostHash(const char* hostname) {
      char hostHash[1024];
    
      // Fall back is the hostname if something fails
      (void) strncpy(hostHash, hostname, sizeof(hostHash));
      int offset = strlen(hostHash);
    
      FILE *file = fopen(HOSTID_FILE, "r");
      if (file != NULL) {
        char *p;
        if (fscanf(file, "%ms", &p) == 1) {
            strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
            free(p);
        }
      }
      fclose(file);
    
      // Make sure the string is terminated
      hostHash[sizeof(hostHash)-1]='\0';
    
      return getHash(hostHash, strlen(hostHash));
    }
    
    static void getHostName(char* hostname, int maxlen) {
      gethostname(hostname, maxlen);
      for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
      }
    }
    
    
    int main(int argc, char* argv[])
    {
      int size = 32*1024*1024;
    
    
      int myRank, nRanks, localRank = 0;
    
    
      //initializing MPI
      MPICHECK(MPI_Init(&argc, &argv));
      MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
      MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    
    
      //calculating localRank which is used in selecting a GPU
      uint64_t hostHashs[nRanks];
      char hostname[1024];
      getHostName(hostname, 1024);
      hostHashs[myRank] = getHostHash(hostname);
      MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
      for (int p=0; p<nRanks; p++) {
         if (p == myRank) break;
         if (hostHashs[p] == hostHashs[myRank]) localRank++;
      }
    
    
      //each process is using two GPUs
      int nDev = 2;
    
    
      float** sendbuff = (float**)malloc(nDev * sizeof(float*));
      float** recvbuff = (float**)malloc(nDev * sizeof(float*));
      cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    
    
      //picking GPUs based on localRank
      for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(localRank*nDev + i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
      }
    
    
      ncclUniqueId id;
      ncclComm_t comms[nDev];
    
    
      //generating NCCL unique ID at one process and broadcasting it to all
      if (myRank == 0) ncclGetUniqueId(&id);
      MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    
    
      //initializing NCCL, group API is required around ncclCommInitRank as it is
      //called across multiple GPUs in each thread/process
      NCCLCHECK(ncclGroupStart());
      for (int i=0; i<nDev; i++) {
         CUDACHECK(cudaSetDevice(localRank*nDev + i));
         NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
      }
      NCCLCHECK(ncclGroupEnd());
    
    
      //calling NCCL communication API. Group API is required when using
      //multiple devices per thread/process
      NCCLCHECK(ncclGroupStart());
      for (int i=0; i<nDev; i++)
         NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
               comms[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
    
    
      //synchronizing on CUDA stream to complete NCCL communication
      for (int i=0; i<nDev; i++)
          CUDACHECK(cudaStreamSynchronize(s[i]));
    
    
      //freeing device memory
      for (int i=0; i<nDev; i++) {
         CUDACHECK(cudaFree(sendbuff[i]));
         CUDACHECK(cudaFree(recvbuff[i]));
      }
    
    
      //finalizing NCCL
      for (int i=0; i<nDev; i++) {
         ncclCommDestroy(comms[i]);
      }
    
    
      //finalizing MPI
      MPICHECK(MPI_Finalize());
    
    
      printf("[MPI Rank %d] Success \n", myRank);
      return 0;
    }

### Example 4: Multiple communicators per device

NCCL allows users to create multiple communicators per device. The following code shows an example with multiple MPI processes, one device per process, and multiple communicators per device:

    // blocking communicators
    CUDACHECK(cudaSetDevice(localRank));
    for (int i = 0; i < commNum; ++i) {
      if (myRank == 0) ncclGetUniqueId(&id);
      MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
      NCCLCHECK(ncclCommInitRank(&blockingComms[i], nRanks, id, myRank));
    }
    
    // non-blocking communicators
    CUDACHECK(cudaSetDevice(localRank));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    for (int i = 0; i < commNum; ++i) {
      if (myRank == 0) ncclGetUniqueId(&id);
      MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
      NCCLCHECK(ncclCommInitRankConfig(&nonblockingComms[i], nRanks, id, myRank, &config));
      do {
        NCCLCHECK(ncclCommGetAsyncError(nonblockingComms[i], &state));
      } while(state == ncclInProgress && checkTimeout() != true);
    }

checkTimeout() should be a user-defined function. For more nonblocking communicator usage, please check [Fault Tolerance](usage/communicators.html#ft). In addition, if you want to split communicators instead of creating a new one, please check [`ncclCommSplit()`](api/comms.html#c.ncclCommSplit "ncclCommSplit").

## Communication Examples

The following examples demonstrate common patterns for executing NCCL collectives.

### Example 1: One Device per Process or Thread

If you have a thread or process per device, then each thread calls the collective operation for its device, for example, AllReduce:

    ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);

After the call, the operation has been enqueued to the stream. Therefore, you can call cudaStreamSynchronize if you want to wait for the operation to be complete:

    cudaStreamSynchronize(stream);

For a complete working example with MPI and single device per MPI process, see “Example 2: One Device per Process or Thread”.

### Example 2: Multiple Devices per Thread

When a single thread manages multiple devices, you need to use group semantics to launch the operation on multiple devices at once:

    ncclGroupStart();
    for (int i=0; i<ngpus; i++)
      ncclAllReduce(sendbuffs[i], recvbuff[i], count, datatype, op, comms[i], streams[i]);
    ncclGroupEnd();

After ncclGroupEnd, all of the operations have been enqueued to the stream. Therefore, you can now call cudaStreamSynchronize if you want to wait for the operation to be complete:

    for (int i=0; i<ngpus; i++)
      cudaStreamSynchronize(streams[i]);

For a complete working example with MPI and multiple devices per MPI process, see [Example 3: Multiple Devices per Thread](#ex3).

---

# NCCL and MPI

## API

The NCCL API and usage is similar to MPI but there are many minor differences. The following list summarizes these differences:

### Using multiple devices per process

Similarly to the concept of MPI endpoints, NCCL does not require ranks to be mapped 1:1 to processes. A NCCL communicator may have many ranks (and, thus, multiple devices) associated to a single process. Hence, if used with MPI, a single MPI rank (a NCCL process) may have multiple devices associated with it.

### ReduceScatter operation

The ncclReduceScatter operation is similar to the MPI\_Reduce\_scatter\_block operation, not the MPI\_Reduce\_scatter operation. The MPI\_Reduce\_scatter function is intrinsically a “vector” function, while MPI\_Reduce\_scatter\_block (defined later to fill the missing semantics) provides regular counts similarly to the mirror function MPI\_Allgather. This is an oddity of MPI which has not been fixed for legitimate retro-compatibility reasons and that NCCL does not follow.

### Send and Receive counts

In many collective operations, MPI allows for different send and receive counts and types, as long as sendcount\*sizeof(sendtype) == recvcount\*sizeof(recvtype). NCCL does not allow that, defining a single count and a single data-type.

For AllGather and ReduceScatter operations, the count is equal to the per-rank size, which is the smallest size; the other count being equal to nranks\*count. The function prototype clearly shows which count is provided. ncclAllGather has a sendcount as argument, while ncclReduceScatter has a recvcount as argument.

Note: When performing or comparing AllReduce operations using a combination of ReduceScatter and AllGather, define the sendcount and recvcount as the total count divided by the number of ranks, with the correct count rounding-up, if it is not a perfect multiple of the number of ranks.

### Other collectives and point-to-point operations

NCCL does not define specific verbs for sendrecv, gather, gatherv, scatter, scatterv, alltoall, alltoallv, alltoallw, nor neighbor collectives. All those operations can be simply expressed using a combination of ncclSend, ncclRecv, and ncclGroupStart/ncclGroupEnd, similarly to how they can be expressed with MPI\_Isend, MPI\_Irecv and MPI\_Waitall.

ncclRecv does not support the equivalent of MPI\_ANY\_SOURCE; a specific source rank must always be provided. Similarly, the provided receive count must match the send count. Further, there is no concept of message tags.

### In-place operations

For more information, see [In-place Operations](usage/inplace.html#in-place-operations).

## Using NCCL within an MPI Program

NCCL can be easily used in conjunction with MPI. NCCL collectives are similar to MPI collectives, therefore, creating a NCCL communicator out of an MPI communicator is straightforward. It is therefore easy to use MPI for CPU-to-CPU communication and NCCL for GPU-to-GPU communication.

However, some implementation details in MPI can lead to issues when using NCCL inside an MPI program.

### MPI Progress

MPI defines a notion of progress which means that MPI operations need the program to call MPI functions (potentially multiple times) to make progress and eventually complete.

In some implementations, progress on one rank may need MPI to be called on another rank. While this is usually bad for performance, it can be argued that this is a valid MPI implementation.

As a result, blocking on a NCCL collective operation, for example calling cudaStreamSynchronize, may create a deadlock in some cases because not calling MPI on one rank could block other ranks, preventing them from reaching the NCCL call that would unblock the NCCL collective on the first rank.

In that case, the cudaStreamSynchronize call should be replaced by a loop like the following:

    cudaError_t err = cudaErrorNotReady;
    int flag;
    while (err == cudaErrorNotReady) {
      err = cudaStreamQuery(args->streams[i]);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }

### Inter-GPU Communication with CUDA-aware MPI

Using NCCL to perform inter-GPU communication concurrently with CUDA-aware MPI may create deadlocks.

NCCL creates inter-device dependencies, meaning that after it has been launched, a NCCL kernel will wait (and potentially block the CUDA device) until all ranks in the communicator launch their NCCL kernel. CUDA-aware MPI may also create such dependencies between devices depending on the MPI implementation.

Using both MPI and NCCL to perform transfers between the same sets of CUDA devices concurrently is therefore not guaranteed to be safe.

---

# Environment Variables

NCCL has an extensive set of environment variables to tune for specific usage.

Environment variables can also be set statically in /etc/nccl.conf (for an administrator to set system-wide values) or in ${NCCL\_CONF\_FILE} (since 2.23; see below). For example, those files could contain :

    NCCL_DEBUG=WARN
    NCCL_SOCKET_IFNAME==ens1f0

There are two categories of environment variables. Some are needed to make NCCL follow system-specific configuration, and can be kept in scripts and system configuration. Other parameters listed in the “Debugging” section should not be used in production nor retained in scripts, or only as workaround, and removed as soon as the issue is resolved. Keeping them set may result in sub-optimal behavior, crashes, or hangs.

## System configuration

### NCCL\_SOCKET\_IFNAME

The `NCCL_SOCKET_IFNAME` variable specifies which IP interfaces to use for communication.

#### Values accepted

Define to a list of prefixes to filter interfaces to be used by NCCL.

Multiple prefixes can be provided, separated by the `,` symbol.

Using the `^` symbol, NCCL will exclude interfaces starting with any prefix in that list.

To match (or not) an exact interface name, begin the prefix string with the `=` character.

Examples:

`eth` : Use all interfaces starting with `eth`, e.g. `eth0`, `eth1`, …

`=eth0` : Use only interface `eth0`

`=eth0,eth1` : Use only interfaces `eth0` and `eth1`

`^docker` : Do not use any interface starting with `docker`

`^=docker0` : Do not use interface `docker0`.

Note: By default, the loopback interface (`lo`) and docker interfaces (`docker*`) would not be selected unless there are no other interfaces available. If you prefer to use `lo` or `docker*` over other interfaces, you would need to explicitly select them using `NCCL_SOCKET_IFNAME`. The default algorithm will also favor interfaces starting with `ib` over others. Setting `NCCL_SOCKET_IFNAME` will bypass the automatic interface selection algorithm and may use all interfaces matching the manual selection.

### NCCL\_SOCKET\_FAMILY

The `NCCL_SOCKET_FAMILY` variable allows users to force NCCL to use only IPv4 or IPv6 interface.

#### Values accepted

Set to `AF_INET` to force the use of IPv4, or `AF_INET6` to force IPv6 usage.

### NCCL\_SOCKET\_RETRY\_CNT

(since 2.24)

The `NCCL_SOCKET_RETRY_CNT` variable specifies the number of times NCCL retries to establish a socket connection after an `ETIMEDOUT`, `ECONNREFUSED`, or `EHOSTUNREACH` error.

#### Values accepted

The default value is 34, any positive value is valid.

### NCCL\_SOCKET\_RETRY\_SLEEP\_MSEC

(since 2.24)

The `NCCL_SOCKET_RETRY_SLEEP_MSEC` variable specifies the number of milliseconds NCCL waits before retrying to establish a socket connection after the first `ETIMEDOUT`, `ECONNREFUSED`, or `EHOSTUNREACH` error. For subsequent errors, the waiting time scales linearly with the error count. The total time will therefore be (N+1) \* N/2 \* `NCCL_SOCKET_RETRY_SLEEP_MSEC`, where N is given by `NCCL_SOCKET_RETRY_CNT`. With the default values of `NCCL_SOCKET_RETRY_CNT` and `NCCL_SOCKET_RETRY_SLEEP_MSEC`, the total retry time will be approx. 60 seconds.

#### Values accepted

The default value is 100 milliseconds, any positive value is valid.

### NCCL\_SOCKET\_POLL\_TIMEOUT\_MSEC

(since 2.28)

The `NCCL_SOCKET_POLL_TIMEOUT_MSEC` variable specifies a timeout in milliseconds for a poll which can reduce the CPU usage during bootstrap. Normally NCCL will retry the operation until it completes. Polling in between attempts should reduce load on the CPU so that it can engage in activities that might make the operation able to complete sooner.

#### Values accepted

Non-negative integer. The old behavior corredponds to 0 (the default). If 0, it will not poll, but keep trying to progress the socket operation without pause. If non-zero, it will poll for up that amount of time before trying to progress the operation again.

### NCCL\_SOCKET\_NTHREADS

(since 2.4.8)

The `NCCL_SOCKET_NTHREADS` variable specifies the number of CPU helper threads used per network connection for socket transport. Increasing this value may increase the socket transport performance, at the cost of a higher CPU usage.

#### Values accepted

1 to 16. On AWS, the default value is 2; on Google Cloud instances with the gVNIC network interface, the default value is 4 (since 2.5.6); in other cases, the default value is 1.

For generic 100G networks, this value can be manually set to 4. However, the product of `NCCL_SOCKET_NTHREADS` and `NCCL_NSOCKS_PERTHREAD` cannot exceed 64. See also `NCCL_NSOCKS_PERTHREAD`.

### NCCL\_NSOCKS\_PERTHREAD

(since 2.4.8)

The `NCCL_NSOCKS_PERTHREAD` variable specifies the number of sockets opened by each helper thread of the socket transport. In environments where per-socket speed is limited, setting this variable larger than 1 may improve the network performance.

#### Values accepted

On AWS, the default value is 8; in other cases, the default value is 1.

For generic 100G networks, this value can be manually set to 4. However, the product of `NCCL_SOCKET_NTHREADS` and `NCCL_NSOCKS_PERTHREAD` cannot exceed 64. See also `NCCL_SOCKET_NTHREADS`.

### NCCL\_CROSS\_NIC

The `NCCL_CROSS_NIC` variable controls whether NCCL should allow rings/trees to use different NICs, causing inter-node communication to use different NICs on different nodes.

To maximize inter-node communication performance when using multiple NICs, NCCL tries to use the same NICs when communicating between nodes, to allow for a network design where each NIC on a node connects to a different network switch (network rail), and avoid any risk of traffic flow interference. The `NCCL_CROSS_NIC` setting is therefore dependent on the network topology, and in particular on whether the network fabric is rail-optimized or not.

This has no effect on systems with only one NIC.

#### Values accepted

0: Always use the same NIC for the same ring/tree, to avoid crossing network rails. Suited for networks with per NIC switches (rails), with a slow inter-rail connection. Note that if the communicator does not contain the same GPUs on each node, NCCL may still need to communicate across NICs.

1: Allow the use of different NICs for the same ring/tree. This is suited for networks where all NICs from a node are connected to the same switch, hence trying to communicate across the same NICs does not help avoiding flow collisions.

2: (Default) Try to use the same NIC for the same ring/tree, but still allow for the use of different NICs if it would result in a better performance.

### NCCL\_IB\_HCA

The `NCCL_IB_HCA` variable specifies which Host Channel Adapter (RDMA) interfaces to use for communication.

#### Values accepted

Define to filter IB Verbs interfaces to be used by NCCL. The list is comma-separated; port numbers can be specified using the `:` symbol. An optional prefix `^` indicates the list is an exclude list. A second optional prefix `=` indicates that the tokens are exact names, otherwise by default NCCL would treat each token as a prefix.

Examples:

`mlx5` : Use all ports of all cards starting with `mlx5`

`=mlx5_0:1,mlx5_1:1` : Use ports 1 of cards `mlx5_0` and `mlx5_1`.

`^=mlx5_1,mlx5_4` : Do not use cards `mlx5_1` and `mlx5_4`.

Note: using `mlx5_1` without a preceding `=` will select `mlx5_1` as well as `mlx5_10` to `mlx5_19`, if they exist. It is therefore always recommended to add the `=` prefix to ensure an exact match.

Note: There is a fixed upper limit of 32 Host Channel Adapter (HCA) devices supported in NCCL.

### NCCL\_IB\_TIMEOUT

The `NCCL_IB_TIMEOUT` variable controls the InfiniBand Verbs Timeout.

The timeout is computed as 4.096 µs \* 2 ^ *timeout*, and the correct value is dependent on the size of the network. Increasing that value can help on very large networks, for example, if NCCL is failing on a call to *ibv\_poll\_cq* with error 12.

For more information, see section 12.7.34 of the InfiniBand specification Volume 1 (Local Ack Timeout).

#### Values accepted

The default value used by NCCL is 20 (since 2.23; it was 18 since 2.14, and 14 before that).

Values can be 1-31.

Note: Setting a value of 0 or \>= 32 will result in an infinite timeout value.

### NCCL\_IB\_RETRY\_CNT

(since 2.1.15)

The `NCCL_IB_RETRY_CNT` variable controls the InfiniBand retry count.

For more information, see section 12.7.38 of the InfiniBand specification Volume 1.

#### Values accepted

The default value is 7.

### NCCL\_IB\_GID\_INDEX

(since 2.1.4)

The `NCCL_IB_GID_INDEX` variable defines the Global ID index used in RoCE mode. See the InfiniBand *show\_gids* command in order to set this value.

For more information, see the InfiniBand specification Volume 1 or vendor documentation.

#### Values accepted

The default value is -1.

### NCCL\_IB\_ADDR\_FAMILY

(since 2.21)

The `NCCL_IB_ADDR_FAMILY` variable defines the IP address family associated to the infiniband GID dynamically selected by NCCL when `NCCL_IB_GID_INDEX` is left unset.

#### Values accepted

The default value is “AF\_INET”.

### NCCL\_IB\_ADDR\_RANGE

(since 2.21)

The `NCCL_IB_ADDR_RANGE` variable defines the range of valid GIDs dynamically selected by NCCL when `NCCL_IB_GID_INDEX` is left unset.

#### Values accepted

By default, ignored if unset.

GID ranges can be defined using the Classless Inter-Domain Routing (CIDR) format for IPv4 and IPv6 families.

### NCCL\_IB\_ROCE\_VERSION\_NUM

(since 2.21)

The `NCCL_IB_ROCE_VERSION_NUM` variable defines the RoCE version associated to the infiniband GID dynamically selected by NCCL when `NCCL_IB_GID_INDEX` is left unset.

#### Values accepted

The default value is 2.

### NCCL\_IB\_SL

(since 2.1.4)

Defines the InfiniBand Service Level.

For more information, see the InfiniBand specification Volume 1 or vendor documentation.

#### Values accepted

The default value is 0.

### NCCL\_IB\_TC

(since 2.1.15)

Defines the InfiniBand traffic class field.

For more information, see the InfiniBand specification Volume 1 or vendor documentation.

#### Values accepted

The default value is 0.

### NCCL\_IB\_FIFO\_TC

(since 2.22.3)

Defines the InfiniBand traffic class for control messages. Control messages are short RDMA write operations which control credit return, contrary to other RDMA operations transmitting large segments of data. This setting allows to have those messages use a high priority, low-latency traffic class and avoid being delayed by the rest of the traffic.

#### Values accepted

The default value is the traffic class set by NCCL\_IB\_TC, which defaults to 0 if not set.

### NCCL\_IB\_RETURN\_ASYNC\_EVENTS

(since 2.23)

IB events are reported to the user as warnings. If enabled, NCCL will also stop IB communications upon fatal IB asynchronous events.

#### Values accepted

The default value is 1, set to 0 to disable

### NCCL\_OOB\_NET\_ENABLE

(since 2.23) The variable `NCCL_OOB_NET_ENABLE` enables the use of NCCL net for out-of-band communications. Enabling the usage of NCCL net will change the implementation of the allgather performed during the communicator initialization.

#### Values accepted

Set the variable to 0 to disable, and to 1 to enable.

### NCCL\_OOB\_NET\_IFNAME

(since 2.23) If NCCL net is enabled for out-of-band communication (see `NCCL_OOB_NET_ENABLE`), the `NCCL_OOB_NET_IFNAME` variable specifies which network interfaces to use.

#### Values accepted

Define to filter interfaces to be used by NCCL for out-of-band communications. The list of accepted interface depends on the network used by NCCL. The list is comma-separated; port numbers can be specified using the `:` symbol. An optional prefix `^` indicates the list is an exclude list. A second optional prefix `=` indicates that the tokens are exact names, otherwise by default NCCL would treat each token as a prefix. If multiple devices are specified, NCCL will select the first matching device in the list.

Example:

`NCCL_NET="IB" NCCL_OOB_NET_ENABLE=1 NCCL_OOB_NET_IFNAME="=mlx5_1"` will use the Infiniband NET, with the interface `mlx5_1`

`NCCL_NET="IB" NCCL_OOB_NET_ENABLE=1 NCCL_OOB_NET_IFNAME="mlx5_1"` will use the Infiniband NET, with the first interface found in the list of `mlx5_1`, `mlx5_10`, `mlx5_11`, etc.

`NCCL_NET="Socket" NCCL_OOB_NET_ENABLE=1 NCCL_OOB_NET_IFNAME="ens1"` will use the socket NET, with the first interface found in the list of `ens1f0`, `ens1f1`, etc.

### NCCL\_UID\_STAGGER\_THRESHOLD

(since 2.23) The `NCCL_UID_STAGGER_THRESHOLD` variable is used to trigger staggering of communications between NCCL ranks and the ncclUniqueId in order to avoid overflowing the ncclUniqueId. If the number of NCCL ranks communicating exceeds the specified threshold, the communications are staggered using the rank value (see NCCL\_UID\_STAGGER\_RATE below). If the number of NCCL ranks per ncclUniqueId is smaller or equal to the threshold, no staggering is performed.

For example, if we have 128 NCCL ranks, 1 ncclUniqueId, and a threshold at 64, staggering is performed. However, if 2 ncclUniqueIds are used with 128 NCCL ranks and a threshold at 64, no staggering is done.

#### Values accepted

The value of `NCCL_UID_STAGGER_THRESHOLD` must be a strictly positive integer. If unspecified, the default value is 256.

### NCCL\_UID\_STAGGER\_RATE

(since 2.23)

The `NCCL_UID_STAGGER_RATE` variable is used to define the message rate targeted when staggering the communications between NCCL ranks and the ncclUniqueId. If staggering is used (see NCCL\_UID\_STAGGER\_THRESHOLD above), the message rate is used to compute the time a given NCCL rank has to wait.

#### Values accepted

The value of `NCCL_UID_STAGGER_RATE` must be a strictly positive integer, expressed in messages/second. If unspecified, the default value is 7000.

### NCCL\_NET

(since 2.10)

Forces NCCL to use a specific network, for example to make sure NCCL uses an external plugin and doesn’t automatically fall back on the internal IB or Socket implementation. Setting this environment variable will override the `netName` configuration in all communicators (see [ncclConfig\_t](api/types.html#ncclconfig)); if not set (undefined), the network module will be determined by the configuration; if not passing configuration, NCCL will automatically choose the best network module.

#### Values accepted

The value of NCCL\_NET has to match exactly the name of the NCCL network used (case-insensitive). Internal network names are “IB” (generic IB verbs) and “Socket” (TCP/IP sockets). External network plugins define their own names. Default value is undefined.

### NCCL\_NET\_PLUGIN

(since 2.11)

  - Set it to either a suffix string or to a library name to choose among multiple NCCL net plugins. This setting will cause NCCL to look for the net plugin library using the following strategy:
    
      - If NCCL\_NET\_PLUGIN is set, attempt loading the library with name specified by NCCL\_NET\_PLUGIN;
    
      - If NCCL\_NET\_PLUGIN is set and previous failed, attempt loading libnccl-net-\<NCCL\_NET\_PLUGIN\>.so;
    
      - If NCCL\_NET\_PLUGIN is not set, attempt loading libnccl-net.so;
    
      - If no plugin was found (neither user defined nor default), use internal network plugin.

For example, setting `NCCL_NET_PLUGIN=foo` will cause NCCL to try load `foo` and, if `foo` cannot be found, `libnccl-net-foo.so` (provided that it exists on the system).

#### Values accepted

Plugin suffix, plugin file name, or “none”.

### NCCL\_TUNER\_PLUGIN

  - Set it to either a suffix string or to a library name to choose among multiple NCCL tuner plugins. This setting will cause NCCL to look for the tuner plugin library using the following strategy:
    
      - If NCCL\_TUNER\_PLUGIN is set, attempt loading the library with name specified by NCCL\_TUNER\_PLUGIN;
    
      - If NCCL\_TUNER\_PLUGIN is set and previous failed, attempt loading libnccl-net-\<NCCL\_TUNER\_PLUGIN\>.so;
    
      - If NCCL\_TUNER\_PLUGIN is not set, attempt loading libnccl-tuner.so;
    
      - If no plugin was found look for the tuner symbols in the net plugin (refer to `NCCL_NET_PLUGIN`);
    
      - If no plugin was found (neither through NCCL\_TUNER\_PLUGIN nor NCCL\_NET\_PLUGIN), use internal tuner plugin.

For example, setting `NCCL_TUNER_PLUGIN=foo` will cause NCCL to try load `foo` and, if `foo` cannot be found, `libnccl-tuner-foo.so` (provided that it exists on the system).

#### Values accepted

Plugin suffix, plugin file name, or “none”.

### NCCL\_PROFILER\_PLUGIN

  - Set it to either a suffix string or to a library name to choose among multiple NCCL profiler plugins. This setting will cause NCCL to look for the profiler plugin library using the following strategy:
    
      - If NCCL\_PROFILER\_PLUGIN is set, attempt loading the library with name specified by NCCL\_PROFILER\_PLUGIN;
    
      - If NCCL\_PROFILER\_PLUGIN is set and previous failed, attempt loading libnccl-profiler-\<NCCL\_PROFILER\_PLUGIN\>.so;
    
      - If NCCL\_PROFILER\_PLUGIN is not set, attempt loading libnccl-profiler.so;
    
      - If no plugin was found (neither user defined nor default), do not enable profiling.
    
      - If NCCL\_PROFILER\_PLUGIN is set to `STATIC_PLUGIN`, the plugin symbols are searched in the program binary.

For example, setting `NCCL_PROFILER_PLUGIN=foo` will cause NCCL to try load `foo` and, if `foo` cannot be found, `libnccl-profiler-foo.so` (provided that it exists on the system).

#### Values accepted

Plugin suffix, plugin file name, or “none”.

### NCCL\_ENV\_PLUGIN

(since 2.28)

  - The `NCCL_ENV_PLUGIN` variable can be used to let NCCL load an external environment plugin. Set it to either a library name or a suffix string to choose among multiple NCCL environment plugins. This setting will cause NCCL to look for the environment plugin library using the following strategy:
    
      - If `NCCL_ENV_PLUGIN` is set to a library name, attempt loading that library (e.g. `NCCL_ENV_PLUGIN=/path/to/library/libfoo.so` will cause NCCL to try load `/path/to/library/libfoo.so`);
    
      - If `NCCL_ENV_PLUGIN` is set to a suffix string, attempt loading `libnccl-env-<NCCL_ENV_PLUGIN>.so` (e.g. `NCCL_ENV_PLUGIN=foo` will cause NCCL to try load `libnccl-env-foo.so` from the system library path);
    
      - If `NCCL_ENV_PLUGIN` is not set, attempt loading the default `libnccl-env.so` library from the system library path;
    
      - If `NCCL_ENV_PLUGIN` is set to “none”, explicitly disable the external plugin and use the internal one;
    
      - If no plugin was found (neither user defined nor default) or the variable is set to “none”, use the internal environment plugin.

#### Values accepted

Plugin library name (e.g., `/path/to/library/libfoo.so`), suffix (e.g., `foo`), or “none”.

### NCCL\_IGNORE\_CPU\_AFFINITY

(since 2.4.6)

The `NCCL_IGNORE_CPU_AFFINITY` variable can be used to cause NCCL to ignore the job’s supplied CPU affinity and instead use the GPU affinity only.

#### Values accepted

The default is 0, set to 1 to cause NCCL to ignore the job’s supplied CPU affinity.

### NCCL\_CONF\_FILE

(since 2.23)

The `NCCL_CONF_FILE` variable allows the user to specify a file with the static configuration. This does not accept the `~` character as part of the path; please convert to a relative or absolute path first.

#### Values accepted

If unset or if the version is prior to 2.23, NCCL uses .nccl.conf in the home directory if available.

### NCCL\_DEBUG

The `NCCL_DEBUG` variable controls the debug information that is displayed from NCCL. This variable is commonly used for debugging.

#### Values accepted

VERSION - Prints the NCCL version at the start of the program.

WARN - Prints an explicit error message whenever any NCCL call errors out.

INFO - Prints debug information

TRACE - Prints replayable trace information on every call.

### NCCL\_DEBUG\_FILE

(since 2.2.12)

The `NCCL_DEBUG_FILE` variable directs the NCCL debug logging output to a file. The filename format can be set to *filename.%h.%p* where *%h* is replaced with the hostname and *%p* is replaced with the process PID. This does not accept the `~` character as part of the path, please convert to a relative or absolute path first.

#### Values accepted

The default output file is *stdout* unless this environment variable is set. The filename can also be set to `/dev/stdout` or `/dev/stderr` to direct NCCL debug logging output to those predefined I/O streams. This also has the effect of making the output line buffered.

Setting `NCCL_DEBUG_FILE` will cause NCCL to create and overwrite any previous files of that name.

Note: If the filename is not unique across all the job processes, then the output may be lost or corrupted.

### NCCL\_DEBUG\_SUBSYS

(since 2.3.4)

The `NCCL_DEBUG_SUBSYS` variable allows the user to filter the `NCCL_DEBUG=INFO` output based on subsystems. The value should be a comma separated list of the subsystems to include in the NCCL debug log traces.

Prefixing the subsystem name with ‘^’ will disable the logging for that subsystem.

#### Values accepted

The default value is INIT,BOOTSTRAP,ENV.

Supported subsystem names are INIT (stands for initialization), COLL (stands for collectives), P2P (stands for peer-to-peer), SHM (stands for shared memory), NET (stands for network), GRAPH (stands for topology detection and graph search), TUNING (stands for algorithm/protocol tuning), ENV (stands for environment settings), ALLOC (stands for memory allocations), CALL (standard for function calls), PROXY (stands for the proxy thread operations), NVLS (standard for NVLink SHARP), BOOTSTRAP (stands for early initialization), REG (stands for memory registration), PROFILE (stands for coarse-grained profiling of initialization), RAS (stands for reliability, availability, and serviceability subsystem) and ALL (includes every subsystem).

### NCCL\_DEBUG\_TIMESTAMP\_FORMAT

(since 2.26)

The `NCCL_DEBUG_TIMESTAMP_FORMAT` variable allows the user to change the format used when printing debug log messages.

The time is printed as a local time. This can be changed by setting the `TZ` environment variable. UTC is available by setting `TZ=UTC`. Valid values for TZ look like: `US/Pacific`, `America/Los_Angeles`, etc.

Note that the non-call `TRACE` level of logs continues to print the microseconds since the NCCL debug subsystem was initialized. The `TRACE` logs can also print the strftime formatted timestamp at the beginning if so configured (see `NCCL_DEBUG_TIMESTAMP_LEVELS`).

(since 2.26) Underscores in the format are rendered as spaces.

#### Value accepted

The value of the environment variable is passed to strftime, so any valid format will work here. The default is ` [%F %T]  `, which is ` [YYYY-MM-DD HH:MM:SS]  `. If the value is set, but empty, then no timestamp will be printed (`NCCL_DEBUG_TIMESTAMP_FORMAT=`).

In addition to conversion specifications supported by strftime, `%Xf` can be specified, where `X` is a single numerical digit from 1-9. This will print fractions of a second. The value of `X` indicates how many digits will be printed. For example, `%3f` will print milliseconds. The value is zero padded. For example: ` [%F %T.%9f]  `. (Note that this can only be used once in the format string.)

### NCCL\_DEBUG\_TIMESTAMP\_LEVELS

(since 2.26)

The `NCCL_DEBUG_TIMESTAMP_LEVELS` variable allows the user to set which log lines get a timestamp depending upon the level of the log.

#### Value accepted

The value should be a comma separated list of the levels which should have the timestamp. Valid levels are: `VERSION`, `WARN`, `INFO`, `ABORT`, and `TRACE`. In addition, `ALL` can be used to turn it on for all levels. Setting it to an empty value disables it for all levels. If the value is prefixed with a caret (`^`) then the listed levels will NOT log a timestamp, and the rest will. The default is to enable timestamps for `WARN`, but disable it for the rest.

For example, `NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO,TRACE` will turn it on for warnings, info logs, and traces. Or, `NCCL_DEBUG_TIMESTAMP_LEVELS=^TRACE` will turn them on for everything but traces, which (except call traces) have their own type of timestamp (microseconds since nccl debug initialization).

### NCCL\_COLLNET\_ENABLE

(since 2.6)

Enable the use of the CollNet plugin.

#### Value accepted

Default is 0, define and set to 1 to use the CollNet plugin.

### NCCL\_COLLNET\_NODE\_THRESHOLD

(since 2.9.9)

A threshold for the number of nodes below which CollNet will not be enabled.

#### Value accepted

Default is 2, define and set to an integer.

### NCCL\_CTA\_POLICY

(since 2.29, legacy values since 2.27)

The `NCCL_CTA_POLICY` variable allows the user to set the policy for the NCCL communicator.

#### Value accepted

Set to `DEFAULT` (or `0`, legacy) to use `NCCL_CTA_POLICY_DEFAULT` policy (default). Set to `EFFICIENCY` (or `1`, legacy) to use `NCCL_CTA_POLICY_EFFICIENCY` policy. Set to `ZERO` (or `2`, legacy) to use `NCCL_CTA_POLICY_ZERO` policy.

Set multiple non-legacy policies with the `|` operator.

For more explanation about NCCL policies, please see [NCCL Communicator CTA Policy Flags](api/flags.html#cta-policy-flags).

### NCCL\_NETDEVS\_POLICY

(since 2.28)

The `NCCL_NETDEVS_POLICY` variable allows the user to set the policy for the assignment of network devices to the GPUs. For each GPU, NCCL detects automatically available network devices, taking into account their network bandwidth and the node topology.

#### Value accepted

If set to `AUTO` (default), NCCL also takes into account the other GPUs in the same communicator in order to assign network devices. In specific scenarios, this policy might lead to different GPUs from different communicators sharing the same network devices, and therefore impacts performance.

If set to `MAX:N`, NCCL uses up to N of the network devices available to each GPU. This is intended to be used when device sharing happens with `AUTO` and impacts the performance.

If set to `ALL`, NCCL will use all the available network devices for each GPU, disregarding other GPUs.

### NCCL\_TOPO\_FILE

(since 2.6)

Path to an XML file to load before detecting the topology. By default, NCCL will load `/var/run/nvidia-topologyd/virtualTopology.xml` if present.

#### Value accepted

A path to an accessible file describing part or all of the topology.

### NCCL\_TOPO\_DUMP\_FILE

(since 2.6)

Path to a file to dump the XML topology to after detection.

#### Value accepted

A path to a file which will be created or overwritten.

### NCCL\_SET\_THREAD\_NAME

(since 2.12)

Give more meaningful names to NCCL CPU threads to ease debugging and analysis.

#### Value accepted

0 or 1. Default is 0 (disabled).

## Debugging

These environment variables should be used with caution. New versions of NCCL could work differently and forcing them to a particular value will prevent NCCL from selecting the best setting automatically. They can therefore cause performance problems in the long term, or even break some functionality.

They are fine to use for experiments, or to debug a problem, but should generally not be set for production code.

### NCCL\_P2P\_DISABLE

The `NCCL_P2P_DISABLE` variable disables the peer to peer (P2P) transport, which uses CUDA direct access between GPUs, using NVLink or PCI.

#### Values accepted

Define and set to 1 to disable direct GPU-to-GPU (P2P) communication.

### NCCL\_P2P\_LEVEL

(since 2.3.4)

The `NCCL_P2P_LEVEL` variable allows the user to finely control when to use the peer to peer (P2P) transport between GPUs. The level defines the maximum distance between GPUs where NCCL will use the P2P transport. A short string representing the path type should be used to specify the topographical cutoff for using the P2P transport.

If this isn’t specified, NCCL will attempt to optimally select a value based on the architecture and environment it’s run in.

#### Values accepted

  - LOC : Never use P2P (always disabled)

  - NVL : Use P2P when GPUs are connected through NVLink

  - PIX : Use P2P when GPUs are on the same PCI switch.

  - PXB : Use P2P when GPUs are connected through PCI switches (potentially multiple hops).

  - PHB : Use P2P when GPUs are on the same NUMA node. Traffic will go through the CPU.

  - SYS : Use P2P between NUMA nodes, potentially crossing the SMP interconnect (e.g. QPI/UPI).

#### Integer Values (Legacy)

There is also the option to declare `NCCL_P2P_LEVEL` as an integer corresponding to the path type. These numerical values were kept for retro-compatibility, for those who used numerical values before strings were allowed.

Integer values are discouraged due to breaking changes in path types - the literal values can change over time. To avoid headaches debugging your configuration, use string identifiers.

  - LOC : 0

  - PIX : 1

  - PXB : 2

  - PHB : 3

  - SYS : 4

Values greater than 4 will be interpreted as SYS. NVL is not supported using the legacy integer values.

### NCCL\_P2P\_DIRECT\_DISABLE

The `NCCL_P2P_DIRECT_DISABLE` variable forbids NCCL to directly access user buffers through P2P between GPUs of the same process. This is useful when user buffers are allocated with APIs which do not automatically make them accessible to other GPUs managed by the same process and with P2P access.

#### Values accepted

Define and set to 1 to disable direct user buffer access across GPUs.

### NCCL\_SHM\_DISABLE

The `NCCL_SHM_DISABLE` variable disables the Shared Memory (SHM) transports. SHM is used between devices when peer-to-peer cannot happen, therefore, host memory is used. NCCL will use the network (i.e. InfiniBand or IP sockets) to communicate between the CPU sockets when SHM is disabled.

#### Values accepted

Define and set to 1 to disable communication through shared memory (SHM).

### NCCL\_BUFFSIZE

The `NCCL_BUFFSIZE` variable controls the size of the buffer used by NCCL when communicating data between pairs of GPUs.

Use this variable if you encounter memory constraint issues when using NCCL or you think that a different buffer size would improve performance.

#### Values accepted

The default is 4194304 (4 MiB).

Values are integers, in bytes. The recommendation is to use powers of 2. For example, 1024 will give a 1KiB buffer.

### NCCL\_NTHREADS

The `NCCL_NTHREADS` variable sets the number of CUDA threads per CUDA block. NCCL will launch one CUDA block per communication channel.

Use this variable if you think your GPU clocks are low and you want to increase the number of threads.

You can also use this variable to reduce the number of threads to decrease the GPU workload.

#### Values accepted

The default is 512 for recent generation GPUs, and 256 for some older generations.

The values allowed are 64, 128, 256 and 512.

### NCCL\_MAX\_NCHANNELS

(NCCL\_MAX\_NRINGS since 2.0.5, NCCL\_MAX\_NCHANNELS since 2.5.0)

The `NCCL_MAX_NCHANNELS` variable limits the number of channels NCCL can use. Reducing the number of channels also reduces the number of CUDA blocks used for communication, hence the impact on GPU computing resources.

The old `NCCL_MAX_NRINGS` variable (used until 2.4) still works as an alias in newer versions but is ignored if `NCCL_MAX_NCHANNELS` is set.

This environment variable has been superseded by `NCCL_MAX_CTAS` which can also be set programmatically using [ncclCommInitRankConfig](api/comms.html#ncclcomminitrankconfig).

#### Values accepted

Any value above or equal to 1.

### NCCL\_MIN\_NCHANNELS

(NCCL\_MIN\_NRINGS since 2.2.0, NCCL\_MIN\_NCHANNELS since 2.5.0)

The `NCCL_MIN_NCHANNELS` variable controls the minimum number of channels you want NCCL to use. Increasing the number of channels also increases the number of CUDA blocks NCCL uses, which may be useful to improve performance; however, it uses more CUDA compute resources.

This is especially useful when using aggregated collectives on platforms where NCCL would usually only create one channel.

The old `NCCL_MIN_NRINGS` variable (used until 2.4) still works as an alias in newer versions, but is ignored if `NCCL_MIN_NCHANNELS` is set.

This environment variable has been superseded by `NCCL_MIN_CTAS` which can also be set programmatically using [ncclCommInitRankConfig](api/comms.html#ncclcomminitrankconfig).

#### Values accepted

The default is platform dependent. Set to an integer value, up to 12 (up to 2.2), 16 (2.3 and 2.4) or 32 (2.5 and later).

### NCCL\_CHECKS\_DISABLE

(since 2.0.5, deprecated in 2.2.12)

The `NCCL_CHECKS_DISABLE` variable can be used to disable argument checks on each collective call. Checks are useful during development but can increase the latency. They can be disabled to improve performance in production.

#### Values accepted

The default is 0, set to 1 to disable checks.

### NCCL\_CHECK\_POINTERS

(since 2.2.12)

The `NCCL_CHECK_POINTERS` variable enables checking of the CUDA memory pointers on each collective call. Checks are useful during development but can increase the latency.

#### Values accepted

The default is 0, set to 1 to enable checking.

Setting to 1 restores the original behavior of NCCL prior to 2.2.12.

### NCCL\_LAUNCH\_MODE

(since 2.1.0)

The `NCCL_LAUNCH_MODE` variable controls how NCCL launches CUDA kernels.

#### Values accepted

The default value is PARALLEL.

Setting is to GROUP will use cooperative groups (CUDA 9.0 and later) for processes managing more than one GPU. This is deprecated in 2.9 and may be removed in future versions.

### NCCL\_IB\_DISABLE

The `NCCL_IB_DISABLE` variable prevents the IB/RoCE transport from being used by NCCL. Instead, NCCL will fall back to using IP sockets.

#### Values accepted

Define and set to 1 to disable the use of InfiniBand Verbs for communication (and force another method, e.g. IP sockets).

### NCCL\_IB\_AR\_THRESHOLD

(since 2.6)

Threshold above which we send InfiniBand data in a separate message which can leverage adaptive routing.

#### Values accepted

Size in bytes, the default value is 8192.

Setting it above NCCL\_BUFFSIZE will disable the use of adaptive routing completely.

### NCCL\_IB\_QPS\_PER\_CONNECTION

(since 2.10)

Number of IB queue pairs to use for each connection between two ranks. This can be useful on multi-level fabrics which need multiple queue pairs to have good routing entropy. See `NCCL_IB_SPLIT_DATA_ON_QPS` for different ways to split data on multiple QPs, as it can affect performance.

#### Values accepted

Number between 1 and 128, default is 1.

### NCCL\_IB\_SPLIT\_DATA\_ON\_QPS

(since 2.18)

This parameter controls how we use the queue pairs when we create more than one. Set to 1 (split mode), each message will be split evenly on each queue pair. This may cause a visible latency degradation if many QPs are used. Set to 0 (round-robin mode), queue pairs will be used in round-robin mode for each message we send. Operations which do not send multiple messages will not use all QPs.

#### Values accepted

0 or 1. Default is 0 (since NCCL 2.20). Setting it to 1 will enable split mode (default in 2.18 and 2.19).

### NCCL\_IB\_CUDA\_SUPPORT

(removed in 2.4.0, see NCCL\_NET\_GDR\_LEVEL)

The `NCCL_IB_CUDA_SUPPORT` variable is used to force or disable the usage of GPU Direct RDMA. By default, NCCL enables GPU Direct RDMA if the topology permits it. This variable can disable this behavior or force the usage of GPU Direct RDMA in all cases.

#### Values accepted

Define and set to 0 to disable GPU Direct RDMA.

Define and set to 1 to force the usage of GPU Direct RDMA.

### NCCL\_IB\_PCI\_RELAXED\_ORDERING

(since 2.12)

Enable the use of Relaxed Ordering for the IB Verbs transport. Relaxed Ordering can greatly help the performance of InfiniBand networks in virtualized environments.

#### Values accepted

Set to 2 to automatically use Relaxed Ordering if available. Set to 1 to force the use of Relaxed Ordering and fail if not available. Set to 0 to disable the use of Relaxed Ordering. Default is 2.

### NCCL\_IB\_ADAPTIVE\_ROUTING

(since 2.16)

Enable the use of Adaptive Routing capable data transfers for the IB Verbs transport. Adaptive routing can improve the performance of communications at scale. A system defined Adaptive Routing enabled SL has to be selected accordingly (cf. `NCCL_IB_SL`).

#### Values accepted

Enabled (1) by default on IB networks. Disabled (0) by default on RoCE networks. Set to 1 to force use of Adaptive Routing capable data transmission.

### NCCL\_IB\_ECE\_ENABLE

(since 2.23)

Enable the use of Enhanced Connection Establishment (ECE) on IB/RoCE Verbs networks. ECE can be used to enable advanced networking features such as Congestion Control, Adaptive Routing and Selective Repeat. Note: These parameters are not interpreted or controlled by NCCL and are passed through directly to the HCAs via the ECE mechanism.

#### Values accepted

Enabled (1) by default (since 2.19). Set to 0 to disable use of ECE network capabilities.

Note: Incorrect configuration of the ECE parameters on a system can adversely affect NCCL performance. Administrators should ensure ECE is correctly configured if it is enabled at the system level.

### NCCL\_MEM\_SYNC\_DOMAIN

(since 2.16)

Sets the default Memory Sync Domain for NCCL kernels (CUDA 12.0 & sm90 and later). Memory Sync Domains can help eliminate interference between the NCCL kernels and the application compute kernels, when they use different domains.

#### Values accepted

Default value is `cudaLaunchMemSyncDomainRemote` (1). Currently supported values are 0 and 1.

### NCCL\_CUMEM\_ENABLE

(since 2.18)

Use CUDA cuMem\* functions to allocate memory in NCCL.

#### Values accepted

0 or 1. Default is 0 in 2.18 (disabled); since 2.19 this feature is auto-enabled by default if the system supports it (NCCL\_CUMEM\_ENABLE can still be used to override the autodetection).

### NCCL\_CUMEM\_HOST\_ENABLE

(since 2.23)

Use CUDA cuMem\* functions to allocate host memory in NCCL. See [Shared memory](troubleshooting.html#cumem-host-allocations) for more information.

#### Values accepted

0 or 1. Default is 0 in 2.23; since 2.24, default is 1 if CUDA driver \>= 12.6, CUDA runtime \>= 12.2, and cuMem host allocations are supported.

### NCCL\_NET\_GDR\_LEVEL (formerly NCCL\_IB\_GDR\_LEVEL)

(since 2.3.4. In 2.4.0, NCCL\_IB\_GDR\_LEVEL was renamed to NCCL\_NET\_GDR\_LEVEL)

The `NCCL_NET_GDR_LEVEL` variable allows the user to finely control when to use GPU Direct RDMA between a NIC and a GPU. The level defines the maximum distance between the NIC and the GPU. A string representing the path type should be used to specify the topographical cutoff for GpuDirect.

If this isn’t specified, NCCL will attempt to optimally select a value based on the architecture and environment it’s run in.

#### Values accepted

  - LOC : Never use GPU Direct RDMA (always disabled).

  - PIX : Use GPU Direct RDMA when GPU and NIC are on the same PCI switch.

  - PXB : Use GPU Direct RDMA when GPU and NIC are connected through PCI switches (potentially multiple hops).

  - PHB : Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. Traffic will go through the CPU.

  - SYS : Use GPU Direct RDMA even across the SMP interconnect between NUMA nodes (e.g., QPI/UPI) (always enabled).

#### Integer Values (Legacy)

There is also the option to declare `NCCL_NET_GDR_LEVEL` as an integer corresponding to the path type. These numerical values were kept for retro-compatibility, for those who used numerical values before strings were allowed.

Integer values are discouraged due to breaking changes in path types - the literal values can change over time. To avoid headaches debugging your configuration, use string identifiers.

  - LOC : 0

  - PIX : 1

  - PXB : 2

  - PHB : 3

  - SYS : 4

Values greater than 4 will be interpreted as SYS.

### NCCL\_NET\_GDR\_C2C

(since 2.26)

The `NCCL_NET_GDR_C2C` variable enables GPU Direct RDMA when sending data via a NIC attached to a CPU (i.e. distance PHB) where the CPU is connected to the GPU via a C2C interconnect. This effectively overrides the `NCCL_NET_GDR_LEVEL` setting for this particular NIC.

#### Values accepted

0 or 1. Define and set to 1 to use GPU Direct RDMA to send data to the NIC directly via C2C connected CPUs.

The default value was 0 in 2.26. The default value is 1 since 2.27.

### NCCL\_NET\_GDR\_READ

The `NCCL_NET_GDR_READ` variable enables GPU Direct RDMA when sending data as long as the GPU-NIC distance is within the distance specified by `NCCL_NET_GDR_LEVEL`. Before 2.4.2, GDR read is disabled by default, i.e. when sending data, the data is first stored in CPU memory, then goes to the InfiniBand card. Since 2.4.2, GDR read is enabled by default for NVLink-based platforms.

Note: Reading directly from GPU memory when sending data is known to be slightly slower than reading from CPU memory on some platforms, such as PCI-E.

#### Values accepted

0 or 1. Define and set to 1 to use GPU Direct RDMA to send data to the NIC directly (bypassing CPU).

Before 2.4.2, the default value is 0 for all platforms. Since 2.4.2, the default value is 1 for NVLink-based platforms and 0 otherwise.

### NCCL\_NET\_SHARED\_BUFFERS

(since 2.8)

Allows the usage of shared buffers for inter-node point-to-point communication. This will use a single large pool for all remote peers, having a constant memory usage instead of increasing linearly with the number of remote peers.

#### Value accepted

Default is 1 (enabled). Set to 0 to disable.

### NCCL\_NET\_SHARED\_COMMS

(since 2.12)

Reuse the same connections in the context of PXN. This allows for message aggregation but can also decrease the entropy of network packets.

#### Value accepted

Default is 1 (enabled). Set to 0 to disable.

### NCCL\_SINGLE\_RING\_THRESHOLD

(since 2.1, removed in 2.3)

The `NCCL_SINGLE_RING_THRESHOLD` variable sets the limit under which NCCL will only use one ring. This will limit bandwidth but improve latency.

#### Values accepted

The default value is 262144 (256kB) on GPUs with compute capability 7 and above. Otherwise, the default value is 131072 (128kB).

Values are integers, in bytes.

### NCCL\_LL\_THRESHOLD

(since 2.1, removed in 2.5)

The `NCCL_LL_THRESHOLD` variable sets the size limit under which NCCL uses low-latency algorithms.

#### Values accepted

The default is 16384 (up to 2.2) or is dependent on the number of ranks (2.3 and later).

Values are integers, in bytes.

### NCCL\_TREE\_THRESHOLD

(since 2.4, removed in 2.5)

The `NCCL_TREE_THRESHOLD` variable sets the size limit under which NCCL uses tree algorithms instead of rings.

#### Values accepted

The default is dependent on the number of ranks.

Values are integers, in bytes.

### NCCL\_ALGO

(since 2.5)

The `NCCL_ALGO` variable defines which algorithms NCCL will use.

#### Values accepted

(since 2.5)

Comma-separated list of algorithms (not case sensitive) among:

| Version     | Algorithm     |
| ----------- | ------------- |
| 2.5+        | Ring          |
| 2.5+        | Tree          |
| 2.5 to 2.13 | Collnet       |
| 2.14+       | CollnetChain  |
| 2.14+       | CollnetDirect |
| 2.17+       | NVLS          |
| 2.18+       | NVLSTree      |
| 2.23+       | PAT           |

NVLS and NVLSTree enable NVLink SHARP offload.

To specify algorithms to exclude (instead of include), start the list with `^`.

(since 2.24)

The accepted values are expanded to allow more flexibility, and parsing will issue a warning and fail if an unexpected token is found. Also, if `ring` is not specified as a valid algorithm then it will not implicitly fall back to `ring` if there is no other valid algorithm for the function. Instead, it will fail.

The format is now a semicolon-separated list of pairs of function name and list of algorithms, where the function name is optional for the first entry. If not present, then it applies to all functions not later listed. A colon separates the function (when present) and the comma-separated list of algorithms. Also, if the first character of the comma-separated list of algorithms is a caret (`^`), then all the selections are inverted.

For example, `NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"` Will enable ring and collnetdirect for all functions, then enable tree and collnetdirect for allreduce and ring for broadcast.

And, `NCCL_ALGO=allreduce:^tree` will allow the default (all algorithms available) for all the functions except allreduce, which will have all algorithms available except tree.

The default is unset, which causes NCCL to automatically choose the available algorithms based on the node topology and architecture.

### NCCL\_PROTO

(since 2.5)

The `NCCL_PROTO` variable defines which protocol(s) NCCL will be allowed to use.

Users are discouraged from setting this variable, with the exception of disabling a specific protocol in case a bug in NCCL is suspected. In particular, enabling LL128 on platforms that don’t support it can lead to data corruption.

#### Values accepted

(since 2.5) Comma-separated list of protocols (not case sensitive) among: `LL`, `LL128`, and `Simple`. To specify protocols to exclude (instead of to include), start the list with `^`.

The default behavior enables all supported algorithms: equivalent to `LL,LL128,Simple` on platforms which support LL128, and `LL,Simple` otherwise.

(since 2.24) The accepted values are expanded to allow more flexibility, just as decribed for `NCCL_ALGO` above, allowing the user to specify protocols for each function.

### NCCL\_NVB\_DISABLE

(since 2.11)

Disable intra-node communication through NVLink via an intermediate GPU.

#### Value accepted

Default is 0, set to 1 to disable this mechanism.

### NCCL\_PXN\_DISABLE

(since 2.12)

Disable inter-node communication using a non-local NIC, using NVLink and an intermediate GPU.

#### Value accepted

Default is 0, set to 1 to disable this mechanism.

### NCCL\_P2P\_PXN\_LEVEL

(since 2.12)

Control in which cases PXN is used for send/receive operations.

#### Value accepted

A value of 0 will disable the use of PXN for send/receive. A value of 1 will enable the use of PXN when the NIC preferred by the destination is not accessible through PCI switches. A value of 2 (default) will cause PXN to always be used, even if the NIC is connected through PCI switches, storing data from all GPUs within the node on an intermediate GPU to maximize aggregation.

### NCCL\_PXN\_C2C

(since 2.27)

Allow NCCL to use the PXN mechanism if the peer GPU is connected through C2C + PCIe to the targeted NIC.

#### Value accepted

Default is 1 (since NCCL 2.28; it was 0 in NCCL 2.27). Set to 1 to enable and to 0 to disable.

### NCCL\_RUNTIME\_CONNECT

(since 2.22)

Dynamically connect peers during runtime (e.g., calling ncclAllreduce()) instead of init stage.

#### Value accepted

Default is 1, set to 0 to connect peers at init stage.

### NCCL\_GRAPH\_REGISTER

(since 2.11)

Enable user buffer registration when NCCL calls are captured by CUDA Graphs.

Effective only when: (i) the CollNet algorithm is being used; (ii) all GPUs within a node have P2P access to each other; (iii) there is at most one GPU per process.

User buffer registration may reduce the number of data copies between user buffers and the internal buffers of NCCL. The user buffers will be automatically de-registered when the CUDA Graphs are destroyed.

#### Value accepted

0 or 1. Default value is 1 (enabled).

### NCCL\_LOCAL\_REGISTER

(since 2.19)

Enable user local buffer registration when users explicitly call *ncclCommRegister*.

#### Value accepted

0 or 1. Default value is 1 (enabled).

### NCCL\_LEGACY\_CUDA\_REGISTER

(since 2.24)

Cuda buffers allocated through *cudaMalloc* (and related memory allocators) are legacy buffers. Registering legacy buffer can cause implicit synchronization, which is unsafe and can possibly cause a hang for NCCL. NCCL disables legacy buffer registration by default, and users should move to cuMem-based memory allocators for buffer registration.

#### Value accepted

0 or 1. Default value is 0 (disabled).

### NCCL\_WIN\_ENABLE

(since 2.27)

Enable window memory registration.

#### Value accepted

0 or 1. Default value is 1 (enabled).

### NCCL\_SET\_STACK\_SIZE

(since 2.9)

Set CUDA kernel stack size to the maximum stack size amongst all NCCL kernels.

It may avoid a CUDA memory reconfiguration on load. Set to 1 if you experience hang due to CUDA memory reconfiguration.

#### Value accepted

0 or 1. Default value is 0 (disabled).

### NCCL\_GRAPH\_MIXING\_SUPPORT

(since 2.13)

Enable/disable support for multiple outstanding NCCL calls from parallel CUDA graphs or a CUDA graph and non-captured NCCL calls. NCCL calls are considered outstanding starting from their host-side launch (e.g., a call to ncclAllreduce() for non-captured calls or cudaGraphLaunch() for captured calls) and ending when the device kernel execution completes. With graph mixing support disabled, the following use cases are NOT supported:

1.  Using a NCCL communicator (or split-shared communicators) from parallel graph launches, where parallel means on different streams without dependencies that would serialize their execution.

2.  Launching a non-captured NCCL collective during an outstanding graph launch that uses the same communicator (or split-shared communicators), regardless of stream ordering.

The ability to disable support is motivated by observed hangs in the CUDA launches when support is enabled and multiple ranks have work launched via cudaGraphLaunch from the same thread.

#### Value accepted

0 or 1. Default is 1 (enabled).

### NCCL\_DMABUF\_ENABLE

(since 2.13)

Enable GPU Direct RDMA buffer registration using the Linux dma-buf subsystem.

The Linux dma-buf subsystem allows GPU Direct RDMA capable NICs to read and write CUDA buffers directly without CPU involvement.

#### Value accepted

0 or 1. Default value is 1 (enabled), but the feature is automatically disabled if the Linux kernel or the CUDA/NIC driver do not support it.

### NCCL\_P2P\_NET\_CHUNKSIZE

(since 2.14)

The `NCCL_P2P_NET_CHUNKSIZE` controls the size of messages sent through the network for ncclSend/ncclRecv operations.

#### Values accepted

The default is 131072 (128 K).

Values are integers, in bytes. The recommendation is to use powers of 2, hence 262144 would be the next value.

### NCCL\_P2P\_LL\_THRESHOLD

(since 2.14)

The `NCCL_P2P_LL_THRESHOLD` is the maximum message size that NCCL will use the LL protocol for P2P operations.

#### Values accepted

Decimal number. Default is 16384.

### NCCL\_ALLOC\_P2P\_NET\_LL\_BUFFERS

(since 2.14)

`NCCL_ALLOC_P2P_NET_LL_BUFFERS` instructs communicators to allocate dedicated LL buffers for all P2P network connections. This enables all ranks to use the LL protocol for latency-bound send and receive operations below `NCCL_P2P_LL_THRESHOLD` sizes. Intranode P2P transfers always have dedicated LL buffers allocated. If running all-to-all workloads with high numbers of ranks, this will result in a high scaling memory overhead.

#### Values accepted

0 or 1. Default value is 0 (disabled).

### NCCL\_COMM\_BLOCKING

(since 2.14)

The `NCCL_COMM_BLOCKING` variable controls whether NCCL calls are allowed to block or not. This includes all calls to NCCL, including init/finalize functions, as well as communication functions which may also block due to the lazy initialization of connections for send/receive calls. Setting this environment variable will override the `blocking` configuration in all communicators (see [ncclConfig\_t](api/types.html#ncclconfig)); if not set (undefined), communicator behavior will be determined by the configuration; if not passing configuration, communicators are blocking.

#### Values accepted

0 or 1. 1 indicates blocking communicators, and 0 indicates nonblocking communicators. The default value is undefined.

### NCCL\_CGA\_CLUSTER\_SIZE

(since 2.16)

Set CUDA Cooperative Group Array (CGA) cluster size. On sm90 and later we have an extra level of hierarchy where we can group together several blocks within the Grid, called Thread Block Clusters. Setting this to non-zero will cause NCCL to launch the communication kernels with the Cluster Dimension attribute set accordingly. Setting this environment variable will override the `cgaClusterSize` configuration in all communicators (see [ncclConfig\_t](api/types.html#ncclconfig)); if not set (undefined), CGA cluster size will be determined by the configuration; if not passing configuration, NCCL will automatically choose the best value.

#### Values accepted

0 to 8. Default value is undefined.

### NCCL\_MAX\_CTAS

(since 2.17)

Set the maximal number of CTAs the NCCL should use. Setting this environment variable will override the `maxCTAs` configuration in all communicators (see [ncclConfig\_t](api/types.html#ncclconfig)); if not set (undefined), maximal CTAs will be determined by the configuration; if not passing configuration, NCCL will automatically choose the best value.

#### Values accepted

Set to a positive integer value up to 64 (32 prior to 2.25). Default value is undefined.

### NCCL\_MIN\_CTAS

(since 2.17)

Set the minimal number of CTAs the NCCL should use. Setting this environment variable will override the `minCTAs` configuration in all communicators (see [ncclConfig\_t](api/types.html#ncclconfig)); if not set (undefined), minimal CTAs will be determined by the configuration; if not passing configuration, NCCL will automatically choose the best value.

#### Values accepted

Set to a positive integer value up to 64 (32 prior to 2.25). Default value is undefined.

### NCCL\_NVLS\_ENABLE

(since 2.17)

Enable the use of NVLink SHARP (NVLS). NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures, allowing collectives such as `ncclAllReduce` to be offloaded to the NVSwitch domain. The default value is 2.

#### Values accepted

0: Disable the use of NVLink SHARP. No NVLink SHARP resources will be allocated.

1: Enable NVLink SHARP. NCCL initialization will not fail if NVLink SHARP is not supported on a given system, but it will fail if the NVLink SHARP resources cannot be allocated.

2: (Default) Same as 1: not failing if there is no support, but failing if resources could not be allocated. In versions 2.27 and 2.28, NCCL tried to silently fall back on other transports if NVLS resource allocation failed. This, however, could result in a hang if the allocation failed on just some of the ranks, so in 2.29 the behavior reverted to pre-2.27 (i.e., same as 1). Future versions may re-enable the silent fallback if it can be made to work reliably.

### NCCL\_IB\_MERGE\_NICS

(since 2.20)

Enable NCCL to combine dual-port IB NICs into a single logical network device. This allows NCCL to more easily aggregate dual-port NIC bandwidth.

#### Values accepted

Default is 1 (enabled), define and set to 0 to disable NIC merging

### NCCL\_MNNVL\_ENABLE

(since 2.21)

Enable NCCL to use Multi-Node NVLink (MNNVL) when available. If the system or driver are not Multi-Node NVLink capable then MNNVL will automatically be disabled. This feature also requires NCCL CUMEM support (`NCCL_CUMEM_ENABLE`) to be enabled. MNNVL requires a fully configured and operational IMEX domain for all the nodes that form the NVLink domain. See the CUDA documentation for more details on IMEX domains.

#### Values accepted

0: Disable MNNVL support.

1: Enable MNNVL support. NCCL initialization will fail if MNNVL is not supported or cannot be enabled.

2: Automatic detection of MNNVL support. Will *not* fail if MNNVL is unsupported or if MNNVL resources cannot be allocated.

### NCCL\_MNNVL\_UUID

(since 2.25) Can be used to set the Multi-Node NVLink (MNNVL) UUID to a user defined value. The supplied value will be assigned to both the upper and lower 64-bit words of the 128-bit UUID. Normally the MNNVL UUID is assigned by the Fabric Manager, and it should not need to be overridden.

#### Values accepted

64-bit integer value.

### NCCL\_MNNVL\_CLIQUE\_ID

(since 2.25) Can be used to set the Multi-Node NVLink (MNNVL) Clique Id to a user defined value. Normally the Clique Id is assigned by the Fabric Manager, but this environment variable can be used to “soft” partition MNNVL jobs. i.e. NCCL will only treat ranks with the same \<UUID,CLIQUE\_ID\> as being part of the same NVLink domain.

#### Values accepted

32-bit integer value.

### NCCL\_RAS\_ENABLE

(since 2.24)

Enable NCCL’s reliability, availability, and serviceability (RAS) subsystem, which can be used to query the health of NCCL jobs during execution (see [RAS](troubleshooting/ras.html)).

#### Values accepted

Default is 1 (enabled); define and set to 0 to disable RAS.

### NCCL\_RAS\_ADDR

(since 2.24)

Specify the IP address and port number of a socket that the RAS subsystem will listen on for client connections. RAS can share this socket between multiple processes but that would not be desirable if multiple independent NCCL jobs share a single node (and if those jobs belong to different users, the OS will not allow the socket to be shared). In such cases, each job should be started with a different value (e.g., `localhost:12345`, `localhost:12346`, etc.). Since `localhost` is normally used, only those with access to the nodes where the job is running can connect to the socket. If desired, the address of an externally accessible network interface can be specified instead, which will make RAS accessible from other nodes (such as a cluster’s head node), but that has security implications that should be considered.

#### Values accepted

Default is `localhost:28028`. Either a host name or an IP address can be used for the first part; an IPv6 address needs to be enclosed in square brackets (e.g., `[::1]`).

### NCCL\_RAS\_TIMEOUT\_FACTOR

(since 2.24)

Specify the multiplier factor to apply to all the timeouts of the RAS subsystem. RAS relies on multiple timeouts, ranging from 5 to 60 seconds, to determine the state of the application and to maintain its internal communication, with complex interdependecies between different timeouts. This variable can be used to scale up all these timeouts in a safe, consistent manner, should any of the defaults turn out to be too small; e.g., if the NCCL application is subject to high-overhead debugging/tracing/etc., which makes its execution less predictable. If one wants to use the `ncclras` client in such circumstances, its timeout may need to be increased as well (or disabled).

#### Values accepted

Default is 1; define and set to larger values to increase the timeouts.

### NCCL\_LAUNCH\_ORDER\_IMPLICIT

(since 2.26)

Implicitly order NCCL operations from different communicators on the same device using the host program order. This ensures the operations will not deadlock. When the CUDA runtime and driver are 12.3+, overlapped execution is permitted. On older CUDA versions the operations will be serialized.

#### Values accepted

Default is 0 (disabled); set to 1 to enable.

### NCCL\_LAUNCH\_RACE\_FATAL

(since 2.26)

Attempt to catch host threads racing to launch to the same device and if so return a fatal error. Such a race would violate the determinacy of the program order relied upon by NCCL\_LAUNCH\_ORDER\_IMPLICIT.

#### Values accepted

Default is 1 (enabled); set to 0 to disable.

### NCCL\_IPC\_USE\_ABSTRACT\_SOCKET

(since 2.29)

Use the Linux Abstract Socket mechanism when creating Unix Domain Sockets (UDS) for intra-node CUDA IPC handle exchange. This is enabled by default, but having it enabled can prevent intra-node GPU communication when using multiple containers in certain situations (e.g. different network namespaces).

#### Values accepted

Default is 1 (enabled); set to 0 to disable.

---

# Troubleshooting

Ensure you are familiar with the following known issues and useful debugging strategies.

## Errors

NCCL calls may return a variety of return codes. Ensure that the return codes are always equal to ncclSuccess. If any call fails and returns a value different from ncclSuccess, setting NCCL\_DEBUG to “WARN” will make NCCL print an explicit warning message before returning the error.

Errors are grouped into different categories.

  - ncclUnhandledCudaError and ncclSystemError indicate that a call to an external library failed.

  - ncclInvalidArgument and ncclInvalidUsage indicates there was a programming error in the application using NCCL.

In either case, refer to the NCCL warning message to understand how to resolve the problem.

## RAS

Starting with version 2.24, NCCL includes a reliability, availability, and serviceability (RAS) subsystem to help with the diagnosis and debugging of crashes and hangs.

  - [RAS](troubleshooting/ras.html)
      - [Principle of Operation](troubleshooting/ras.html#principle-of-operation)
      - [RAS Queries](troubleshooting/ras.html#ras-queries)
      - [Sample Output](troubleshooting/ras.html#sample-output)
      - [JSON Output](troubleshooting/ras.html#json-output)
      - [Monitoring Mode](troubleshooting/ras.html#monitoring-mode)

## GPU Direct

NCCL heavily relies on GPU Direct for inter-GPU communication. This refers to the ability for a GPU to directly communicate with another device, such as another GPU or a network card, using direct point-to-point PCI messages.

Direct point-to-point PCI messages can fail or perform poorly for a variety of reasons, like missing components, a bad configuration of a virtual machine or a container, or some BIOS settings.

### GPU-to-GPU communication

To make sure GPU-to-GPU communication is working correctly, look for the `p2pBandwidthLatencyTest` from the CUDA samples found here: <https://github.com/nvidia/cuda-samples>

    cd cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
    make
    ./p2pBandwidthLatencyTest

The test should run to completion and report good performance between GPUs.

Another tool for checking GPU-to-GPU performance is called `nvbandwidth`. This can be downloaded and built from the code and instructions found here: <https://github.com/NVIDIA/nvbandwidth>

### GPU-to-NIC communication

GPUs can also communicate directly with network cards using GPU Direct RDMA (GDRDMA). This requires having a compatible network cards and drivers, plus loading an extra kernel module called `nvidia-peermem`. The `nvidia-peermem` module is now supplied with the CUDA drivers, however it must be loaded on each node boot with:

    sudo modprobe nvidia-peermem

GDRDMA can also be enabled by using the DMA-BUF feature of recent Linux kernels combined with the Open Source Nvidia GPU driver. In this case, NCCL will automatically detect and enable DMA-BUF so the nvidia-peermem module will not be necessary.

### PCI Access Control Services (ACS)

**Baremetal systems**

IO virtualization (also known as VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI point-to-point traffic to the CPU root complex, causing a significant performance reduction or even a hang. You can check whether ACS is enabled on PCI bridges by running:

    sudo lspci -vvv | grep ACSCtl

If lines show “SrcValid+”, then ACS might be enabled. Looking at the full output of lspci, one can check if a PCI bridge has ACS enabled.

    sudo lspci -vvv

If PCI switches have ACS enabled, it needs to be disabled. On some systems this can be done from the BIOS by disabling IO virtualization or VT-d. For Broadcom PLX devices, it can be done from the OS but needs to be done again after each reboot.

Use the command below to find the PCI bus IDs of PLX PCI bridges:

    sudo lspci | grep PLX

Next, use setpci to disable ACS with the command below, replacing 03:00.0 by the PCI bus ID of each PCI bridge.

    sudo setpci -s 03:00.0 ECAP_ACS+0x6.w=0000

Or you can use a script similar to this:

    for BDF in `lspci -d "*:*:*" | awk '{print $1}'`; do
      # skip if it doesn't support ACS
      sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
      if [ $? -ne 0 ]; then
        continue
      fi
      sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w=0000
    done

**Virtual machines**

Virtual machines require ACS to function, hence disabling ACS is not an option. To run with maximum performance inside virtual machines, ATS needs to be enabled in network adapters.

## Topology detection

NCCL relies on /sys to discover the PCI topology of GPUs and network cards. When running inside a virtual machine or container, make sure /sys is properly mounted. Having /sys expose a virtual PCI topology can result in sub-optimal performance.

## Memory issues

### Shared memory

To communicate between processes and even between threads of a process, NCCL creates shared memory segments, traditionally in /dev/shm. The operating system’s limits on these resources may need to be increased accordingly. Please see your system’s documentation for details.

If insufficient shared memory is available, NCCL will fail to initialize. Running with NCCL\_DEBUG=WARN will show a message similar to this:

    NCCL WARN Error: failed to extend /dev/shm/nccl-03v824 to 4194660 bytes

**Docker**

In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a container, please make sure to adjust the shared memory size inside the container, for example by adding the following arguments to the docker launch command line:

    --shm-size=1g --ulimit memlock=-1

**Systemd**

When running jobs using mpirun or SLURM, systemd may remove files in shared memory when it detects that the corresponding user is not logged in, in an attempt to clean up old temporary files. This can cause NCCL to crash during init with an error like:

    NCCL WARN unlink shared memory /dev/shm/nccl-d5rTd0 failed, error: No such file or directory

Given mpirun and SLURM jobs can run on the node without the user being seen as logged in by systemd, system administrators need to disable that clean-up mechanism, which can be performed by SLURM epilogue scripts instead. To do this, the following line needs to be set in /etc/systemd/logind.conf:

    RemoveIPC=no

Once updated, the daemons should be restarted with:

    sudo systemctl restart systemd-logind

**cuMem host allocations**

Starting with version 2.23, NCCL supports an alternative shared memory mechanism using cuMem host allocations. From NCCL 2.24, if CUDA driver \>= 12.6 and CUDA runtime \>= 12.2, it is enabled by default in favor of /dev/shm.

However, cuMem host allocations rely on correctly configured and working NUMA support, which may not be available in some VM and containerization scenarios. In particular, Docker by default disables NUMA support (it can be enabled by invoking Docker with `--cap-add SYS_NICE`). From version 2.26.5, NCCL checks if cuMem host allocations work and, if needed, automatically falls back to the /dev/shm code. In prior versions, the same outcome can be achieved by manually specifying `NCCL_CUMEM_HOST_ENABLE=0`. We still recommend configuring the underlying system to ensure that cuMem host allocations work, as they provide improved reliability during communicator aborts.

cuMem host allocations may fail on systems without CUDA P2P connectivity if CUDA driver version prior to 13.0 is being used. Furthermore, [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) feature can affect NCCL’s ability to accurately determine the current driver version, resulting in cuMem host allocations being enabled on older drivers than intended. We continue to investigate additional mechanisms to detect such circumstances; in the meantime, use `NCCL_CUMEM_HOST_ENABLE=0` to deactivate this feature if it causes issues.

### Stack size

NCCL’s graph search algorithm is highly recursive and, especially on MNNVL systems where many ranks are reachable via CUDA P2P, may temporarily require more than 2 MB of thread stack during communicator creation. While the default Linux stack size limit (8 MB) is known to be sufficient, we’ve seen crashes if the limit is changed to `unlimited`. Due to an idiosyncracy of GNU libc (see the man page of `pthread_create(3)`), such a setting results in a *decrease* of the stack size of NCCL’s background threads to just 2 MB, which may not be sufficiently large. Use `ulimit -s` in bash to print the current limit; if needed, reset it to 8192 KB using `ulimit -s 8192` (one also needs to ensure that the new setting is propagated to other nodes when launching a multi-node NCCL job). Starting with version 2.28, NCCL queries the default stack size for newly launched threads and, if necessary, changes it to a safe value for the current job. We still recommend that users on affected systems attempt to get the system-wide setting fixed as – however well intentioned – it is a potentially serious misconfiguration that could have negative effects extending beyond NCCL jobs.

### Unified Memory (UVM)

Starting with version 2.23, NCCL utilizes CUDA memory pools to optimize graph capturing. This feature relies on UVM being available. While UVM may not be on by default in some virtual machine (VM) setups, it can typically be enabled through a configuration change.

## Networking issues

### IP Network Interfaces

NCCL auto-detects which network interfaces to use for inter-node communication. If some interfaces are in the UP state but are not able to communicate between nodes, NCCL may try to use them anyway and therefore fail during the init functions or even hang.

For information about how to specify which interfaces to use, see the Environment Variables section, particularly the `NCCL_SOCKET_IFNAME` environment variable.

### IP Ports

NCCL opens TCP ports to connect processes together and exchange connection information. To restrict the range of ports used by NCCL, one can set the `net.ipv4.ip_local_port_range` property of the Linux kernel.

This example shows how to restrict NCCL ports to 50000-51000:

    echo 50000 51000 > /proc/sys/net/ipv4/ip_local_port_range

Or to make this permanent, add a line to /etc/sysctl.conf:

    echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf

Restricting the port range can be useful to open a corresponding range in the firewall, for example on Google Cloud:

    gcloud compute --project=myproject firewall-rules create ncclnet0-ingress --direction=INGRESS --priority=1 --network=ncclnet --action=ALLOW --rules=tcp:50000-51000,22,1024-1039 --destination-ranges=0.0.0.0/0 --target-tags=ncclnet

### InfiniBand

Before running NCCL on InfiniBand, running low-level InfiniBand tests (and in particular the ib\_write\_bw test) can help verify whether the nodes are able to communicate properly.

A common issue seen with InfiniBand is the library not being able to register sufficient pinned memory. In such cases you may see an error like:

    NCCL WARN Call to ibv_create_qp failed

or

    NCCL WARN Call to ibv_reg_mr failed

The solution is to remove the user limits on registering pinned memory. This can be done by adding these lines:

    * soft memlock unlimited
    * hard memlock unlimited

To the `/etc/security/limits.conf` configuration file or equivalent on your Linux distribution.

### RDMA over Converged Ethernet (RoCE)

Before running NCCL on RoCE, running low-level RDMA tests (and in particular the `ib_write_bw` test) can help verify whether the nodes are able to communicate properly.

A common issue seen with RoCE is the incorrect GID Index being selected for the RoCE v2 NICs. This can result in the following error:

    NCCL WARN Call to ibv_modify_qp failed with error Invalid argument

With NCCL 2.21 and later the GID index is dynamically selected, but with prior versions the user would need to run:

    show_gids

And then set `NCCL_IB_GID_INDEX` to the GID INDEX for the RoCE v2 VER GID. With NCCL 2.21 and later releases, this environment variable should *not* be set.

Users may also need to set `NCCL_IB_TC` when using RoCE based networks. Refer to your vendor’s documentation for the values this should be set to.

---

# RAS

Since NCCL 2.24, the reliability, availability, and serviceability (RAS) subsystem can be used to query the health of NCCL jobs during execution. This can help with the diagnosis and debugging of crashes and hangs. RAS is a low-overhead infrastructure that NCCL users and developers can use while the application is running. It provides a global view of the state of the running application and can aide in the detection of outliers such as unresponsive processes. With that information, users can then narrow down on the suspected root cause(s) through other techniques such as interactive debugging, system log analysis, etc.

## Principle of Operation

RAS is built into NCCL and launches during NCCL initialization. It consists of a set of threads (one per process) that establish connections with each other, forming a network that the RAS threads then use to exchange information and monitor each other’s health. In a typical configuration, the RAS network traffic (which uses plain TCP/IP sockets on top of the bootstrap/out-of-band network interface that NCCL uses during initialization) should not compete with the main NCCL traffic (which utilizes RDMA networking). RAS is lightweight and should not interfere with the main NCCL job; as such, it is enabled by default (but see [NCCL\_RAS\_ENABLE](../env.html#env-nccl-ras-enable)).

The RAS threads communicate with each other about any changes to the job configuration; they also exchange regular keep-alive messages. If a NCCL process crashes or hangs, the RAS threads running on other NCCL processes learn about it through the RAS network connections to that process being shut down or becoming unresponsive.

## RAS Queries

The RAS threads also listen for client connections on `localhost`, port `28028` (these defaults can be changed using [NCCL\_RAS\_ADDR](../env.html#env-nccl-ras-addr)). The `ncclras` binary client can be used to connect to that socket and query the RAS subsystem for the current job status, which is then printed to standard output. The client accepts the `-h` and `-p` arguments to specify the host name and port, `-v` to produce a more verbose output in case of problems, and `-t` to specify a different timeout (`5` seconds by default; 0 disables the timeout).

As the client communication protocol is fully text-based, standard networking tools such as telnet or netcat can be used instead of the `ncclras` binary. The relevant commands include `STATUS`, `VERBOSE STATUS` (equivalent to the `ncclras` client’s `-v` argument), and `TIMEOUT <seconds>` (equivalent to `-t`); e.g., `echo verbose status | nc localhost 28028`.

Irrespective of how the query is submitted, the receiving RAS thread sends back the job summary information as well as the summary information about all the NCCL communicators; the latter is collected from all the job’s processes so, for jobs experiencing problems or ones that are particularly large, the response may take several seconds to generate. In case any issues were encountered, additional information is provided.

## Sample Output

This section contains excerpts of the RAS status output. Please note that the exact format and scope of the information being made available varies from release to release; the excerpts are provided for illustrative purposes only. For a more machine-friendly format, see [JSON Output](#ras-json) below.

Here’s an example output from a job that is progressing normally:

    Job summary
    ===========
    
      Nodes  Processes         GPUs  Processes     GPUs
    (total)   per node  per process    (total)  (total)
          4          8            1         32       32

We’ve got a job consisting of 32 GPUs (1 GPU per process) running on 4 nodes.

    Communicators... (0.00s)
    =============
    
    Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
        #  in group  per comm  per node  per comm  in group
        0         8         4         1         4        32   RUNNING      OK

The GPUs are split into 8 communicators, 1 GPU per node. RAS attempts to make the summary output as short as possible by grouping together objects having the same size and other important properties.

For jobs that are actively communicating during the RAS query, the following output can sometimes be observed:

    Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
        #  in group  per comm  per node  per comm  in group
        0         1         4         8        32        32   RUNNING  MISMATCH

The output indicates that there is an inconsistency in the information provided by different communicator ranks. Additional information is printed underneath (in this case it’s in the Warnings section, indicating a potentially lower severity):

    Warnings
    ========
    
    #0-0 (27a079b828ff1a75) MISMATCH
      Communicator ranks have different collective operation counts
      26 ranks have launched up to operation 6650
      6 ranks have launched up to operation 6649
      Rank 0 -- GPU 0 managed by process 483072 on node 172.16.64.210
      Rank 2 -- GPU 2 managed by process 483074 on node 172.16.64.210
      Rank 3 -- GPU 3 managed by process 483075 on node 172.16.64.210
      Rank 4 -- GPU 4 managed by process 483076 on node 172.16.64.210
      Rank 5 -- GPU 5 managed by process 483077 on node 172.16.64.210
      Rank 7 -- GPU 7 managed by process 483079 on node 172.16.64.210

Communicators are referred to using the `#<x>-<y>` identifiers, where `<x>` is the group number from the summary output and `<y>` is the communicator number within the group, both starting with 0 (in this example there is only one (32-GPU) communicator so, unsurprisingly, the identifier is `#0-0`). The identifier is followed by a communicator hash, which is a value that can be found in NCCL’s regular debug output as well, and the rank information. RAS groups together the ranks with the same relevant property (the count of issued collective operations in this case; starting with NCCL 2.26, this is broken down per collective operation type). If a group constitutes an outlier, RAS prints additional information about each group member. By default this is done if the group size is at most 25% of the total *and* the group has no more than 10 members; enabling verbose output relaxes this to under 50% of the total and lifts the group size limit.

The particular case above should not be a cause for concern, as long as the counts increase across repeated queries. NCCL collectives, being optimized for speed, can easily outpace the RAS collective queries, especially if the size of the collectives is fairly small. An application may also exhibit work imbalance, with certain ranks routinely arriving to the collective operations later than others – an experience with a particular workload is needed to determine what’s normal and what’s not. However, if the output does not change across subsequent RAS queries, it may indicate that the communicator is “stuck” for some reason, which could warrant an investigation.

Similar effects can sometimes be observed during communicator initialization or tear-down:

    Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
        #  in group  per comm  per node  per comm  in group
        0         1         4       1-2        32        32  FINALIZE  MISMATCH
        1         7         4         1         4        28   RUNNING      OK
        2         1         4         1         4         4      INIT      OK
    
    [...]
    
    #0-0 (9e17999afaa87dbb) MISMATCH
      Communicator ranks have different status
      26 ranks have status UNKNOWN
      4 ranks have status RUNNING
      Rank 0 -- GPU 0 managed by process 507285 on node 172.16.64.210
      Rank 8 -- GPU 0 managed by process 1598388 on node 172.16.64.212
      Rank 16 -- GPU 0 managed by process 3500071 on node 172.16.64.213
      Rank 24 -- GPU 0 managed by process 2405067 on node 172.16.64.222
      2 ranks have status FINALIZE
      Rank 4 -- GPU 4 managed by process 507289 on node 172.16.64.210
      Rank 20 -- GPU 4 managed by process 3500075 on node 172.16.64.213

The above snapshot depicts a transitional situation as the initial, 32-GPU communicator is being replaced by eight 4-GPU communicators (one of which is still initializing, so it is listed separately (group `#2`) from the already initialized seven (group `#1`)). The 32-GPU communicator (`#0-0`) is being torn down, with two ranks in the middle of ncclCommFinalize, four ranks that have *not* called ncclCommFinalize yet, and the remaining 26 ranks “unknown” – meaning that they didn’t provide any information about that communicator when RAS was collecting data, simply because their call to ncclCommFinalize has already completed so they are in fact no longer that communicator’s members (NCCL 2.26 and later print `NOCOMM` instead). Again, as long as the situation is resolved when the query is repeated, it can be ignored.

Here’s an excerpt from an invocation right after artificially creating a problem with one of the job processes:

    Communicators... (2.05s)
    =============
    
    Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors
        #  in group  per comm  per node  per comm  in group
        0         1         4       7-8        32        32   RUNNING  INCOMPLETE
    
    Errors
    ======
    
    INCOMPLETE
      Missing communicator data from 1 job process
      Process 3487984 on node 172.16.64.213 managing GPU 5
    
    #0-0 (cf264af53edbe986) INCOMPLETE
      Missing communicator data from 1 rank
      The missing rank: 21
    
    Warnings
    ========
    
    TIMEOUT
      Encountered 2 communication timeouts while gathering communicator data

In this case the summary takes a few seconds to generate because RAS waits for the data from the process experiencing problems (the process is unresponsive – it was stopped – but RAS doesn’t know it yet). Repeated queries should be much faster because once RAS determines that a process is unresponsive, it reconfigures the RAS network to route around it.

RAS will attempt to reestablish communication with the unresponsive process; if it’s unable to do so for 60 seconds, it will declare the process dead (permanently):

    Errors
    ======
    
    DEAD
      1 job process is considered dead (unreachable via the RAS network)
      Process 3487984 on node 172.16.64.213 managing GPU 5
    
    #0-0 (cf264af53edbe986) INCOMPLETE
      Missing communicator data from 1 rank
      The missing rank: 21

RAS will simply stop attempting to communicate with such processes over the RAS network anymore, leaving it up to the user to determine if any additional action is warranted.

## JSON Output

Starting with NCCL 2.28.7, RAS can generate output in JSON format to support machine-parsable metrics collection.

The `ncclras` binary gains an additional option `-f` followed by an argument: `text` or `json`, with `text` being the default. The equivalent wire-level protocol command is `SET FORMAT <format>`. Sample output can be found below:

    {
      "nccl_version": "2.29.1",
      "cuda_runtime_version": 13000,
      "cuda_driver_version": 13000,
      "timestamp": "2025-12-19 13:06:53",
      "communicators_count": 1,
      "communicators": [
        {
          "hash": "0xae94423cfbb2ef4a",
          "secondary_hash": "0xb7e7187447156001:0xb8242ed28a71381e",
          "size": 2,
          "ranks_count": 1,
          "missing_ranks_count": 1,
          "ranks": [
            {
              "rank": 0,
              "host": "172.16.64.245",
              "pid": 1524344,
              "cuda_dev": 0,
              "nvml_dev": 0,
              "status": {
                "init_state": 0,
                "async_error": 0,
                "finalize_called": false,
                "destroy_flag": false,
                "abort_flag": false
              },
              "collective_counts": {
                "Broadcast": 0,
                "Reduce": 0,
                "AllGather": 0,
                "ReduceScatter": 0,
                "AllReduce": 0
              }
            }
          ],
          "missing_ranks": [
            {
              "rank": 1,
              "host": "172.16.64.245",
              "pid": 1524345,
              "cuda_dev": 1,
              "nvml_dev": 1,
              "status": {
                "unresponsive": true,
                "considered_dead": false
              }
            }
          ]
        }
      ],
      "ras": {
        "collection_time_sec": 0.000,
        "timeouts_count": 0
      }
    }

As can be observed, the JSON output is considerably more verbose than the text one (which is optimized for human consumption). It is essentially a dump of all raw data collected by RAS; the analysis and interpretation is left to the consumer.

Most of the fields should be fairly self-explanatory, with the possible exception of:

  - `secondary_hash`: can be used in conjunction with `hash` to create a communicator identifier that is guaranteed to be unique (which `hash` by itself is not, although in practice collisions are highly unlikely).

  - `nvml_dev`: normally the same as `cuda_dev`, unless CUDA\_VISIBLE\_DEVICES is being used (in which case `cuda_dev` may not be unique among ranks on a single node, whereas `nvml_dev` always is).

  - `init_state`: `0` (ncclSuccess) if a rank is fully initialized, `7` (ncclInProgress) if initialization is still ongoing; any other value indicates an error.

  - `async_error`: the value that would be returned if the rank called ncclGetAsyncError.

  - `unresponsive`: true if RAS was unable to reach the rank when collecting data – this is the most common reason why a rank would be considered missing. There are, however, corner cases during communicator initialization and termination when a process may be reachable but does not consider itself to be a member of a given communicator.

## Monitoring Mode

Starting with NCCL 2.29, RAS adds a monitoring mode for real-time status updates.

The `ncclras` binary gains an additional option `-m` that switches it to monitoring mode. The equivalent wire-level protocol command is `MONITOR`.

When in monitoring mode, the RAS client prints a welcome message and does not terminate until it is interrupted or the job finishes:

    RAS Monitor Mode - watching for peer changes (Ctrl+C to exit)...
    ================================================================

A sample event of interest could be a process being declared dead:

    [2025-12-19 13:07:07] PEER_DEAD: Process 1524345 on node 172.16.64.245 managing GPU 1

Monitoring mode can also be used in conjunction with JSON output:

    {
      "timestamp": "2025-12-19 13:07:07",
      "group": "LIFECYCLE",
      "event": "PEER_DEAD",
      "peer": {
        "host": "172.16.64.245",
        "pid": 1524345,
        "cuda_devs": [1],
        "nvml_devs": [1]
      },
      "details": ""
    }

Unlike in the previously shown communicator output (where each rank was printed separately), here the entity of concern is a process so `cuda_devs` and `nvml_devs` need to be arrays (since a process can manage multiple GPUs).

---

# Index

[**B**](#B) | [**C**](#C) | [**D**](#D) | [**G**](#G) | [**L**](#L) | [**M**](#M) | [**N**](#N) | [**R**](#R) | [**T**](#T)

## B

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.barrierCount">barrierCount (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## C

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.cudaDev">cudaDev (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## D

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.deviceApiSupport">deviceApiSupport (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## G

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.ginCounterCount">ginCounterCount (C member)</a></li>
</ul></td>
<td><ul>
<li><a href="api/device.html#c.ginSignalCount">ginSignalCount (C member)</a></li>
<li><a href="api/device.html#c.ginType">ginType (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## L

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.lsaBarrierCount">lsaBarrierCount (C member)</a></li>
</ul></td>
<td><ul>
<li><a href="api/device.html#c.lsaMultimem">lsaMultimem (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## M

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.multimemSupport">multimemSupport (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## N

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/flags.html#c.NCCL_CTA_POLICY_DEFAULT">NCCL_CTA_POLICY_DEFAULT (C macro)</a></li>
<li><a href="api/flags.html#c.NCCL_CTA_POLICY_EFFICIENCY">NCCL_CTA_POLICY_EFFICIENCY (C macro)</a></li>
<li><a href="api/flags.html#c.NCCL_CTA_POLICY_ZERO">NCCL_CTA_POLICY_ZERO (C macro)</a></li>
<li><a href="api/device.html#c.NCCL_GIN_TYPE_GDAKI">NCCL_GIN_TYPE_GDAKI (C macro)</a></li>
<li><a href="api/device.html#c.NCCL_GIN_TYPE_NONE">NCCL_GIN_TYPE_NONE (C macro)</a></li>
<li><a href="api/device.html#c.NCCL_GIN_TYPE_PROXY">NCCL_GIN_TYPE_PROXY (C macro)</a></li>
<li><a href="api/flags.html#c.NCCL_SHRINK_ABORT">NCCL_SHRINK_ABORT (C macro)</a></li>
<li><a href="api/flags.html#c.NCCL_SHRINK_DEFAULT">NCCL_SHRINK_DEFAULT (C macro)</a></li>
<li><a href="api/flags.html#c.NCCL_WIN_COLL_SYMMETRIC">NCCL_WIN_COLL_SYMMETRIC (C macro)</a></li>
<li><a href="api/flags.html#c.NCCL_WIN_DEFAULT">NCCL_WIN_DEFAULT (C macro)</a></li>
<li><a href="api/colls.html#c.ncclAllGather">ncclAllGather (C function)</a></li>
<li><a href="api/colls.html#c.ncclAllReduce">ncclAllReduce (C function)</a></li>
<li><a href="api/colls.html#c.ncclAlltoAll">ncclAlltoAll (C function)</a></li>
<li><a href="api/colls.html#c.ncclBcast">ncclBcast (C function)</a></li>
<li><a href="api/colls.html#c.ncclBroadcast">ncclBroadcast (C function)</a></li>
<li><a href="api/types.html#c.ncclComm_t">ncclComm_t (C type)</a></li>
<li><a href="api/comms.html#c.ncclCommAbort">ncclCommAbort (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommCount">ncclCommCount (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommCuDevice">ncclCommCuDevice (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommDeregister">ncclCommDeregister (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommDestroy">ncclCommDestroy (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommFinalize">ncclCommFinalize (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommGetAsyncError">ncclCommGetAsyncError (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommGetUniqueId">ncclCommGetUniqueId (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommGrow">ncclCommGrow (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommInitAll">ncclCommInitAll (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommInitRank">ncclCommInitRank (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommInitRankConfig">ncclCommInitRankConfig (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommInitRankScalable">ncclCommInitRankScalable (C function)</a></li>
<li><a href="api/device.html#c.ncclCommProperties_t">ncclCommProperties_t (C type)</a></li>
<li><a href="api/device.html#c.ncclCommQueryProperties">ncclCommQueryProperties (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommRegister">ncclCommRegister (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommRevoke">ncclCommRevoke (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommShrink">ncclCommShrink (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommSplit">ncclCommSplit (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommUserRank">ncclCommUserRank (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommWindowDeregister">ncclCommWindowDeregister (C function)</a></li>
<li><a href="api/comms.html#c.ncclCommWindowRegister">ncclCommWindowRegister (C function)</a></li>
<li><a href="api/types.html#c.ncclConfig_t">ncclConfig_t (C type)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.blocking">ncclConfig_t.blocking (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.cgaClusterSize">ncclConfig_t.cgaClusterSize (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.collnetEnable">ncclConfig_t.collnetEnable (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.commName">ncclConfig_t.commName (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.CTAPolicy">ncclConfig_t.CTAPolicy (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.maxCTAs">ncclConfig_t.maxCTAs (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.minCTAs">ncclConfig_t.minCTAs (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.NCCL_CONFIG_INITIALIZER">ncclConfig_t.NCCL_CONFIG_INITIALIZER (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.netName">ncclConfig_t.netName (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.nvlsCTAs">ncclConfig_t.nvlsCTAs (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.shrinkShare">ncclConfig_t.shrinkShare (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.splitShare">ncclConfig_t.splitShare (C macro)</a></li>
<li><a href="api/types.html#c.ncclConfig_t.trafficClass">ncclConfig_t.trafficClass (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t">ncclDataType_t (C type)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclBfloat16">ncclDataType_t.ncclBfloat16 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclChar">ncclDataType_t.ncclChar (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclDouble">ncclDataType_t.ncclDouble (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclFloat">ncclDataType_t.ncclFloat (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclFloat16">ncclDataType_t.ncclFloat16 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclFloat32">ncclDataType_t.ncclFloat32 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclFloat64">ncclDataType_t.ncclFloat64 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclFloat8e4m3">ncclDataType_t.ncclFloat8e4m3 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclFloat8e5m2">ncclDataType_t.ncclFloat8e5m2 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclHalf">ncclDataType_t.ncclHalf (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclInt">ncclDataType_t.ncclInt (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclInt32">ncclDataType_t.ncclInt32 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclInt64">ncclDataType_t.ncclInt64 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclInt8">ncclDataType_t.ncclInt8 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclUint32">ncclDataType_t.ncclUint32 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclUint64">ncclDataType_t.ncclUint64 (C macro)</a></li>
<li><a href="api/types.html#c.ncclDataType_t.ncclUint8">ncclDataType_t.ncclUint8 (C macro)</a></li>
<li><a href="api/device.html#c.ncclDevComm">ncclDevComm (C type)</a></li>
<li><a href="api/device.html#c.ncclDevComm.ginContextCount">ncclDevComm.ginContextCount (C member)</a></li>
<li><a href="api/device.html#c.ncclDevComm.lsaRank">ncclDevComm.lsaRank (C member)</a></li>
<li><a href="api/device.html#c.ncclDevComm.lsaSize">ncclDevComm.lsaSize (C member)</a></li>
<li><a href="api/device.html#c.ncclDevComm.nRanks">ncclDevComm.nRanks (C member)</a></li>
<li><a href="api/device.html#c.ncclDevComm.rank">ncclDevComm.rank (C member)</a></li>
<li><a href="api/device.html#c.ncclDevCommCreate">ncclDevCommCreate (C function)</a></li>
<li><a href="api/device.html#c.ncclDevCommDestroy">ncclDevCommDestroy (C function)</a></li>
<li><a href="api/device.html#c.ncclDevCommRequirements">ncclDevCommRequirements (C type)</a></li>
<li><a href="api/colls.html#c.ncclGather">ncclGather (C function)</a></li>
<li><a href="api/comms.html#c.ncclGetErrorString">ncclGetErrorString (C function)</a></li>
<li><a href="api/comms.html#c.ncclGetLastError">ncclGetLastError (C function)</a></li>
<li><a href="api/device.html#_CPPv419ncclGetLocalPointer12ncclWindow_t6size_t">ncclGetLocalPointer (C++ function)</a></li>
</ul></td>
<td><ul>
<li><a href="api/device.html#c.ncclGetLsaDevicePointer">ncclGetLsaDevicePointer (C function)</a></li>
<li><a href="api/device.html#c.ncclGetLsaMultimemDevicePointer">ncclGetLsaMultimemDevicePointer (C function)</a></li>
<li><a href="api/device.html#_CPPv425ncclGetLsaMultimemPointer12ncclWindow_t6size_tRK11ncclDevComm">ncclGetLsaMultimemPointer (C++ function)</a></li>
<li><a href="api/device.html#_CPPv417ncclGetLsaPointer12ncclWindow_t6size_ti">ncclGetLsaPointer (C++ function)</a></li>
<li><a href="api/device.html#c.ncclGetMultimemDevicePointer">ncclGetMultimemDevicePointer (C function)</a></li>
<li><a href="api/device.html#c.ncclGetPeerDevicePointer">ncclGetPeerDevicePointer (C function)</a></li>
<li><a href="api/device.html#_CPPv418ncclGetPeerPointer12ncclWindow_t6size_ti">ncclGetPeerPointer (C++ function)</a></li>
<li><a href="api/comms.html#c.ncclGetUniqueId">ncclGetUniqueId (C function)</a></li>
<li><a href="api/comms.html#c.ncclGetVersion">ncclGetVersion (C function)</a></li>
<li><a href="api/device.html#_CPPv47ncclGin">ncclGin (C++ class)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin5flushE4CoopN4cuda12memory_orderE">ncclGin::flush (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin7ncclGinERK11ncclDevCommi">ncclGin::ncclGin (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE">ncclGin::put (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin11readCounterE16ncclGinCounter_tiN4cuda12memory_orderE">ncclGin::readCounter (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin10readSignalE15ncclGinSignal_tiN4cuda12memory_orderE">ncclGin::readSignal (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin12resetCounterE16ncclGinCounter_t">ncclGin::resetCounter (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin11resetSignalE15ncclGinSignal_t">ncclGin::resetSignal (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE">ncclGin::signal (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin11waitCounterE4Coop16ncclGinCounter_t8uint64_tiN4cuda12memory_orderE">ncclGin::waitCounter (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N7ncclGin10waitSignalE4Coop15ncclGinSignal_t8uint64_tiN4cuda12memory_orderE">ncclGin::waitSignal (C++ function)</a></li>
<li><a href="api/device.html#_CPPv418ncclGin_CounterInc">ncclGin_CounterInc (C++ class)</a></li>
<li><a href="api/device.html#_CPPv4N18ncclGin_CounterInc7counterE">ncclGin_CounterInc::counter (C++ member)</a></li>
<li><a href="api/device.html#_CPPv417ncclGin_SignalAdd">ncclGin_SignalAdd (C++ class)</a></li>
<li><a href="api/device.html#_CPPv4N17ncclGin_SignalAdd6signalE">ncclGin_SignalAdd::signal (C++ member)</a></li>
<li><a href="api/device.html#_CPPv4N17ncclGin_SignalAdd5valueE">ncclGin_SignalAdd::value (C++ member)</a></li>
<li><a href="api/device.html#_CPPv417ncclGin_SignalInc">ncclGin_SignalInc (C++ class)</a></li>
<li><a href="api/device.html#_CPPv4N17ncclGin_SignalInc6signalE">ncclGin_SignalInc::signal (C++ member)</a></li>
<li><a href="api/device.html#_CPPv4I0E21ncclGinBarrierSession">ncclGinBarrierSession (C++ class)</a></li>
<li><a href="api/device.html#_CPPv4N21ncclGinBarrierSession21ncclGinBarrierSessionE4Coop7ncclGin15ncclTeamTagRail8uint32_t">ncclGinBarrierSession::ncclGinBarrierSession (C++ function)</a>, <a href="api/device.html#_CPPv4N21ncclGinBarrierSession21ncclGinBarrierSessionE4Coop7ncclGin8ncclTeam20ncclGinBarrierHandle8uint32_t">[1]</a></li>
<li><a href="api/device.html#_CPPv4N21ncclGinBarrierSession4syncE4CoopN4cuda12memory_orderE17ncclGinFenceLevel">ncclGinBarrierSession::sync (C++ function)</a></li>
<li><a href="api/device.html#_CPPv416ncclGinCounter_t">ncclGinCounter_t (C++ type)</a></li>
<li><a href="api/device.html#_CPPv415ncclGinSignal_t">ncclGinSignal_t (C++ type)</a></li>
<li><a href="api/device.html#c.ncclGinType_t">ncclGinType_t (C type)</a></li>
<li><a href="api/group.html#c.ncclGroupEnd">ncclGroupEnd (C function)</a></li>
<li><a href="api/group.html#c.ncclGroupSimulateEnd">ncclGroupSimulateEnd (C function)</a></li>
<li><a href="api/group.html#c.ncclGroupStart">ncclGroupStart (C function)</a></li>
<li><a href="api/device.html#_CPPv4I0E21ncclLsaBarrierSession">ncclLsaBarrierSession (C++ class)</a></li>
<li><a href="api/device.html#_CPPv4N21ncclLsaBarrierSession6arriveE4CoopN4cuda12memory_orderE">ncclLsaBarrierSession::arrive (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N21ncclLsaBarrierSession21ncclLsaBarrierSessionE4CoopRK11ncclDevComm14ncclTeamTagLsa8uint32_tb">ncclLsaBarrierSession::ncclLsaBarrierSession (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N21ncclLsaBarrierSession4syncE4CoopN4cuda12memory_orderE">ncclLsaBarrierSession::sync (C++ function)</a></li>
<li><a href="api/device.html#_CPPv4N21ncclLsaBarrierSession4waitE4CoopN4cuda12memory_orderE">ncclLsaBarrierSession::wait (C++ function)</a></li>
<li><a href="api/comms.html#c.ncclMemAlloc">ncclMemAlloc (C function)</a></li>
<li><a href="api/comms.html#c.ncclMemFree">ncclMemFree (C function)</a></li>
<li><a href="api/p2p.html#c.ncclPutSignal">ncclPutSignal (C function)</a></li>
<li><a href="api/p2p.html#c.ncclRecv">ncclRecv (C function)</a></li>
<li><a href="api/types.html#c.ncclRedOp_t">ncclRedOp_t (C type)</a></li>
<li><a href="api/types.html#c.ncclRedOp_t.ncclAvg">ncclRedOp_t.ncclAvg (C macro)</a></li>
<li><a href="api/types.html#c.ncclRedOp_t.ncclMax">ncclRedOp_t.ncclMax (C macro)</a></li>
<li><a href="api/types.html#c.ncclRedOp_t.ncclMin">ncclRedOp_t.ncclMin (C macro)</a></li>
<li><a href="api/types.html#c.ncclRedOp_t.ncclProd">ncclRedOp_t.ncclProd (C macro)</a></li>
<li><a href="api/types.html#c.ncclRedOp_t.ncclSum">ncclRedOp_t.ncclSum (C macro)</a></li>
<li><a href="api/ops.html#c.ncclRedOpCreatePreMulSum">ncclRedOpCreatePreMulSum (C function)</a></li>
<li><a href="api/ops.html#c.ncclRedOpDestroy">ncclRedOpDestroy (C function)</a></li>
<li><a href="api/colls.html#c.ncclReduce">ncclReduce (C function)</a></li>
<li><a href="api/colls.html#c.ncclReduceScatter">ncclReduceScatter (C function)</a></li>
<li><a href="api/types.html#c.ncclResult_t">ncclResult_t (C type)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclInProgress">ncclResult_t.ncclInProgress (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclInternalError">ncclResult_t.ncclInternalError (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclInvalidArgument">ncclResult_t.ncclInvalidArgument (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclInvalidUsage">ncclResult_t.ncclInvalidUsage (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclRemoteError">ncclResult_t.ncclRemoteError (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclSuccess">ncclResult_t.ncclSuccess (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclSystemError">ncclResult_t.ncclSystemError (C macro)</a></li>
<li><a href="api/types.html#c.ncclResult_t.ncclUnhandledCudaError">ncclResult_t.ncclUnhandledCudaError (C macro)</a></li>
<li><a href="api/types.html#c.ncclScalarResidence_t">ncclScalarResidence_t (C type)</a></li>
<li><a href="api/types.html#c.ncclScalarResidence_t.ncclScalarDevice">ncclScalarResidence_t.ncclScalarDevice (C macro)</a></li>
<li><a href="api/types.html#c.ncclScalarResidence_t.ncclScalarHostImmediate">ncclScalarResidence_t.ncclScalarHostImmediate (C macro)</a></li>
<li><a href="api/colls.html#c.ncclScatter">ncclScatter (C function)</a></li>
<li><a href="api/p2p.html#c.ncclSend">ncclSend (C function)</a></li>
<li><a href="api/p2p.html#c.ncclSignal">ncclSignal (C function)</a></li>
<li><a href="api/types.html#c.ncclSimInfo_t">ncclSimInfo_t (C type)</a></li>
<li><a href="api/types.html#c.ncclSimInfo_t.estimatedTime">ncclSimInfo_t.estimatedTime (C macro)</a></li>
<li><a href="api/types.html#c.ncclSimInfo_t.NCCL_SIM_INFO_INITIALIZER">ncclSimInfo_t.NCCL_SIM_INFO_INITIALIZER (C macro)</a></li>
<li><a href="api/p2p.html#c.ncclWaitSignal">ncclWaitSignal (C function)</a></li>
<li><a href="api/p2p.html#c.ncclWaitSignalDesc_t">ncclWaitSignalDesc_t (C type)</a></li>
<li><a href="api/p2p.html#c.ncclWaitSignalDesc_t.ctx">ncclWaitSignalDesc_t.ctx (C member)</a></li>
<li><a href="api/p2p.html#c.ncclWaitSignalDesc_t.opCnt">ncclWaitSignalDesc_t.opCnt (C member)</a></li>
<li><a href="api/p2p.html#c.ncclWaitSignalDesc_t.peer">ncclWaitSignalDesc_t.peer (C member)</a></li>
<li><a href="api/p2p.html#c.ncclWaitSignalDesc_t.sigIdx">ncclWaitSignalDesc_t.sigIdx (C member)</a></li>
<li><a href="api/types.html#c.ncclWindow_t">ncclWindow_t (C type)</a></li>
<li><a href="api/types.html#c.nChannelsPerNetPeer">nChannelsPerNetPeer (C macro)</a></li>
<li><a href="api/types.html#c.nChannelsPerNetPeer.graphUsageMode">nChannelsPerNetPeer.graphUsageMode (C macro)</a></li>
<li><a href="api/device.html#c.nRanks">nRanks (C member)</a></li>
<li><a href="api/device.html#c.nvmlDev">nvmlDev (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## R

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.railGinBarrierCount">railGinBarrierCount (C member)</a></li>
</ul></td>
<td><ul>
<li><a href="api/device.html#c.rank">rank (C member)</a></li>
<li><a href="api/device.html#c.resourceRequirementsList">resourceRequirementsList (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

## T

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><ul>
<li><a href="api/device.html#c.teamRequirementsList">teamRequirementsList (C member)</a></li>
</ul></td>
</tr>
</tbody>
</table>

---

Please activate JavaScript to enable the search functionality.
