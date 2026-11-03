"""The scheduler of Caesura inside SGLang: one planning round per step on two tensor-parallel ranks.

Every step, rank 0 snapshots the waiting queue with its prefix matches, runs the planner, and applies the release-aware
rollout and the veto when Caesura is enabled. Both ranks then agree on the schedule and stage it: a scheduled request
prefills a whole number of pages, or its last chunk, and a started prefill that is not scheduled pauses and later
resumes. All running decodes run in every step. When no waiting request fits with its smallest chunk, the decode-only
certificate skips the search, and both ranks reuse the certificate while the queue and resources stay the same. The
numeric exchange (tp_wire) replaces pickled object exchanges between the ranks with fixed-size integer messages.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import hashlib
import json
import logging
import math
import os
import time

from sglang.srt.managers.b300_page_objectives_v14 import (
    PAGE, CostModel, Limits, Row, paged, solve, forced_current_action, completion_guard, merge_locks, usage,
)
from sglang.srt.managers.b300_page_resources_v9 import (
    resource_budget, future_output_guard,
)

LOG = logging.getLogger(__name__)
STATE = None


class Runtime:
    def __init__(self, scheduler):
        import torch
        self.torch = torch
        self.dist = torch.distributed
        self.group = scheduler.tree_cache.tp_group
        if (not self.dist.is_initialized() or scheduler.tree_cache.pp_size != 1
                or self.dist.get_world_size(self.group) != 2
                or str(self.dist.get_backend(self.group)).lower() != 'gloo'):
            raise RuntimeError('the planner requires a TP2 PP1 Gloo cache group')
        self.rank = self.dist.get_rank(self.group)
        self.source = self.dist.get_global_rank(self.group,0)
        self.method = os.environ.get('B300_PAGE_METHOD')
        if self.method != 'b01':
            raise RuntimeError('page method b01 required')
        if (not scheduler.enable_overlap or not scheduler.is_mixed_chunk
                or not scheduler.enable_hierarchical_cache
                or scheduler.page_size != PAGE or scheduler.chunked_prefill_size != 8192
                or scheduler.max_prefill_tokens != 16384
                or scheduler.max_running_requests != 128
                or type(scheduler.tree_cache).__name__ != 'HiRadixCache'
                or scheduler.tree_cache.disable or scheduler.tree_cache.supports_mamba()
                or not scheduler.spec_algorithm.is_none()
                or scheduler.dllm_config is not None
                or getattr(scheduler.disaggregation_mode,'value',None) != 'null'
                or any(getattr(scheduler,k,False) for k in
                       ('enable_lora','enable_hisparse','enable_dynamic_chunking',
                        'enable_hicache_storage','enable_priority_preemption'))):
            raise RuntimeError('unsupported serving configuration for the planner')
        from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS
        self.native_clip = int(CLIP_MAX_NEW_TOKENS)
        scheduler._b300_v9_clip = self.native_clip
        self.native_retracted_total = self.native_recovered_total = 0
        self.native_memory_checks = self.native_recovery_calls = 0
        self.model = CostModel()
        from sglang.srt.managers.b300_page_cache import MetaCache
        from sglang.srt.managers.b300_page_protocol import synchronize_root
        self.meta_cache = MetaCache(scheduler.tree_cache, page=PAGE)
        self.root_sync = synchronize_root
        self.search_budget_s = float(os.environ.get('B300_PAGE_SEARCH_WALL_S', '.3'))
        if not math.isfinite(self.search_budget_s) or not .001 <= self.search_budget_s <= .3:
            raise RuntimeError('invalid planner search budget')
        # Caesura's rollout, veto and numeric exchange on top of the planner; none for the planner alone.
        self.configure_b01_ext(_b01_ext_from_environment(self.method))
        self.planning_allowance_s = .008
        self.geometry_signature = self.geometry_digest = None
        self.resource_certificate = None
        self.resource_certificate_hits = 0
        self.early_resource_certificate = None
        self.early_resource_certificate_hits = 0
        self.rounds = self.partial_chunks = self.pauses = self.resumes = 0
        self.cpu_s = self.max_cpu_s = 0.
        self.last_launch_s = None
        self.last_duration_s = 0.
        LOG.info('B300_PAGE_SCHEDULER_INIT method=%s TP2 page=128 search_wall_s=%s b01_ext=%s',
                 self.method,self.search_budget_s,self.b01_ext)

    def configure_b01_ext(self, ext):
        """Bind the settings of Caesura (None: the planner alone) with fresh rollout and veto state."""
        if ext is not None and self.method != 'b01':
            raise RuntimeError('B01 extension requires method b01')
        self.b01_ext = None if ext is None else dict(ext, rc_steps=list(ext['rc_steps']))
        self.tp_wire = bool(ext is not None and ext['tp_wire'])
        self.tpwire_lock_exchanges = self.tpwire_lock_fallbacks = 0
        self.b01_rules = None
        self.tpwire = None
        if ext is not None:
            from sglang.srt.managers.b300_b01_release_veto_v1 import ReleaseVeto
            self.b01_rules = ReleaseVeto(self.b01_ext)
            LOG.info('B300_PAGE_B01_EXT rank=%d rc_steps=%s veto_tokens=%d tp_wire=%d',
                     self.rank, self.b01_ext['rc_steps'], self.b01_ext['veto_tokens'], self.b01_ext['tp_wire'])
        if self.tp_wire:
            from sglang.srt.managers import b300_page_tpwire_v1 as tpwire
            self.tpwire = tpwire

    def consensus(self, value):
        flag=self.torch.tensor(int(bool(value)),dtype=self.torch.int,device='cpu')
        self.dist.all_reduce(flag,op=self.dist.ReduceOp.MIN,group=self.group)
        if not flag.item():
            raise RuntimeError('TP consensus on the planned step failed')

    def prepare_root(self, rows, limits, digest, valid):
        began=time.perf_counter()
        proof=dict(forced=False)
        try:
            if valid:
                key=(getattr(self.meta_cache,'version',None),
                     getattr(self,'resource_policy',None),
                     getattr(self,'frontier_identity',()),
                     getattr(self,'decoder_identity',()),
                     tuple((r.index,r.total,r.gpu_prefix,r.host_prefix,r.output_limit,
                            r.held_slot,id(r.locks),r.future_output_guard) for r in rows))
                bounds=(limits.token_budget,limits.kv_budget,limits.slots,limits.batch_slots)
                prior=self.resource_certificate
                if (limits.decode_count and prior is not None and prior[0]==key
                        and all(now<=old for now,old in zip(bounds,prior[1]))):
                    proof=dict(forced=True)
                    self.resource_certificate_hits+=1
                else:
                    proof=forced_current_action(rows,limits)
                    self.resource_certificate=(key,bounds) if proof['forced'] else None
        except Exception as exc:
            # A local error must still reach the same collectives as the other rank.
            valid=False
            self.resource_certificate=None
            LOG.error('B300_PAGE_RESOURCE_PROOF_FAILURE rank=%d error=%s',self.rank,type(exc).__name__)
        if not proof['forced']:
            self.resource_minimums=None
        self.last_resource_proof_s=time.perf_counter()-began
        # With the numeric exchange, one integer all_gather replaces the MIN and MAX all_reduce pair.
        sync = self.tpwire.synchronize_root if self.tp_wire else self.root_sync
        return sync(self.torch, self.dist, self.group, limits, digest,
                    len(rows), proof['forced'], valid)

    def try_early_root(self, scheduler, running):
        """Reuse the decode-only certificate of the previous round without a new snapshot of the queue.

        It holds while the waiting queue, the decodes and the policy are unchanged and the resources are the same or
        tighter on both ranks. A miss on either rank makes both ranks take a normal planning round."""
        began=time.perf_counter()
        certificate=self.early_resource_certificate
        plan=certificate['plan'] if certificate is not None else None
        with self.meta_cache.early_touch_lease(plan,scheduler.waiting_queue) as lease:
            valid=False;limits=digest=None
            try:
                if lease is not None and certificate is not None:
                    limits,decoders=_live_limits(scheduler,running)
                    identity=_decoder_identity(decoders)
                    bounds=(limits.token_budget,limits.kv_budget,limits.slots,limits.batch_slots)
                    valid=(limits.decode_count>0
                        and getattr(self,'resource_policy',None)==certificate['policy']
                        and identity==certificate['decoders']
                        and int(scheduler.tree_cache.load_back_threshold)==certificate['threshold']
                        and (all(now<=old for now,old in zip(bounds,certificate['bounds']))
                             or _cached_minimums_blocked(certificate.get('minimums'),limits,
                                                        len(certificate['rows']))))
                    if valid:
                        digest=_root_digest(self,certificate['geometry'],decoders)
            except Exception:
                valid=False
            if self.tp_wire:
                # The validity flag and the root packet travel in one integer all_gather.
                try:
                    waiting=len(certificate['rows']) if valid else 0
                except Exception:
                    valid=False;waiting=0
                fused=self.tpwire.early_root(self.torch,self.dist,self.group,limits,digest,waiting,valid)
                if fused is None:
                    return None,time.perf_counter()-began
            else:
                flag=self.torch.tensor([int(valid)],dtype=self.torch.int64,device='cpu')
                self.dist.all_reduce(flag,op=self.dist.ReduceOp.MIN,group=self.group)
                if not flag.item():
                    return None,time.perf_counter()-began
            if self.tp_wire:
                limits,forced=self.tpwire.unpack_root(fused[0],fused[1],limits)
            else:
                limits,forced=self.root_sync(self.torch,self.dist,self.group,limits,
                    digest,len(certificate['rows']),True,True)
            if not forced:
                raise RuntimeError('reused decode-only certificate lost its TP proof')
            lease.replay()
            self.decoder_identity=certificate['decoders']
            self.resource_minimums=certificate.get('minimums')
            self.resource_certificate_hits+=1
            self.early_resource_certificate_hits+=1
            return (certificate['rows'],limits,digest),time.perf_counter()-began

    def capture_early_root(self, scheduler, rows, limits):
        try:
            threshold=int(scheduler.tree_cache.load_back_threshold)
            expected=tuple((r.gpu_prefix,r.host_prefix,r.output_limit,r.held_slot,r.locks) for r in rows)
            plan=self.meta_cache.capture_touch_plan(scheduler.waiting_queue,
                expected=expected,host_threshold=threshold)
            # The smallest-chunk resource needs of each request, as in usage(), for the next reuse.
            minimums=[]
            for row in rows:
                q=min(PAGE,row.query)
                kv=paged(q)+row.host_prefix+PAGE+sum(dict(row.locks).values())
                if q==row.query:kv+=completion_guard(row)
                minimums.append((paged(q),kv,int(not row.held_slot),1))
            self.resource_minimums=tuple(minimums)
            self.early_resource_certificate=dict(plan=plan,rows=tuple(rows),
                minimums=self.resource_minimums,
                geometry=_waiting_geometry(scheduler,rows),decoders=self.decoder_identity,
                policy=getattr(self,'resource_policy',None),
                bounds=(limits.token_budget,limits.kv_budget,limits.slots,limits.batch_slots),
                threshold=threshold)
        except Exception:
            # Only an optimization: without it, the next round is a normal planning round.
            self.early_resource_certificate=None
            self.resource_minimums=None


def _cached_minimums_blocked(minimums, limits, waiting):
    """Whether the certificate still holds with larger resources: no request fits with its smallest chunk."""
    if (not limits.decode_count or type(waiting) is not int or waiting <= 0
            or not isinstance(minimums, tuple) or len(minimums) != waiting):
        return False
    for used in minimums:
        if (not isinstance(used, tuple) or len(used) != 4
                or any(type(value) is not int for value in used)):
            return False
        tokens, kv, slots, batches = used
        if tokens <= 0 or kv < 0 or slots not in (0,1) or batches != 1:
            return False
        if (tokens <= limits.token_budget and kv < limits.kv_budget
                and slots <= limits.slots and batches <= limits.batch_slots):
            return False
    return True


def runtime(scheduler):
    global STATE
    if STATE is None:
        STATE=Runtime(scheduler)
    return STATE


def _base_stash_round(scheduler,exclude):
    """SGLang's stashing of a started chunk, for every paused or running partial prefill."""
    for req in getattr(scheduler,'_b300_page_last_partials',()):
        exclude.add(req)
        if req.req_pool_idx is not None and req.extend_range is not None:
            if req.extend_range.end>len(req.prefix_indices):
                scheduler.stash_chunked_request(req)
    # No partial prefill may join the decode batch.
    for req in scheduler.waiting_queue:
        if getattr(req,'_b300_page_started',False) and not req.output_ids:
            exclude.add(req)


def _decoder_identity(decoders):
    return tuple((id(req),int(req.sampling_params.max_new_tokens)) for req in decoders)


def _live_limits(scheduler,running):
    decoders=[req for req in running.reqs if not req.finished()]
    ratio,clip=_native_policy(scheduler)
    required=running.new_tokens_required_next_decode() if decoders else 0
    available=scheduler.token_to_kv_pool_allocator.available_size()+scheduler.tree_cache.evictable_size()
    budget=resource_budget(int(available),int(required),decoders,ratio,clip,
                           bool(scheduler.enable_overlap))
    reserve=budget['reserved_tokens']
    limits=Limits(min(8192-len(decoders),16384-len(decoders)),
                  max(0,int(available-reserve)),scheduler.req_to_token_pool.available_size(),
                  max(0,128-len(decoders)),len(decoders),
                  sum(int(r.kv_committed_len) for r in decoders))
    if STATE is not None:
        STATE.resource_policy=(ratio,clip,bool(scheduler.enable_overlap))
    return limits,decoders


def _waiting_geometry(scheduler,rows):
    return tuple((r.index,scheduler.waiting_queue[r.index].rid,r.total,r.gpu_prefix,
                  r.host_prefix,r.output_limit,r.held_slot,r.future_output_guard) for r in rows)


def _root_digest(st,waiting_geometry,decoders):
    """Digest of the queue and decode geometry that both ranks must agree on."""
    geometry=(getattr(st,'resource_policy',None),waiting_geometry,
              tuple((r.rid,len(r.output_ids),r.kv_committed_len,
                    int(r.sampling_params.max_new_tokens)) for r in decoders))
    if geometry != getattr(st,'geometry_signature',None):
        st.geometry_signature=geometry
        st.geometry_digest=hashlib.sha256(repr(geometry).encode()).hexdigest()
    return st.geometry_digest


def _snapshot(scheduler,running,st):
    now=time.perf_counter()
    rows=[]
    identity=tuple((id(req),req.rid) for req in scheduler.waiting_queue)
    if identity!=getattr(st,'frontier_identity',None):
        st.meta_cache.prune(scheduler.waiting_queue)
    st.frontier_identity=identity
    for i,req in enumerate(scheduler.waiting_queue):
        if (req.output_ids or req.finished() or req.return_logprob
                or any(getattr(req,k,None) is not None for k in
                       ('multimodal_inputs','input_embeds','positional_embed_overrides'))):
            raise RuntimeError('unsupported request in the waiting queue of the planner')
        held=req.req_pool_idx is not None
        meta = st.meta_cache.lookup(req, held=held)
        path=meta.path
        prefix, host, locks = meta.gpu_prefix, meta.host_prefix, meta.locks
        # HiCache does not load back fewer tokens than this threshold; they are prefilled instead.
        if host<int(scheduler.tree_cache.load_back_threshold):host=0
        if prefix+host>=len(req.origin_input_ids):
            raise RuntimeError('a waiting request must have at least one token to prefill')
        rows.append(Row(i,max(0.,now-req.time_stats.scheduler_recv_time),
                        len(req.origin_input_ids),prefix,host,
                        max(0,int(req.sampling_params.max_new_tokens)),held,
                        locks,path,future_output_guard(int(req.sampling_params.max_new_tokens),
                                                     int(scheduler._b300_v9_clip))))
    limits,decoders=_live_limits(scheduler,running)
    st.decoder_identity=_decoder_identity(decoders)
    digest=_root_digest(st,_waiting_geometry(scheduler,rows),decoders)
    return rows,limits,digest


def _fresh_prefill(scheduler,running,native_adder,batch_cls,stats_cls,set_time_batch):
    """One planning round for the waiting requests that have not started decoding."""
    st=runtime(scheduler)
    started=time.perf_counter()
    if scheduler.grammar_manager.has_waiting_grammars():
        for req in scheduler.grammar_manager.get_ready_grammar_requests():
            scheduler._add_request_to_queue(req)
    scheduler.tree_cache.check_hicache_events()
    running.filter_batch()
    running.batch_is_full=False
    if not scheduler.waiting_queue:
        scheduler._b300_page_last_partials=()
        return None,running
    error=None
    early,_=st.try_early_root(scheduler,running)
    try:
        rows,limits,digest=early if early is not None else _snapshot(scheduler,running,st)
        local=dict(digest=digest,limits=asdict(limits),
                   locks=[] if early is not None else [r.locks for r in rows],error=None)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        rows=();limits=None
        local=dict(error=error)
    forced=False
    if early is not None:
        forced=True
        gathered=[None,None]
    else:
        limits,forced=st.prepare_root(rows,limits,local.get('digest','0'*64),error is None)
        # The lock costs of the two ranks are exchanged only when a prefill is still possible.
        gathered=[None,None]
        if not forced:
            if st.tp_wire:
                gathered=st.tpwire.exchange_locks(st,local['locks'])
            else:
                st.dist.all_gather_object(gathered,local['locks'],group=st.group)
        if forced:
            st.capture_early_root(scheduler,rows,limits)
        else:
            st.early_resource_certificate=None
    payload=None
    if forced:
        # The decode-only certificate holds on both ranks: no search.
        predicted=st.model.predict(rows,{},limits)
        decision=dict(action={},score=None,wall_s=0.,predicted_execution_s=predicted,
            status='forced_resource_decode',stats=dict(evaluated=0,feasible=1,
                origin='forced_resource_decode',selected_incumbent_origin_id=1,
                dfs_exhausted=True))
        payload=dict(error=None,digest=digest,limits=asdict(limits),decision=decision)
    elif st.rank==0:
        try:
            rows=merge_locks(rows,gathered)
            preparation=time.perf_counter()-started
            # The step in flight ends after its predicted duration; the planned step starts then.
            wait=max(0.,st.last_duration_s-(time.perf_counter()-st.last_launch_s)) if st.last_launch_s else 0.
            decision=solve(rows,limits,st.model,budget_s=st.search_budget_s,
                           planning_allowance_s=st.planning_allowance_s,
                           predicted_wait_s=wait+preparation,rotation_round=st.rounds)
            st.planning_allowance_s=.9*st.planning_allowance_s+.1*decision['wall_s']
            if st.b01_ext is not None:
                # Caesura: the release-aware rollout and the veto; any error keeps the planner's schedule.
                decision=_b01_ext_round(st,scheduler,running,rows,limits,decision,wait+preparation)
            payload=dict(error=None,digest=digest,limits=asdict(limits),decision=decision)
        except Exception as exc:
            payload=dict(error=type(exc).__name__+': '+str(exc))
    if not forced:
        shared=[payload]
        st.dist.broadcast_object_list(shared,src=st.source,group=st.group)
        payload=shared[0]
    valid=error is None and isinstance(payload,dict) and not payload.get('error')
    if valid:
        limits=Limits(**payload['limits'])
        action=payload['decision']['action']
        # Only rank 0 holds the step-time model; the other rank checks the resources of the schedule.
        consumed=usage(rows,action)
        valid=(payload['digest']==local['digest'] and consumed is not None
               and all(i in range(len(rows)) for i in action)
               and consumed[0]<=limits.token_budget and (not action or consumed[1]<limits.kv_budget)
               and consumed[2]<=limits.slots and consumed[3]<=limits.batch_slots)
        if not action:
            valid=(payload['decision']['stats']['origin']=='no_feasible_prefill'
                   or limits.decode_count>0 or payload['decision']['status']=='stale_root_replan')
        if not forced and (action or limits.decode_count) and payload['decision']['status']!='stale_root_replan':
            valid=valid and bool(payload['decision']['stats'].get('selected_action_scored',True))
    if not valid:
        LOG.error('B300_PAGE_TP_CHECK_FAILURE rank=%d local_error=%s root_error=%s local_digest=%s root_digest=%s limits=%s consumed=%s',
            st.rank,error,payload.get('error') if isinstance(payload,dict) else 'invalid_payload',
            local.get('digest'),payload.get('digest') if isinstance(payload,dict) else None,
            payload.get('limits') if isinstance(payload,dict) else None,locals().get('consumed'))
    if not forced:
        st.consensus(valid)
    elif not valid:
        raise RuntimeError('invalid decode-only certificate schedule')
    decision=payload['decision']; action=decision['action']
    elapsed=time.perf_counter()-started
    st.rounds+=1;st.cpu_s+=elapsed;st.max_cpu_s=max(st.max_cpu_s,elapsed)
    selected=[scheduler.waiting_queue[i] for i in sorted(action)]
    if not selected:
        scheduler._b300_page_last_partials=()
        return None,running
    # Match only the scheduled requests in SGLang's radix cache, and check that both ranks see the planned prefixes.
    initial_geometry=[];initial_error=None
    try:
        for i in sorted(action):
            req=scheduler.waiting_queue[i];row=rows[i]
            req.init_next_round_input(None if req.req_pool_idx is not None else scheduler.tree_cache)
            host=0 if row.held_slot else int(req.host_hit_length)
            if host<int(scheduler.tree_cache.load_back_threshold):host=0
            initial_geometry.append((i,len(req.prefix_indices),host))
            if len(req.prefix_indices)!=row.gpu_prefix or host!=row.host_prefix:
                raise RuntimeError('cached prefix metadata differs from the radix cache match')
    except Exception as exc:
        initial_error=type(exc).__name__
    if st.tp_wire:
        st.tpwire.agree_geometry(st,initial_error is None,initial_geometry)
    else:
        matched=[None,None]
        st.dist.all_gather_object(matched,(initial_error,initial_geometry),group=st.group)
        st.consensus(initial_error is None and matched[0]==matched[1])
    adder=native_adder(PAGE,scheduler.tree_cache,scheduler.token_to_kv_pool_allocator,
        running,0.,16384,8192,limits.decode_count,0,
        max_prefill_bs=int(scheduler.max_prefill_bs),max_running_requests=128,
        waiting_queue_len=len(rows))
    guards=[]
    partials=[]
    committed={};actual_rows=list(rows);host_reused=0
    try:
        restore_error=None;geometry=[]
        try:
            # Lock the matched prefixes of all scheduled requests before any load-back or allocation can evict them.
            for req in selected:
                if req.req_pool_idx is None:
                    node=req.last_node
                    scheduler.tree_cache.inc_lock_ref(node)
                    guards.append(node)
            from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
            for i,q in sorted(action.items()):
                req=scheduler.waiting_queue[i];row=rows[i]
                loaded=0
                if row.host_prefix:
                    indices,node=scheduler.tree_cache.init_load_back(InitLoadBackParams(
                        best_match_node=req.best_match_node,host_hit_length=row.host_prefix,req=req))
                    loaded=len(indices)
                    if loaded>row.host_prefix:raise RuntimeError('unexpected host restore growth')
                if not row.held_slot:
                    # An earlier load-back may have brought shared host pages to the GPU, so match again.
                    req.init_next_round_input(scheduler.tree_cache)
                    scheduler.tree_cache.inc_lock_ref(req.last_node)
                    guards.append(req.last_node)
                prefix=len(req.prefix_indices)
                if prefix<row.gpu_prefix or prefix>=row.total or loaded>prefix:
                    raise RuntimeError('selected GPU prefix unexpectedly changed')
                host_reused+=max(0,prefix-row.gpu_prefix-loaded)
                q=committed_query(row,q,prefix)
                req.host_hit_length=loaded
                actual_rows[i]=replace(row,gpu_prefix=prefix-loaded,host_prefix=loaded)
                if q:committed[i]=q
            geometry=[(i,len(scheduler.waiting_queue[i].prefix_indices),actual_rows[i].host_prefix,q)for i,q in sorted(committed.items())]
        except Exception as exc:
            # A local failure must still reach the same collective as the other rank.
            restore_error=type(exc).__name__
        # Both ranks check the prefixes after load-back before allocating. Load-back can only reduce the planned
        # work; a request that could now complete waits for a planning round that reserves its output.
        if st.tp_wire:
            st.tpwire.agree_geometry(st,restore_error is None,geometry)
        else:
            gathered_geometry=[None,None]
            st.dist.all_gather_object(gathered_geometry,(restore_error,geometry),group=st.group)
            st.consensus(restore_error is None and gathered_geometry[0]==gathered_geometry[1])
        action=committed;rows=actual_rows
        for i,q in sorted(action.items()):
            req=scheduler.waiting_queue[i];row=rows[i]
            if not row.held_slot:
                adder._req_inc_lock_ref(req)
                if getattr(req,'_b300_page_started',False):st.resumes+=1
            prefix=len(req.prefix_indices)
            if getattr(req,'_b300_page_started',False) and prefix<req.already_computed:
                # Evicted progress of a paused prefill is computed again.
                req.retracted_stain=True
            req.set_extend_range(prefix,prefix+q)
            partial=prefix+q<len(req.full_untruncated_fill_ids)
            if partial:
                if q%PAGE or (prefix+q)%PAGE:
                    raise RuntimeError('partial prefill endpoint must be page aligned')
                partials.append(req);req.inflight_middle_chunks+=1
                st.partial_chunks+=1
            req._b300_page_started=True
            req._b300_page_inflight=getattr(req,'_b300_page_inflight',0)+1
            adder.can_run_list.append(req)
            adder._update_prefill_budget(prefix,q,0,req.retracted_stain,
                host_hit_len=row.host_prefix,storage_hit_len=0)
    finally:
        for node in reversed(guards):scheduler.tree_cache.dec_lock_ref(node)
    selected=[scheduler.waiting_queue[i]for i in sorted(action)]
    total_elapsed=time.perf_counter()-started
    st.cpu_s+=total_elapsed-elapsed;st.max_cpu_s=max(st.max_cpu_s,total_elapsed)
    if not selected:
        scheduler._b300_page_last_partials=()
        return None,running
    completed={req for req in selected if req not in partials}
    scheduler.waiting_queue=[r for r in scheduler.waiting_queue if r not in completed]
    scheduler._b300_page_last_partials=tuple(partials)
    set_time_batch(selected,'set_forward_entry_time')
    batch=batch_cls.init_new(selected,scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,scheduler.tree_cache,
        scheduler.model_config,scheduler.enable_overlap,scheduler.spec_algorithm)
    batch.b300_page_partials=tuple(partials)
    batch.b300_page_features=st.model.features(rows,action,limits)
    batch.b300_page_decoders=limits.decode_count
    batch.contains_last_prefill_chunk=bool(completed)
    batch.hicache_consumer_index=scheduler.tree_cache.ready_to_load_host_cache()
    batch.prepare_for_extend()
    batch.prefill_stats=stats_cls.from_adder(adder,running.reqs,False,
        num_pending_tokens=scheduler.load_inquirer._get_num_pending_tokens(chunk_deduct=0))
    if not running.is_empty():
        running.prepare_for_decode()
        batch.mix_with_running(running);batch.decoding_reqs=running.reqs
        running=batch_cls(reqs=[],batch_is_full=False)
    else:batch.decoding_reqs=None
    if st.rounds==1 or st.rounds%128==0:
        LOG.info('B300_PAGE_SCHEDULER_STATS method=%s rank=%d rounds=%d partial_chunks=%d pauses=%d resumes=%d cpu_total_s=%.6f cpu_max_s=%.6f model_observations=%d',st.method,st.rank,st.rounds,st.partial_chunks,st.pauses,st.resumes,st.cpu_s,st.max_cpu_s,st.model.observations)
    return batch,running


def committed_query(row,planned,prefix):
    """The tokens to prefill after load-back: never more than planned, and whole pages unless the plan completes."""
    remaining=row.total-prefix
    q=min(planned,remaining)
    if q==remaining and planned!=row.query:
        # The plan did not reserve the output of a request that now completes.
        q=(remaining-1)//PAGE*PAGE
    elif q<remaining:q=q//PAGE*PAGE
    return max(0,q)


def timing_features(batch):
    # A running batch keeps these fields when it turns into a decode batch, so the forward mode decides.
    if batch.forward_mode.is_extend():
        return getattr(batch,'b300_page_features',None) or (0,0,0)
    return (0,0,0)


def _base_note_result(scheduler,batch,result):
    st=runtime(scheduler)
    timing=getattr(batch,'b300_page_timing',None)
    if timing is not None:
        # The result is already synchronized, so reading the CUDA events adds no barrier.
        cuda_s=timing[0].elapsed_time(timing[1])/1000.
        features=timing_features(batch)
        decoders=getattr(batch,'b300_page_decoders',0) if features[0] else len(batch.reqs)
        if st.rank==0:st.model.observe(features,decoders,cuda_s)
    from sglang.srt.mem_cache.common import release_kv_cache
    for req in batch.reqs:
        count=getattr(req,'_b300_page_inflight',0)
        if count:
            req._b300_page_inflight=count-1
        if (req in batch.b300_page_partials and getattr(req,'_b300_page_inflight',0)==0
                and not getattr(req,'_b300_v9_restore_pending',False)
                and not req.output_ids and not req.finished()):
            # A paused prefill whose last chunk has drained releases its prefix to the cache and its slot, so that
            # the cache can evict it instead of holding its KV.
            release_kv_cache(req,scheduler.tree_cache,is_insert=True)
            req.prefix_indices=st.torch.empty((0,),dtype=st.torch.int64)
            req.last_node=None;req.cache_protected_len=0
            req.kv_committed_len=0;req.extend_range=None
            req.host_hit_length=0;req.last_host_node=None;req.best_match_node=None
            if req.inflight_middle_chunks!=0:
                raise RuntimeError('paused prefill has outstanding middle chunks')
            st.pauses+=1


def note_launch(scheduler,batch):
    st=runtime(scheduler)
    st.last_launch_s=time.perf_counter()
    features=timing_features(batch)
    if features[0]:
        q,attention,host=features
        scale=st.model.mixed_scale if batch.b300_page_decoders else st.model.pure_scale
        st.last_duration_s=(st.model.intercept_s+st.model.query_token_s*q+
            st.model.query_context_billion_s*attention/1e9)*scale+host*st.model.transfer_token_s
    else:st.last_duration_s=st.model.decode_s


def _native_policy(scheduler):
    """SGLang's current new-token ratio and output clip."""
    from sglang.srt.managers.b300_page_resources_v9 import _policy
    return _policy(scheduler.new_token_ratio_tracker.current,
                   scheduler._b300_v9_clip)


def _request_geometry(req):
    """Scalars of a request that both ranks must agree on."""
    extent=getattr(req,'extend_range',None)
    return (req.rid,len(req.origin_input_ids),len(req.output_ids),
            int(req.kv_committed_len),int(req.sampling_params.max_new_tokens),
            getattr(req,'req_pool_idx',None),getattr(req,'priority',None),
            bool(req.finished()),len(getattr(req,'prefix_indices',())),
            None if extent is None else (int(extent.start),int(extent.end)))


def _synchronize_native_geometry(st,geometry,valid=True,local_ok=True):
    """Fixed-size handshake of both ranks on a geometry digest; a mismatch fails on both."""
    try:
        if type(valid) is not bool or type(local_ok) is not bool:
            raise ValueError('invalid native memory result')
        digest=hashlib.sha256(repr(geometry).encode()).hexdigest()
        words=[int(digest[i:i+8],16) for i in range(0,64,8)]
    except Exception:
        valid=False;local_ok=False;words=[0]*8
    values=[int(valid),int(local_ok),*words]
    packet=st.torch.tensor(values,dtype=st.torch.int64,device='cpu')
    gathered=[packet.clone(),packet.clone()]
    st.dist.all_gather(gathered,packet,group=st.group)
    left,right=(part.tolist() for part in gathered)
    if not (left[0] and right[0]) or left[2:]!=right[2:]:
        raise RuntimeError('decoder geometry differs between TP ranks')
    st.native_sync_calls=getattr(st,'native_sync_calls',0)+1
    return bool(left[1] and right[1])


def _checked_native_geometry(st,builder,valid=True):
    try:
        geometry=builder()
    except Exception:
        valid=False;geometry=()
    return _synchronize_native_geometry(st,geometry,valid)


def synchronize_decode_memory(batch,local_ok,selected_indices=None):
    """SGLang's decode memory check on both ranks: if either rank is short of KV, both retract.

    Retraction calls it again with the remaining requests; a mismatch fails on both ranks."""
    st=STATE
    if st is None:
        raise RuntimeError('decode memory check before the page scheduler started')
    valid=bool(getattr(st,'v9_memory_envelope_valid',True));geometry=()
    try:
        indices=tuple(range(len(batch.reqs))) if selected_indices is None else tuple(selected_indices)
        if any(type(i) is not int or not 0<=i<len(batch.reqs) for i in indices) or len(set(indices))!=len(indices):
            raise ValueError('invalid native decoder subset')
        geometry=(getattr(st,'v9_memory_envelope',None),
                  tuple(_request_geometry(batch.reqs[i]) for i in indices))
    except Exception:
        valid=False
    return _synchronize_native_geometry(st,geometry,valid,local_ok)


def _invalidate_resource_certificates(st):
    st.resource_certificate=None
    st.early_resource_certificate=None
    st.resource_minimums=None
    st.geometry_signature=st.geometry_digest=None


def _is_recovery(req):
    return bool(getattr(req,'_b300_v9_restore_pending',False) or req.output_ids)


def stash_round(scheduler,exclude):
    """Stash started chunks. SGLang's own chunked request may only be a retracted decode being restored."""
    st=runtime(scheduler)
    st.v9_checked_running=None
    chunk=getattr(scheduler,'chunked_req',None)
    if chunk is not None:
        if not _is_recovery(chunk):
            raise RuntimeError('SGLang chunked request must be a retracted decode')
        exclude.add(chunk)
        extent=getattr(chunk,'extend_range',None)
        if extent is not None and extent.end>len(chunk.prefix_indices):
            scheduler.stash_chunked_request(chunk)
    _base_stash_round(scheduler,exclude)


def prepare_running(scheduler,running):
    """Run SGLang's filter, decode memory check and retraction of the running batch once per step.

    The decode tensors are prepared only after the prefill schedule is known (prepare_decode).
    """
    st=runtime(scheduler)
    if getattr(st,'v9_checked_running',None) is running:
        return running
    before=tuple(running.reqs)
    witnesses={}
    old_policy=None
    # Both ranks run one memory check even for an empty or finished batch, on the state before filtering, so that
    # the collectives stay in the same order on both ranks.
    envelope_valid=True
    try:
        native_globals=scheduler._b300_v9_check_running_batch.__func__.__globals__
    except AttributeError:
        native_globals=getattr(scheduler._b300_v9_check_running_batch,'__globals__',{})
    try:
        witnesses={id(r):(r,r.output_ids,len(r.output_ids)) for r in before}
        old_policy=_native_policy(scheduler)
        test_retract=native_globals.get('TEST_RETRACT',False)
        if test_retract is not False:
            raise ValueError('unsupported test-only decoder retraction')
        tracker=scheduler.new_token_ratio_tracker
        if (not math.isfinite(float(tracker.min)) or not 0<=float(tracker.min)<=1
                or not math.isfinite(float(tracker.decay)) or float(tracker.decay)<0):
            raise ValueError('invalid native ratio controls')
        control=(old_policy,float(tracker.min),float(tracker.decay),
                 scheduler.server_args.retraction_policy,
                 bool(scheduler.server_args.schedule_low_priority_values_first),False)
        route=(any(_is_recovery(r) for r in scheduler.waiting_queue),
               getattr(scheduler,'chunked_req',None) is not None)
        live_indices=[i for i,r in enumerate(before) if not r.finished()]
        st.v9_memory_envelope=(tuple(_request_geometry(r) for r in before),
                               tuple(live_indices),control,route)
    except Exception:
        envelope_valid=False;live_indices=[]
        st.v9_memory_envelope=('invalid_native_control',)
    # An invalid state reaches the other rank through the same fixed packet instead of an exception.
    st.v9_memory_envelope_valid=envelope_valid
    original_check=running.check_decode_mem
    try:
        if live_indices:
            initial_ok=original_check(selected_indices=live_indices)
        else:
            # Without live decodes there is nothing to allocate, only the handshake.
            initial_ok=synchronize_decode_memory(running,True,selected_indices=[])
    finally:
        st.v9_memory_envelope=None
        st.v9_memory_envelope_valid=True
    consumed=False
    instance_had_check='check_decode_mem' in running.__dict__
    instance_check=running.__dict__.get('check_decode_mem')

    def consume_initial_check(selected_indices=None):
        nonlocal consumed
        if not consumed and selected_indices is None:
            # SGLang's filter_batch changes no request or cache state, so its first memory check reuses the
            # result of the check above.
            consumed=True
            return initial_ok
        return original_check(selected_indices=selected_indices)

    running.check_decode_mem=consume_initial_check
    try:
        # SGLang frees KV and slots, updates its ratio and requeues retracted requests; every check after a
        # release runs again on both ranks.
        checked=scheduler._b300_v9_check_running_batch(running)
    finally:
        if instance_had_check:running.check_decode_mem=instance_check
        else:del running.check_decode_mem
    if checked is None:
        raise RuntimeError('unexpected native running-batch result')
    queued={id(r) for r in scheduler.waiting_queue}
    retained={id(r) for r in checked.reqs}
    retracted=[r for r in before if id(r) not in retained and id(r) in queued and not r.finished()]
    preserved=all(r.output_ids is witnesses[id(r)][1] and len(r.output_ids)==witnesses[id(r)][2]
                  for r in retracted)
    for req in retracted:
        req._b300_v9_restore_pending=True
        req._b300_v9_restore_output_container=req.output_ids
        req._b300_v9_restore_output_count=len(req.output_ids)
    if not initial_ok:
        _checked_native_geometry(st,lambda: (
            tuple(_request_geometry(r) for r in checked.reqs),
            tuple(_request_geometry(r) for r in retracted),_native_policy(scheduler),
            any(_is_recovery(r) for r in scheduler.waiting_queue),
            getattr(scheduler,'chunked_req',None) is not None),preserved)
    st.v9_checked_route=(any(_is_recovery(r) for r in scheduler.waiting_queue),
                         getattr(scheduler,'chunked_req',None) is not None)
    if retracted or old_policy!=_native_policy(scheduler):
        _invalidate_resource_certificates(st)
    st.native_retracted_total+=len(retracted)
    st.native_memory_checks+=int(bool(before))
    st.v9_checked_running=checked
    # Refresh the resource policy after the release.
    _live_limits(scheduler,checked)
    return checked


def prepare_decode(scheduler,running):
    """Prepare the decode step once, after the memory check of prepare_running."""
    st=runtime(scheduler)
    if getattr(st,'v9_checked_running',None) is not running:
        running=prepare_running(scheduler,running)
    if not running.is_empty():
        running.prepare_for_decode()
    return running


def _recovery_features(batch,restoring):
    q=attention=host=0
    for req in restoring:
        count=int(req.extend_range.end-req.extend_range.start)
        prefix=int(req.extend_range.start)
        q+=count
        attention+=count*(prefix+(count+1)/2)
        host+=max(0,int(getattr(req,'host_hit_length',0)))
    return q,attention,host


def prefill(scheduler,running,native_adder,batch_cls,stats_cls,set_time_batch,
            *,prefill_delayer_single_pass=None):
    """The prefill of a step: a planning round, or SGLang's own admission to restore retracted decodes."""
    st=runtime(scheduler)
    # Move ready grammar requests into the queue first, so that a new request cannot enter the restore lane.
    if scheduler.grammar_manager.has_waiting_grammars():
        for req in scheduler.grammar_manager.get_ready_grammar_requests():
            scheduler._add_request_to_queue(req)
    recovery=[r for r in scheduler.waiting_queue if _is_recovery(r)]
    chunk=getattr(scheduler,'chunked_req',None)
    # Both ranks must take the same route; prepare_running has already checked it.
    if getattr(st,'v9_checked_running',None) is not running:
        _synchronize_native_geometry(st,(bool(recovery),chunk is not None))
    if not recovery and chunk is None:
        return _fresh_prefill(scheduler,running,native_adder,batch_cls,
                              stats_cls,set_time_batch)
    fresh=[r for r in scheduler.waiting_queue if not _is_recovery(r)]
    if chunk is not None and not _is_recovery(chunk):
        raise RuntimeError('SGLang chunked request must be a retracted decode')
    _invalidate_resource_certificates(st)
    _checked_native_geometry(st,lambda: (
        tuple(_request_geometry(r) for r in recovery),
        None if chunk is None else _request_geometry(chunk),_native_policy(scheduler)))
    scheduler.waiting_queue=list(recovery)
    error=None;batch=None
    try:
        batch,running=scheduler._b300_v9_native_recovery_prefill(
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            running_batch=running)
    except Exception as exc:
        error=exc
    finally:
        # Keep the new requests and the retracted decodes that were not restored.
        remaining=scheduler.waiting_queue
        if any(not _is_recovery(r) for r in remaining):
            error=error or RuntimeError('fresh request entered native restore lane')
        scheduler.waiting_queue=fresh+remaining
    # Both ranks take part even when one of them failed.
    _checked_native_geometry(st,lambda: (
        None if batch is None else tuple(_request_geometry(r) for r in batch.reqs),
        tuple(_request_geometry(r) for r in running.reqs),
        tuple(_request_geometry(r) for r in scheduler.waiting_queue if _is_recovery(r)),
        None if scheduler.chunked_req is None else _request_geometry(scheduler.chunked_req)),
        error is None)
    if error is not None:
        raise RuntimeError('restoring retracted decodes failed') from error
    partials=() if batch is None or batch.chunked_req is None else (batch.chunked_req,)
    decoding=() if batch is None else (batch.decoding_reqs or ())
    selected=[] if batch is None else [r for r in batch.reqs if r not in decoding]
    if any(not _is_recovery(r) for r in selected):
        raise RuntimeError('fresh request selected by native restore lane')
    if batch is not None:
        batch.b300_v9_recovery=True
        batch.b300_page_partials=partials
        batch.b300_page_features=_recovery_features(batch,selected)
        batch.b300_page_decoders=len(batch.decoding_reqs or ())
    st.native_recovery_calls+=1
    return batch,running


def note_result(scheduler,batch,result):
    """After SGLang has processed the result of a step."""
    st=runtime(scheduler)
    recovered=[]
    if getattr(batch,'b300_v9_recovery',False) and batch.forward_mode.is_extend():
        partials=getattr(batch,'b300_page_partials',())
        recovered=[r for r in batch.reqs if getattr(r,'_b300_v9_restore_pending',False)
                   and r not in partials and not getattr(r,'is_retracted',False)]
        preserved=all(r.output_ids is getattr(r,'_b300_v9_restore_output_container',None)
                      and len(r.output_ids)>getattr(r,'_b300_v9_restore_output_count',0)
                      for r in recovered)
        st.consensus(preserved)
        for req in recovered:
            req._b300_v9_restore_pending=False
            req._b300_v9_restore_output_container=None
        st.native_recovered_total+=len(recovered)
    _base_note_result(scheduler,batch,result)
    if st.rank==0 and st.b01_rules is not None:
        st.b01_rules.note_finished(batch.reqs)   # output lengths for the predictor of the rollout


def _b01_ext_from_environment(method):
    """The settings of Caesura from the environment, or None for the planner alone."""
    spec=json.loads(os.environ.get('B300_PAGE_B01_EXT_JSON','null'))
    if spec is None:
        return None
    if method!='b01':
        raise RuntimeError('b01_ext requires method b01')
    from sglang.srt.managers.b300_persistent_dispatch_rollout_v12 import validate_b01_ext
    return validate_b01_ext(spec)


def _b01_ext_round(st,scheduler,running,rows,limits,decision,predicted_wait_s):
    """Rank 0 only. Any error keeps the planner's decision, which the other rank then validates as usual."""
    try:
        from sglang.srt.managers.b300_b01_release_veto_v1 import apply_round
        return apply_round(st.b01_rules,st,scheduler,running,rows,limits,decision,predicted_wait_s)
    except Exception:
        st.b01_ext_hook_errors=getattr(st,'b01_ext_hook_errors',0)+1
        return decision
