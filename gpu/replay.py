"""Portable CPU verification and open-loop HTTP replay of synthetic token IDs.

Python 3.11+; standard library only. Copy this file beside
load_synthetic_workload.py, or run it from the repository. The HTTP URL is the
full OpenAI-compatible /v1/completions endpoint. No server deployment, policy
change, retry, cancellation-on-error, or worker-log inference is performed.
"""
from __future__ import annotations

import argparse
import asyncio
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import sys
import time
from urllib.parse import urlsplit
import uuid

try:
    from .load_synthetic_workload import Workload
except ImportError:
    from load_synthetic_workload import Workload


SCHEMA = 'portable-synthetic-client-replay-v1'
ROUTING_SCHEMA = 'synthetic-worker-routing-v1'
MAX_HEADERS = 65536
MAX_EVENT = 1024 * 1024
MAX_BODY = 64 * 1024 * 1024


class ReplayError(ValueError):
    """An allowlisted error code; response bodies and exception text stay private."""


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_workload(workload):
    m = workload.manifest
    if (not finite_number(m['duration_s']) or m['duration_s'] <= 0
            or not finite_number(m['measurement_start_s'])
            or not 0 <= m['measurement_start_s'] <= m['duration_s']):
        raise ReplayError('invalid_workload_duration')
    for event in m['events']:
        if (not finite_number(event['offset_s'])
                or type(event['request']) is not int
                or event['prompt_tokens'] <= 0 or event['output_tokens'] <= 0):
            raise ReplayError('invalid_workload_event')


@dataclass(frozen=True)
class PreparedRequest:
    event: dict
    body_parts: tuple[bytes, ...]
    body_bytes: int


def unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ReplayError('routing_duplicate_json_key')
        value[key] = item
    return value


def select_worker(workload, prepared, path, worker):
    """Validate a complete separate routing manifest, then preserve selected events."""
    if type(worker) is not int or worker not in range(4):
        raise ReplayError('routing_worker_invalid')
    raw = Path(path).read_bytes()
    try:
        routing = json.loads(raw, object_pairs_hook=unique_json_object)
    except (UnicodeError, json.JSONDecodeError):
        raise ReplayError('routing_json_invalid') from None
    required = {'schema', 'workload_expanded_token_sha256', 'request_count', 'assignments'}
    if not isinstance(routing, dict) or set(routing) != required or routing['schema'] != ROUTING_SCHEMA:
        raise ReplayError('routing_schema_invalid')
    events = workload.manifest['events']
    if type(routing['request_count']) is not int or routing['request_count'] != len(events):
        raise ReplayError('routing_request_count_mismatch')
    assignments = routing['assignments']
    if not isinstance(assignments, list) or len(assignments) != len(events):
        raise ReplayError('routing_coverage_invalid')
    assigned = {}
    for item in assignments:
        if (not isinstance(item, dict) or set(item) != {'request', 'worker'}
                or type(item['request']) is not int or type(item['worker']) is not int
                or not 0 <= item['request'] < len(events) or item['worker'] not in range(4)):
            raise ReplayError('routing_assignment_invalid')
        if item['request'] in assigned:
            raise ReplayError('routing_duplicate_request')
        assigned[item['request']] = item['worker']
    if set(assigned) != set(range(len(events))):
        raise ReplayError('routing_coverage_invalid')
    if routing['workload_expanded_token_sha256'] != workload.verify()['expanded_token_sha256']:
        raise ReplayError('routing_workload_hash_mismatch')
    selected = [request for request in prepared if assigned[request.event['request']] == worker]
    selected_warmup = sum(request.event['warmup'] for request in selected)
    full_warmup = sum(event['warmup'] for event in events)
    metadata = dict(schema=ROUTING_SCHEMA, worker=worker,
                    routing_file_sha256=hashlib.sha256(raw).hexdigest(),
                    full_requests=len(events), selected_requests=len(selected),
                    full_warmup_requests=full_warmup,
                    selected_warmup_requests=selected_warmup,
                    full_measurement_requests=len(events) - full_warmup,
                    selected_measurement_requests=len(selected) - selected_warmup,
                    original_arrival_offsets_preserved=True, original_warmup_flags_preserved=True,
                    first_selected_offset_s=selected[0].event['offset_s'] if selected else None)
    return selected, metadata


def prepare_requests(workload, model):
    """Encode shared trie edges once, before any scheduled request clock starts."""
    validate_workload(workload)
    if not isinstance(model, str) or not model:
        raise ReplayError('invalid_model')
    encoded_edges = []
    for node in workload.manifest['nodes']:
        ids = array('I')
        ids.frombytes(workload.edges[node['offset']:node['offset'] + node['bytes']])
        if sys.byteorder != 'little':
            ids.byteswap()
        encoded_edges.append(json.dumps(list(ids), separators=(',', ':')).encode()[1:-1])
    prepared = []
    for event in workload.manifest['events']:
        chain = []
        node = event['node']
        while node != -1:
            if encoded_edges[node]:
                chain.append(encoded_edges[node])
            node = workload.manifest['nodes'][node]['parent']
        parts = [b'{"prompt":[']
        for edge in reversed(chain):
            if len(parts) > 1:
                parts.append(b',')
            parts.append(edge)
        fields = dict(model=model, max_tokens=event['output_tokens'], temperature=0,
                      ignore_eos=True, stream=True, stream_options={'include_usage': True})
        if event['namespace']:
            fields['cache_salt'] = 'synthetic-namespace-' + str(event['namespace'])
        parts.append(b'],' + json.dumps(fields, separators=(',', ':')).encode()[1:])
        prepared.append(PreparedRequest(dict(event), tuple(parts), sum(map(len, parts))))
    return prepared


def prepare_smoke(workload, model):
    """Four tiny requests from generated bundle tokens; a separate smoke workload."""
    validate_workload(workload)
    if not workload.manifest['events']:
        raise ReplayError('empty_smoke_source')
    first = workload.manifest['events'][0]
    ids = array('I')
    ids.frombytes(workload.token_bytes(first['node'])[:1024 * 4])
    if sys.byteorder != 'little':
        ids.byteswap()
    body = json.dumps(dict(model=model, prompt=list(ids), max_tokens=8, temperature=0,
                           ignore_eos=True, stream=True,
                           stream_options={'include_usage': True}), separators=(',', ':')).encode()
    return [PreparedRequest(dict(request=i, warmup=False, offset_s=i * .1,
                                 prompt_tokens=len(ids), output_tokens=8), (body,), len(body))
            for i in range(4)]


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    tls: ssl.SSLContext | None
    header_prefix: bytes


def make_endpoint(url, api_key=None):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.fragment or any(c.isspace() for c in url)):
            raise ValueError()
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        host_header = '[' + host + ']' if ':' in host else host
        if parsed.port is not None:
            host_header += ':' + str(port)
        prefix = (f'POST {path} HTTP/1.1\r\nHost: {host_header}\r\n'
                  'Content-Type: application/json\r\nAccept: text/event-stream\r\n'
                  'Accept-Encoding: identity\r\nConnection: close\r\n')
        if api_key is not None:
            if not api_key or '\r' in api_key or '\n' in api_key:
                raise ValueError()
            prefix += 'Authorization: Bearer ' + api_key + '\r\n'
        return Endpoint(host, port, ssl.create_default_context() if parsed.scheme == 'https'
                        else None, prefix.encode('ascii'))
    except (ValueError, UnicodeError):
        raise ReplayError('invalid_endpoint_or_credentials') from None


async def read_headers(reader):
    status_line = await reader.readline()
    size = len(status_line)
    parts = status_line.split()
    if (len(parts) < 2 or parts[0] not in (b'HTTP/1.1', b'HTTP/1.0')
            or len(parts[1]) != 3 or not parts[1].isdigit()):
        raise ReplayError('http_status_invalid')
    headers = {}
    while True:
        line = await reader.readline()
        size += len(line)
        if size > MAX_HEADERS:
            raise ReplayError('http_headers_too_large')
        if line == b'\r\n':
            return int(parts[1]), headers
        if not line or not line.endswith(b'\r\n') or b':' not in line:
            raise ReplayError('http_headers_invalid')
        name, value = line[:-2].split(b':', 1)
        name = name.lower()
        if not name or any(c <= 32 or c >= 127 for c in name):
            raise ReplayError('http_headers_invalid')
        # Repeated framing/interpretation headers are ambiguous and rejected.
        if name in headers and name in (b'content-length', b'transfer-encoding',
                                       b'content-type', b'content-encoding'):
            raise ReplayError('http_headers_duplicate')
        headers[name] = value.strip()


async def body_chunks(reader, headers):
    transfer = headers.get(b'transfer-encoding')
    length = headers.get(b'content-length')
    encoding = headers.get(b'content-encoding', b'identity').lower()
    if encoding != b'identity':
        raise ReplayError('http_content_encoding_unsupported')
    if transfer is not None:
        if transfer.lower() != b'chunked' or length is not None:
            raise ReplayError('http_framing_invalid')
        total = 0
        while True:
            line = await reader.readline()
            token = line.split(b';', 1)[0].strip()
            if (not line.endswith(b'\r\n') or not token
                    or any(c not in b'0123456789abcdefABCDEF' for c in token)):
                raise ReplayError('http_chunk_invalid')
            size = int(token, 16)
            total += size
            if total > MAX_BODY:
                raise ReplayError('http_body_too_large')
            if size == 0:
                trailer_bytes = 0
                while True:
                    trailer = await reader.readline()
                    trailer_bytes += len(trailer)
                    if trailer_bytes > MAX_HEADERS:
                        raise ReplayError('http_trailers_too_large')
                    if trailer == b'\r\n':
                        return
                    if not trailer.endswith(b'\r\n') or b':' not in trailer:
                        raise ReplayError('http_trailers_invalid')
            remaining = size
            while remaining:
                chunk = await reader.read(min(remaining, 65536))
                if not chunk:
                    raise ReplayError('http_body_truncated')
                remaining -= len(chunk)
                yield chunk
            if await reader.readexactly(2) != b'\r\n':
                raise ReplayError('http_chunk_invalid')
    elif length is not None:
        if not length.isdigit() or int(length) > MAX_BODY:
            raise ReplayError('http_content_length_invalid')
        remaining = int(length)
        while remaining:
            chunk = await reader.read(min(remaining, 65536))
            if not chunk:
                raise ReplayError('http_body_truncated')
            remaining -= len(chunk)
            yield chunk
    else:
        total = 0
        while chunk := await reader.read(65536):
            total += len(chunk)
            if total > MAX_BODY:
                raise ReplayError('http_body_too_large')
            yield chunk


class SSEState:
    """Count-only SSE state. Generated text is observed then discarded."""

    def __init__(self):
        self.buffer = b''
        self.data = []
        self.event_bytes = 0
        self.done = False
        self.finish_reason = None
        self.usage = None
        self.first_nonempty_at = None
        self.events = 0

    def feed(self, chunk, now):
        self.buffer += chunk
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            line = line.removesuffix(b'\r')
            self.event_bytes += len(line)
            if self.event_bytes > MAX_EVENT:
                raise ReplayError('sse_event_too_large')
            if not line:
                if self.data:
                    self.consume(b'\n'.join(self.data), now)
                self.data = []
                self.event_bytes = 0
            elif line.startswith(b'data:'):
                self.data.append(line[5:].removeprefix(b' '))
        if len(self.buffer) + self.event_bytes > MAX_EVENT:
            raise ReplayError('sse_event_too_large')

    def consume(self, data, now):
        if self.done:
            raise ReplayError('sse_data_after_done')
        if data == b'[DONE]':
            self.done = True
            return
        try:
            row = json.loads(data.decode('utf-8'))
        except (ValueError, UnicodeError):
            raise ReplayError('sse_json_invalid') from None
        if not isinstance(row, dict):
            raise ReplayError('sse_object_invalid')
        if 'error' in row:
            raise ReplayError('server_stream_error')
        self.events += 1
        choices = row.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ReplayError('sse_choices_invalid')
        for choice in choices:
            if (not isinstance(choice, dict) or type(choice.get('index')) is not int
                    or choice['index'] != 0):
                raise ReplayError('sse_choice_invalid')
            text = choice.get('text', '')
            if not isinstance(text, str):
                raise ReplayError('sse_text_invalid')
            if text:
                if self.finish_reason is not None:
                    raise ReplayError('sse_text_after_finish')
                if self.first_nonempty_at is None:
                    self.first_nonempty_at = now
            reason = choice.get('finish_reason')
            if reason is not None:
                if self.finish_reason is not None:
                    raise ReplayError('sse_duplicate_finish')
                # Never persist arbitrary server strings.
                self.finish_reason = reason if reason in ('length', 'stop') else 'other'
        usage = row.get('usage')
        if usage is not None:
            if not isinstance(usage, dict):
                raise ReplayError('usage_invalid')
            safe = {key: usage.get(key) for key in
                    ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            if any(value is not None and (type(value) is not int or value < 0)
                   for value in safe.values()):
                raise ReplayError('usage_count_invalid')
            if self.usage is not None and self.usage != safe:
                raise ReplayError('usage_conflicting')
            self.usage = safe

    def violations(self, event):
        errors = []
        if self.buffer or self.data:
            errors.append('sse_unterminated_event')
        if not self.done:
            errors.append('stream_not_complete')
        if self.finish_reason != 'length':
            errors.append('finish_reason_not_length')
        if self.usage is None:
            errors.append('missing_usage')
        else:
            if self.usage['prompt_tokens'] != event['prompt_tokens']:
                errors.append('prompt_tokens_mismatch')
            if self.usage['completion_tokens'] != event['output_tokens']:
                errors.append('output_tokens_mismatch')
            total = self.usage['total_tokens']
            if total is not None and total != event['prompt_tokens'] + event['output_tokens']:
                errors.append('total_tokens_mismatch')
        return errors


async def send_request(endpoint, request, header, origin, dispatched_at, timeout_s, request_id):
    loop = asyncio.get_running_loop()
    started = loop.time()
    event = request.event
    state = SSEState()
    writer = None
    row = dict(request=event['request'], request_id=request_id, warmup=event['warmup'],
               scheduled_offset_s=event['offset_s'],
               dispatched_offset_s=dispatched_at - origin,
               started_offset_s=started - origin,
               schedule_lateness_s=max(0., started - origin - event['offset_s']),
               expected_prompt_tokens=event['prompt_tokens'],
               expected_output_tokens=event['output_tokens'], http_status=None,
               connect_s=None, request_write_s=None, client_ttft_s=None,
               client_first_nonempty_delta_available=False, error=None)
    try:
        async with asyncio.timeout(timeout_s):
            reader, writer = await asyncio.open_connection(endpoint.host, endpoint.port,
                                                          ssl=endpoint.tls, limit=MAX_HEADERS)
            row['connect_s'] = loop.time() - started
            writer.write(header)
            for part in request.body_parts:
                writer.write(part)
            await writer.drain()
            row['request_write_s'] = loop.time() - started
            status, headers = await read_headers(reader)
            row['http_status'] = status
            if status != 200:
                raise ReplayError('http_non_200')
            if headers.get(b'content-type', b'').split(b';', 1)[0].strip().lower() != b'text/event-stream':
                raise ReplayError('http_content_type_invalid')
            async for chunk in body_chunks(reader, headers):
                state.feed(chunk, loop.time())
    except ReplayError as exc:
        row['error'] = str(exc)
    except TimeoutError:
        row['error'] = 'request_timeout'
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        row['error'] = 'transport_failed'
    except (ValueError, UnicodeError, asyncio.LimitOverrunError):
        row['error'] = 'protocol_invalid'
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 1.)
            except (TimeoutError, ConnectionError, OSError):
                pass
    row['finished_offset_s'] = loop.time() - origin
    row['client_latency_s'] = row['finished_offset_s'] - row['started_offset_s']
    row['stream_done'] = state.done
    row['finish_reason'] = state.finish_reason
    row['usage'] = state.usage
    row['sse_events'] = state.events
    row['violations'] = state.violations(event)
    row['transport_ok'] = row['error'] is None
    if state.first_nonempty_at is not None:
        row['client_first_nonempty_delta_available'] = True
        row['client_ttft_s'] = state.first_nonempty_at - started
    row['ok'] = row['transport_ok'] and not row['violations']
    return row


async def replay(prepared, endpoint, timeout_s, on_record=None, on_start=None):
    """Fixed arrivals, unlimited scheduled concurrency, no retries; drain all tasks."""
    if not finite_number(timeout_s) or timeout_s <= 0:
        raise ReplayError('invalid_request_timeout')
    run_nonce = uuid.uuid4().hex
    request_ids = ['req-' + run_nonce + '-' + str(r.event['request']) for r in prepared]
    headers = [endpoint.header_prefix + b'X-Request-ID: ' + request_id.encode('ascii')
               + b'\r\nContent-Length: ' + str(r.body_bytes).encode() + b'\r\n\r\n'
               for r, request_id in zip(prepared, request_ids)]
    loop = asyncio.get_running_loop()
    tasks = []
    resolution_ns = math.ceil(max(time.get_clock_info('time').resolution,
                                  time.get_clock_info('monotonic').resolution) * 1e9)
    before = loop.time()
    unix_ns = time.time_ns()
    after = loop.time()
    origin = (before + after) / 2
    origin_metadata = dict(run_nonce=run_nonce, request_id_format='req-<run_nonce>-<synthetic_request_index>',
                           loop_origin_unix_ns=unix_ns, loop_origin_monotonic_s=origin,
                           clock_anchor_uncertainty_ns=math.ceil((after - before) * 1e9 / 2) + resolution_ns,
                           clock_anchor_uncertainty_scope='sampling bracket and reported clock resolution; excludes wall clock accuracy and interhost skew')
    if on_start is not None:
        # Persist the sampled anchor before any request is sent. Callback time
        # contributes to reported scheduling lateness; it never shifts arrivals.
        on_start(origin_metadata)
    record_write_failed = False

    async def send_and_record(request, header, request_id, dispatched_at):
        nonlocal record_write_failed
        row = await send_request(endpoint, request, header, origin, dispatched_at, timeout_s, request_id)
        if on_record is not None and not record_write_failed:
            try:
                on_record(row)
            except (OSError, ValueError, TypeError):
                # Preserve the arrival schedule and finish outstanding client tasks
                # even when result storage fails; leave RUN.json incomplete.
                record_write_failed = True
        return row

    for request, header, request_id in zip(prepared, headers, request_ids):
        due = origin + request.event['offset_s']
        if due > loop.time():
            await asyncio.sleep(due - loop.time())
        tasks.append(asyncio.create_task(send_and_record(request, header, request_id, loop.time())))
        # Permit requests with identical arrival times to start promptly.
        await asyncio.sleep(0)
    rows = await asyncio.gather(*tasks)
    if record_write_failed:
        raise ReplayError('client_record_write_failed')
    return rows, loop.time() - origin


def summary(workload, rows, elapsed_s, encoding_s, prepared, smoke=False, routing=None):
    measurement = [row for row in rows if not row['warmup']]
    all_successful = len(rows) == len(prepared) and all(row['ok'] for row in rows)
    return dict(schema=SCHEMA, status='complete',
                run_kind='worker_partition' if routing else ('smoke' if smoke else 'full'),
                routing=routing,
                created_utc=datetime.now(timezone.utc).isoformat(),
                workload=dict(requests=len(rows), duration_s=.4 if smoke else workload.manifest['duration_s'],
                              measurement_start_s=0 if smoke else workload.manifest['measurement_start_s']),
                protocol=dict(fixed_arrivals=True, fixed_output_lengths=True,
                              token_id_inputs=True, temperature=0, ignore_eos=True,
                              retries=0, abort_on_request_error=False,
                              all_client_tasks_finished=len(rows) == len(prepared),
                              server_requests_completed_confirmed=all_successful,
                              server_state_after_client_failure='no_client_failure' if all_successful else 'unknown',
                              backend_idle_confirmation_required_before_next_run_or_shutdown=True),
                status_definition='complete means client attempt sequence finished, regardless of request success',
                records_persistence='incremental JSONL, flushed after each client task finishes',
                metrics_scope='client_only; not equivalent to historical complete diagnostics',
                client_ttft_definition='request start (before connect) to first nonempty SSE text',
                unavailable_metrics=['frontend_ttft', 'worker_scheduler_counters',
                                     'native_retractions', 'prefix_cache_hits', 'gpu_metrics'],
                preencoding_s=encoding_s,
                encoded_request_bytes=sum(r.body_bytes for r in prepared),
                elapsed_s=elapsed_s, requests=len(rows),
                succeeded=sum(row['ok'] for row in rows),
                failed=sum(not row['ok'] for row in rows),
                measurement_requests=len(measurement),
                measurement_succeeded=sum(row['ok'] for row in measurement),
                measurement_failed=sum(not row['ok'] for row in measurement),
                max_schedule_lateness_s=max((r['schedule_lateness_s'] for r in rows), default=0.))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify', action='store_true', help='CPU only: validate and preencode; no HTTP')
    mode.add_argument('--run-http', action='store_true', help='send every synthetic request and drain')
    parser.add_argument('--url', help='full HTTP(S) completions URL')
    parser.add_argument('--model', help='explicit serving model alias; token IDs are never retokenized')
    parser.add_argument('--output-dir', type=Path, help='new directory for count/timing-only results')
    parser.add_argument('--api-key-env', help='environment variable containing a bearer token')
    parser.add_argument('--smoke', action='store_true',
                        help='separate 4-request smoke: up to 1024 generated input tokens / 8 output')
    parser.add_argument('--routing', type=Path, help='separate complete synthetic-worker-routing-v1 JSON')
    parser.add_argument('--worker', type=int, choices=range(4),
                        help='select this saved worker partition; preserves original arrival offsets')
    parser.add_argument('--request-timeout-s', type=float, default=3600.,
                        help='per-request timeout, including connect/upload/drain (default: 3600)')
    args = parser.parse_args(argv)
    if args.run_http and (not args.url or not args.model or args.output_dir is None):
        parser.error('--run-http requires --url, --model, and --output-dir')
    if not math.isfinite(args.request_timeout_s) or args.request_timeout_s <= 0:
        parser.error('--request-timeout-s must be positive and finite')
    if (args.routing is None) != (args.worker is None):
        parser.error('--routing and --worker must be supplied together')
    if args.smoke and args.routing is not None:
        parser.error('--smoke and --routing are separate workloads; choose one')
    try:
        workload = Workload(args.bundle)
        start = time.monotonic()
        prepare = prepare_smoke if args.smoke else prepare_requests
        prepared = prepare(workload, args.model or workload.manifest['model'])
        routing = None
        if args.routing is not None:
            prepared, routing = select_worker(workload, prepared, args.routing, args.worker)
        run_kind = 'worker_partition' if routing else ('smoke' if args.smoke else 'full')
        encoding_s = time.monotonic() - start
        if args.verify:
            print(json.dumps(dict(schema=SCHEMA, mode='verify', network_used=False,
                                  run_kind=run_kind, routing=routing,
                                  requests=len(prepared),
                                  prompt_tokens=sum(r.event['prompt_tokens'] for r in prepared),
                                  output_tokens=sum(r.event['output_tokens'] for r in prepared),
                                  encoded_request_bytes=sum(r.body_bytes for r in prepared),
                                  preencoding_s=encoding_s)))
            return 0
        api_key = None
        if args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
            if api_key is None:
                raise ReplayError('api_key_environment_missing')
        endpoint = make_endpoint(args.url, api_key)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = args.output_dir / 'RUN.json'
        initial_manifest = dict(schema=SCHEMA, status='running', run_kind=run_kind, routing=routing,
                                records_persistence='incremental JSONL', requests=len(prepared))
        manifest_path.write_text(json.dumps(initial_manifest, indent=2) + '\n')
        origin_metadata = {}

        def record_origin(metadata):
            origin_metadata.update(metadata)
            initial_manifest.update(metadata)
            manifest_path.write_text(json.dumps(initial_manifest, indent=2, allow_nan=False) + '\n')

        with (args.output_dir / 'client_records.jsonl').open('x') as stream:
            def persist(row):
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
            rows, elapsed_s = asyncio.run(replay(prepared, endpoint, args.request_timeout_s,
                                                 on_record=persist, on_start=record_origin))
        report = summary(workload, rows, elapsed_s, encoding_s, prepared, args.smoke, routing)
        report.update(origin_metadata)
        report['request_timeout_s'] = args.request_timeout_s
        manifest_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, allow_nan=False))
        return 0 if report['failed'] == 0 else 1
    except ReplayError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, KeyError, IndexError, TypeError, OSError):
        # Never echo file paths, model identifiers, response text, URL, or credentials.
        print('replay_setup_or_output_failed', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('replay_interrupted; RUN.json remains incomplete', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
