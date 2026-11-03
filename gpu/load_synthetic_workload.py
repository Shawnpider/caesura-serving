"""Portable stdlib-only reader for the compact, generated-token workload.

Example:
    python3 load_synthetic_workload.py /path/to/bundle --verify
For replay, iterate ``Workload(path).requests()`` for event/payload pairs.
The caller schedules event['offset_s'] on an open-loop relative clock.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

FIELDS={'request','family','namespace','offset_s','warmup','prompt_tokens','output_tokens','node'}


class Workload:
    def __init__(self,path):
        path=Path(path).resolve()
        self.manifest=json.loads((path/'WORKLOAD.json').read_text())
        m=self.manifest
        if m.get('schema')!='content-independent-prefix-trie-compact-v1' or m.get('edge_file')!='edges.u32le':
            raise ValueError('unknown_bundle_format')
        self.edges=(path/'edges.u32le').read_bytes()
        if hashlib.sha256(self.edges).hexdigest()!=m['edge_sha256']:
            raise ValueError('edge_hash_mismatch')
        end=0
        for i,node in enumerate(m['nodes']):
            if set(node)!={'parent','offset','bytes','depth'} or any(type(v) is not int for v in node.values()):
                raise ValueError('node_fields_invalid')
            parent=node['parent']
            if not -1<=parent<i or node['offset']!=end or node['bytes']<0 or node['bytes']%4:
                raise ValueError('node_span_or_parent_invalid')
            depth=m['nodes'][parent]['depth'] if parent>=0 else 0
            if node['depth']!=depth+node['bytes']//4:raise ValueError('node_depth_invalid')
            end+=node['bytes']
        if end!=len(self.edges):raise ValueError('edge_coverage_incomplete')
        last=-1
        for i,event in enumerate(m['events']):
            if set(event)!=FIELDS or event['request']!=i or event['offset_s']<last:
                raise ValueError('event_fields_or_order_invalid')
            for key in ('family','namespace','node','prompt_tokens','output_tokens'):
                if type(event[key]) is not int or event[key]<0:raise ValueError('event_numeric_invalid')
            if event['node']>=len(m['nodes']) or m['nodes'][event['node']]['depth']!=event['prompt_tokens']:
                raise ValueError('request_length_invalid')
            if not 0<=event['offset_s']<m['duration_s']:raise ValueError('event_clock_invalid')
            if type(event['warmup']) is not bool or event['warmup']!=(event['offset_s']<m['measurement_start_s']):
                raise ValueError('warmup_partition_invalid')
            last=event['offset_s']
        if m['sampling']!={'temperature':0,'ignore_eos':True,'stream':True}:
            raise ValueError('sampling_contract_invalid')

    def token_bytes(self,node):
        chunks=[]
        while node!=-1:
            row=self.manifest['nodes'][node]
            chunks.append(self.edges[row['offset']:row['offset']+row['bytes']]);node=row['parent']
        return b''.join(reversed(chunks))

    def requests(self):
        for event in self.manifest['events']:
            ids=array('I');ids.frombytes(self.token_bytes(event['node']))
            if sys.byteorder!='little':ids.byteswap()
            payload=dict(model=self.manifest['model'],prompt=list(ids),max_tokens=event['output_tokens'],
                temperature=0,ignore_eos=True,stream=True,stream_options={'include_usage':True})
            if event['namespace']!=0:
                payload['cache_salt']='synthetic-namespace-'+str(event['namespace'])
            yield dict(event),payload

    def verify(self):
        digest=hashlib.sha256();total=output=0
        for event in self.manifest['events']:
            raw=self.token_bytes(event['node']);digest.update(raw);total+=len(raw)//4;output+=event['output_tokens']
        return dict(requests=len(self.manifest['events']),prompt_tokens=total,output_tokens=output,
            expanded_token_sha256=digest.hexdigest(),loaded_edge_bytes=len(self.edges))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--verify',action='store_true',required=True)
    args=parser.parse_args()
    print(json.dumps(Workload(args.bundle).verify()))


if __name__=='__main__':main()
