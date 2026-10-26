"""Group recorded capture failures with evidence; no native replay is required."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import numpy as np


def diagnose(capture, evaluation, result, latency_ms=100.):
    reasons=set(evaluation.get('reasons',[]))
    labels=[]
    evidence=[]
    missing=sorted(set(evaluation.get('expected_ids',[]))-set(result.get('detected_ids',[])))
    if 'pipeline_error' in reasons:
        labels.append('pipeline_error');evidence.append(result.get('pipeline_error','Pipeline returned an error'))
    if 'unexpected_tag' in reasons:
        labels.append('false_detection');evidence.append(f"Unexpected tag IDs: {evaluation.get('unexpected_ids',[])}")
    if 'nonfinite_or_invalid_pose' in reasons:labels.append('invalid_pose')
    if missing and reasons:
        labels.append('missed_tags');evidence.append(f'Clearly visible tags not detected: {missing}')
    candidates=result.get('pose_candidates',[])
    ambiguous=False
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            if a.get('id')!=b.get('id'):continue
            costs=[a.get('reprojection_rmse_px'),b.get('reprojection_rmse_px')]
            if any(v is None or not math.isfinite(v) for v in costs) or abs(costs[0]-costs[1])>.25:continue
            pa,pb=np.asarray(a.get('camera_position',[])),np.asarray(b.get('camera_position',[]))
            ra,rb=np.asarray(a.get('camera_rotation',[])),np.asarray(b.get('camera_rotation',[]))
            separation=float(np.linalg.norm(pa-pb)) if pa.shape==pb.shape==(3,) else 0.
            angle=math.degrees(math.acos(float(np.clip((np.trace(ra.T@rb)-1)/2,-1,1)))) if ra.shape==rb.shape==(3,3) else 0.
            if separation>.2 or angle>5:
                ambiguous=True
                evidence.append(f"Suspected pose ambiguity: tag {a.get('id')}, similar reprojection costs, {separation:.2f} m / {angle:.1f} deg candidate separation")
                break
        if ambiguous:break
    if ambiguous and reasons:labels.append('ambiguous_pose')
    if reasons & {'heading_error','tilt_error'}:
        labels.append('incorrect_rotation')
        evidence.append(f"Heading error: {evaluation.get('heading_error_deg')} deg; tilt error: {evaluation.get('tilt_error_deg')} deg")
    if 'position_error' in reasons:labels.append('position_error')
    if 'missing_pose_with_visible_tags' in reasons and not missing:labels.append('pose_rejected')
    timing=capture.get('frame',{}).get('result',{})
    timing_values=[timing.get('render_ms'),timing.get('vortex_ms'),timing.get('sensor',{}).get('injected_latency_ms',0.)]
    total=None
    if all(isinstance(v,(int,float)) and math.isfinite(v) for v in timing_values):
        total=sum(timing_values)
        if total>latency_ms:
            labels.append('latency')
            evidence.append(f'Original capture processing + injected delay: {total:.1f} ms (threshold {latency_ms:g} ms); not proof of an accuracy failure')
    experiment=capture.get('category')=='configuration_experiment'
    primary='calibration_experiment' if experiment else 'resolved' if not reasons else labels[0] if labels else 'unclassified'
    return dict(primary=primary,labels=labels,evidence=evidence,recorded_latency_ms=total,
                latency_threshold_ms=latency_ms,experiment=experiment,
                routine=capture.get('frame',{}).get('result',{}).get('routine'))


def summarize(cases):
    groups={}
    for case in cases:
        analysis=case['analysis']
        group=groups.setdefault(analysis['primary'],dict(count=0,cases=[],indicators={}))
        group['count']+=1;group['cases'].append(case['case_id'])
        group['indicators']=dict(Counter(group['indicators'])+Counter(analysis['labels']))
    return groups


def write_markdown(report,path):
    rows=['# Capture failure groups','',report.get('analysis_source','Current Vortex replay.'),'',
          'Groups are diagnostic triage, not confirmed root causes. Latency is recorded capture timing, not current replay timing.',
          'Calibration experiments are separate from zero-error regressions.','',
          '| Primary group | Cases |','|---|---:|']
    for name,group in sorted(report['failure_groups'].items(),key=lambda item:-item[1]['count']):
        rows.append(f"| {name} | {group['count']} |")
    for name,group in sorted(report['failure_groups'].items(),key=lambda item:-item[1]['count']):
        rows.extend(['',f'## {name}',''])
        cases=sorted((c for c in report['cases'] if c['analysis']['primary']==name),
                     key=lambda c:c['evaluation'].get('severity',0),reverse=True)
        for case in cases:
            folder=Path(case['case'])
            def link(filename):
                target=(folder/filename).resolve()
                try:return Path(os.path.relpath(target,path.parent.resolve())).as_posix()
                except ValueError:return target.as_uri()  # Reports on another Windows drive.
            metadata=link('capture.json')
            pixels=link('image.png')
            rows.append(f"- [{case['case_id']}](<{metadata}>) · [image](<{pixels}>) — {', '.join(case['evaluation'].get('reasons',[])) or 'resolved'}")
            rows.extend(f'  - {item}' for item in case['analysis']['evidence'])
            if case['analysis']['routine']:rows.append(f"  - Routine: {case['analysis']['routine']}")
    if report.get('errors'):
        rows.extend(['','## Unreadable captures',''])
        rows.extend(f"- {item['case']}: {item['error']}" for item in report['errors'])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(rows)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path,nargs='?',default=Path(__file__).parent/'generated/captures')
    parser.add_argument('--output',type=Path,help='Report directory (default: PATH/analysis)')
    parser.add_argument('--latency-ms',type=float,default=100.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.latency_ms) or args.latency_ms<=0:parser.error('Latency threshold must be positive')
    root=args.path.resolve()
    if not root.is_dir():parser.error('Capture directory must exist')
    paths=[root/'capture.json'] if (root/'capture.json').is_file() else sorted(root.rglob('capture.json'))
    output=args.output or root/'analysis'
    report=dict(analysis_source='Recorded capture failures; current Vortex has not been replayed.',cases=[],errors=[])
    for path in paths:
        try:
            capture=json.loads(path.read_text(encoding='utf-8'))
            evaluation=capture['failure'];result=capture['frame']['result']
            report['cases'].append(dict(case=str(path.parent),case_id=path.parent.relative_to(root).as_posix(),
                evaluation=evaluation,analysis=diagnose(capture,evaluation,result,args.latency_ms)))
        except (OSError,ValueError,KeyError,TypeError) as error:
            report['errors'].append(dict(case=str(path.parent),error=str(error)))
    report['total']=len(report['cases']);report['failure_groups']=summarize(report['cases'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_markdown(report,output/'analysis.md')
    print(f"Grouped {report['total']} captures: "+', '.join(f"{name}={group['count']}" for name,group in report['failure_groups'].items()))
    print(f'Reports: {output/"analysis.md"} and analysis.json')
    return int(bool(report['errors']))


if __name__=='__main__':raise SystemExit(main())
