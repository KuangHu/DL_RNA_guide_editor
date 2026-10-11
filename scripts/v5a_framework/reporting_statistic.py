"""One reporting statistic and configuration identity for scan and nulls."""
from dataclasses import asdict
from hashlib import sha256
import json

STATISTIC_VERSION = 'verified_max_passing_mean_m_v1'


def effective_configuration(cfg) -> dict:
    value = asdict(cfg)
    if cfg.coordinate_mode == 'verified':
        value['bg']['method'] = 'gapped'
    return value


def config_fingerprint(cfg) -> str:
    return sha256(json.dumps(effective_configuration(cfg), sort_keys=True).encode()).hexdigest()


def bag_statistics(result) -> dict:
    rows = result.passing
    best = max(rows, key=lambda cb: (cb[0].mean_m_frac, cb[0].n_support,
                                    cb[1].n_local_bg_groups, -cb[0].flank_pos_mad)) if rows else None
    return dict(statistic_version=STATISTIC_VERSION,
                status=result.status, eligible=result.status == 'ready',
                n_passing=len(rows),
                max_mean_m=best[0].mean_m_frac if best else 0.0,
                max_n_support=max((c.n_support for c, _ in rows), default=0),
                max_n_bg_groups=max((b.n_local_bg_groups for _, b in rows), default=0),
                best_L=best[0].L if best else 0,
                best_m_at_top=best[0].mean_m_frac * best[0].L if best else 0.0)


def read_calibration(path, cfg) -> dict:
    with path.open(encoding='utf-8') as fh:
        value = json.load(fh)
    if value.get('statistic_version') != STATISTIC_VERSION or value.get('config_fingerprint') != config_fingerprint(cfg):
        raise ValueError('calibration statistic/configuration does not match this run')
    if value.get('status') != 'empirically_validated' or value.get('threshold') is None:
        raise ValueError('calibration has no validated threshold; inspect candidate evidence first')
    if cfg.coordinate_mode != 'verified' or not 0 <= value['threshold'] <= 1:
        raise ValueError('formal calibration requires verified coordinates and a valid threshold')
    return value
