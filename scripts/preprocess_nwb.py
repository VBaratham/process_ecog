from __future__ import print_function

import argparse
import h5py
import time
import os

import numpy as np

try:
    from tqdm import tqdm
except:
    def tqdm(x, *args, **kwargs):
        return x

from ecog.signal_processing import resample
from ecog.signal_processing import subtract_CAR
from ecog.signal_processing import linenoise_notch
from ecog.signal_processing import hilbert_transform
from ecog.signal_processing import gaussian
from ecog.utils import load_bad_electrodes, bands

def _load_data(datafilename, acq_field):
    with h5py.File(datafilename, 'r') as datafile:
        # TODO: take depth electrodes data
        acq_fieldname = 'acquisition/Raw/{}'.format(acq_field)
        X = datafile['{}/data'.format(acq_fieldname)][:].T
        freq = datafile['{}/starting_time'.format(acq_fieldname)].attrs['rate']
        unit = datafile['{}/starting_time'.format(acq_fieldname)].attrs['unit']

        assert unit.lower() == "seconds"

    return X, freq

def _resample(X, new_freq, old_freq):
    if not np.allclose(new_freq, old_freq):
        assert new_freq < old_freq
        X = resample(X, new_freq, old_freq)
    return X

def _notch_filter(X, rate):
    return linenoise_notch(X, rate)

def _subtract_CAR(X):
    return subtract_CAR(X)

def _hilbert_transform(X, rate, cfs, sds):
    Y = np.zeros(shape=(len(cfs), X.shape[0], X.shape[1]), dtype=np.complex)
    for i, (cf, sd) in enumerate(tqdm(zip(cfs, sds),
                                      'Applying Hilbert transform',
                                      total=len(cfs))):
        kernel = gaussian(X, rate, cf, sd)
        Y[i] = hilbert_transform(X, rate, kernel)

    return Y

def _write_data(datafilename, acq_field, Y, decomp_type, cfs, sds):
    with h5py.File(datafilename, 'r+') as datafile:
        # TODO: Get the dataset name right
        ds_group_name = 'processing/preprocessed/Hilb_ChangBands'
        if ds_group_name in datafile:
            del datafile[ds_group_name]
        ds_group = datafile.create_group(ds_group_name)
        ds_name = 'Hilb_{}'.format(acq_field)
        dset = ds_group.create_dataset(ds_name, data=Y)

        dset.dims[0].label = 'filter'
        dset.dims[1].label = 'channel'
        dset.dims[2].label = 'time'
        for val, name in ((cfs, 'filter_center'),
                          (sds, 'filter_sigma')):
            if name not in ds_group.keys():
                ds_group[name] = val
            dset.dims.create_scale(ds_group[name], name)
            dset.dims[0].attach_scale(ds_group[name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing ecog data from nwb.')
    parser.add_argument('datafile', type=str, help="Input/output .nwb file")
    parser.add_argument('--acq-field', type=str, default='ECoG128')
    parser.add_argument('-r', '--rate', type=float, default=400.,
                        help='Resample data to this rate.')
    parser.add_argument('--cfs', type=float, nargs='+', default=None,
                        help="Center frequency of the Gaussian filter")
    parser.add_argument('--sds', type=float, nargs='+', default=None,
                        help="Standard deviation of the Gaussian filter")
    parser.add_argument('--no-notch', default=False, action='store_true',
                        help="Do not perform notch filtering")
    parser.add_argument('--no-car', default=False, action='store_true',
                        help="Do not perform common avg reference subtraction")
    parser.add_argument('--decomp-type', type=str, default='hilbert',
                        choices=['hilbert', 'hil'],
                        help="frequency decomposition method")
    args = parser.parse_args()

    # PARSE ARGS
    if args.cfs:
        cfs = args.cfs
    else:
        cfs = bands.chang_lab['cfs']

    if args.sds:
        sds = args.sds
    else:
        sds = bands.chang_lab['sds']

    assert len(cfs) == len(sds)

    if args.decomp_type in ('hilbert', 'hil'):
        decomp_type = 'hilbert'
    else:
        raise NotImplementedError()

    # LOAD DATA
    start = time.time()
    X, freq = _load_data(args.datafile, args.acq_field)
    print("Time to load {}: {} sec".format(args.datafile, time.time()-start))

    # RESAMPLE
    start = time.time()
    X = _resample(X, args.rate, freq)
    print("Time to resample: {} sec".format(time.time()-start))

    # TODO: remove bad electrodes

    # CAR REMOVAL
    if not args.no_car:
        start = time.time()
        X = _subtract_CAR(X, args.rate)
        print("Time to subtract CAR: {} sec".format(time.time()-start))

    # NOTCH FILTER
    if not args.no_notch:
        start = time.time()
        X = _notch_filter(X)
        print("Time to notch filter: {} sec".format(time.time()-start))

    # FREQUENCY DECOMPOSITION
    if decomp_type == 'hilbert':
        start = time.time()
        Y = _hilbert_transform(X, args.rate, cfs, sds)
        print("Time to Hilbert transform: {} sec".format(time.time()-start))
    else:
        raise NotImplementedError()

    # WRITE DATA
    start = time.time()
    _write_data(args.datafile, args.acq_field, Y, decomp_type, cfs, sds)
    print("Time to write {}: {} sec".format(args.datafile, time.time()-start))
