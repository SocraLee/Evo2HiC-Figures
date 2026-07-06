from hic_utils import *

def find_closest_resolution(res, resolutions):
    closest_r = 1
    for r in resolutions:
        if res % r == 0 and r > closest_r:
            closest_r = r
    
    return closest_r

def read_hic(hic, resolution, format, chrname=None, chrid=None, L=None, norm = 'NONE') -> np.matrix|scipy.sparse.coo_matrix:
    r = find_closest_resolution(resolution, hic.getResolutions())
    if chrname is not None:
        ch = find_chr_in_hic(id2chr(chrname), hic)
        if ch is None:
            ch = find_chr_in_hic(id2chr(chrname)[3:], hic)
        assert ch is not None, f'chromosome {ch} doesn\'t exist.'

        mzd = hic.getMatrixZoomData(ch.name, ch.name, "observed", norm, "BP", r)
    elif chrid is not None:
        ch = hic.getChromosomes()[chrid]
        mzd = hic.getMatrixZoomData(ch.name, ch.name, "observed", norm, "BP", r)

    if L is None:
        L = ch.length

    records = mzd.getRecords(0, L, 0, L)
    X, Y, data = records2npy(records)
    del records
    if resolution != r:
        pool(X, Y, data, resolution)
    X, Y = X // resolution, Y // resolution

    if format == 'matrix':
        return hic2sparse(X, Y, data, (L-1)//resolution + 1).toarray()
    elif format == 'upper':
        return hic2upper(X, Y, data, (L-1)//resolution + 1)

def normalize_eval(data, clip, log):
    data = data.astype(np.int32)
    data = np.maximum(data, 0)
    if clip > 0:
        if log == 'No':
            data = np.minimum(data, clip)/clip
        elif log == 'Yes':
            data = np.log1p(np.minimum(data, clip)) / np.log1p(clip)
    else:
        if log == 'Yes':
            data = np.log1p(data)
                
    return data
