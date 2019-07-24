import h5py

if __name__ == '__main__':
    f = h5py.File('sample.hdf5')
    data = f['free_z_with_all_bands_and_uncertainties.png'][...]
    print(data.shape)