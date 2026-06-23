# Data Preparation

Data preparation follows https://github.com/eeeric-code/I3Net.

## Expected Training Format

`trainSet` expects `traindata_path` to point to a directory containing one subdirectory per volume:

```text
imagesTr/
  case_000/
    slice_000.pth
    slice_001.pth
    ...
```

Each slice file is loaded with `pickle.load` and should contain:

```python
{
    "image": numpy_array_or_array_like
}
```

The loader stacks adjacent slices, crops the center `256 x 256` region, normalizes by `data_range`, and samples the requested interpolation scale.

## Expected Test Format

`testSet` expects `testdata_path` to point to a directory containing test volumes. Each item can be:

- A `.pt` / `.pth` pickle file with an `image` field.
- A folder containing sorted `.pth` or `.npy` slice files.

Example:

```text
imagesTs/
  case_000.pth
  case_001.pth
```

or:

```text
imagesTs/
  case_000/
    slice_000.npy
    slice_001.npy
    ...
```

