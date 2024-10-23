# ImageAnalyst RKNN

ImageAnalyst RKNN is an extension to the [ImageAnalyst](https://github.com/BergLucas/ImageAnalyst) library, providing additional models and functions using [rknn-toolkit-lite2](https://github.com/airockchip/rknn-toolkit2).

## Requirements

The application requires:

- [Python](https://www.python.org/) >=3.9, <3.11
- [pip](https://pip.pypa.io/en/stable/)
- [librknnrt.so](https://github.com/airockchip/rknn-toolkit2/blob/7efa763d1da04c4c4447fa5632dc2d9c94fb3063/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so)

## Download & Installation

There is only one way to download and install the application.

### Using the GitHub releases

You can download the application on the [downloads page](https://github.com/BergLucas/ImageAnalystRKNN/releases). Then, you can install the application by running the following command:

```bash
pip install image_analyst_rknn-X.X.X-py3-none-any.whl
```

(Note: The X.X.X must be replaced by the version that you want to install.)

## License

All code is licensed for others under a MIT license (see [LICENSE](https://github.com/BergLucas/ImageAnalystRKNN/blob/main/LICENSE)).
