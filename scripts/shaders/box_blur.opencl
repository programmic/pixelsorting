__kernel void box_blur(__global const uchar *input, __global uchar *output,
                       const int width, const int height, const int channels,
                       const int blur_kernel) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= width || y >= height)
    return;

  for (int c = 0; c < channels; c++) {
    int sum = 0;
    int count = 0;

    for (int ky = -blur_kernel; ky <= blur_kernel; ky++) {
      for (int kx = -blur_kernel; kx <= blur_kernel; kx++) {
        int nx = clamp(x + kx, 0, width - 1);
        int ny = clamp(y + ky, 0, height - 1);
        int idx = (ny * width + nx) * channels + c;
        sum += input[idx];
        count++;
      }
    }

    int out_idx = (y * width + x) * channels + c;
    output[out_idx] = sum / count;
  }
}
